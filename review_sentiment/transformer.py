"""
RoBERTa, a BERT-style transformer, used in two ways:

- as a frozen encoder: each review becomes the mean of its contextual token vectors, and a
  logistic regression is trained on top (contextual embeddings)
- fine-tuned as a sentiment classifier: a new output layer and the top half of the
  transformer are trained on the reviews

The encoder and its tokenizer come from spaCy's en_core_web_trf pipeline, a RoBERTa-base model
that spaCy further trained for tagging, parsing and named entities. It installs with pip, so
the results can be reproduced exactly.
"""

import argparse
import copy
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

with warnings.catch_warnings():
    # curated-transformers uses torch.jit.script, which recent PyTorch versions flag as deprecated
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
    from curated_transformers.models.attention import AttentionMask

from .data import RESULTS_DIR

BOS, PAD, EOS = 0, 1, 2  # RoBERTa's <s>, <pad> and </s> ids
MAX_PIECES = 256  # longest input, <s> and </s> included
HIDDEN = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    """
    Load RoBERTa and its byte-level BPE tokenizer from spaCy's en_core_web_trf pipeline.

    Returns:
    tuple: The PyTorch encoder and the tokenizer.
    """
    from .preprocess import load_spacy

    nlp = load_spacy("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    model = nlp.get_pipe("transformer").model
    encoder = model.get_ref("transformer").shims[0]._model
    tokenizer = model.get_ref("piece_encoder").attrs["byte_bpe_processor"]
    return encoder, tokenizer


def encode(texts, tokenizer, max_pieces=MAX_PIECES):
    """
    Turn texts into RoBERTa token ids. A review longer than the limit keeps its beginning and its
    end, where reviewers tend to sum up their opinion, and loses the middle.

    Parameters:
    texts (list): Review texts.
    tokenizer: The byte-level BPE tokenizer.
    max_pieces (int): Maximum number of tokens, <s> and </s> included.

    Returns:
    list: One list of token ids per text.
    """
    budget = max_pieces - 2
    encoded = []
    for text in texts:
        ids = tokenizer.encode_as_ids(text)
        if len(ids) > budget:
            head = budget // 2
            ids = ids[:head] + ids[len(ids) - (budget - head):]
        encoded.append([BOS] + ids + [EOS])
    return encoded


def pad(batch):
    """Pad a batch of token id lists to the same length; returns ids and a mask of real tokens."""
    width = max(len(ids) for ids in batch)
    ids = torch.full((len(batch), width), PAD, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.bool)
    for row, piece_ids in enumerate(batch):
        ids[row, :len(piece_ids)] = torch.tensor(piece_ids)
        mask[row, :len(piece_ids)] = True
    return ids.to(DEVICE), mask.to(DEVICE)


def mean_pool(hidden, mask):
    """Average the token vectors of each review, ignoring padding."""
    weights = mask.unsqueeze(-1).float()
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1)


class SentimentClassifier(torch.nn.Module):
    """RoBERTa followed by dropout and a linear layer on the mean token vector."""

    def __init__(self, encoder, n_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(HIDDEN, n_classes)

    def forward(self, ids, mask):
        hidden = self.encoder(ids, AttentionMask(mask)).last_hidden_layer_states
        return self.head(self.dropout(mean_pool(hidden, mask)))


def batches_by_length(encoded, batch_size):
    """Indices grouped into batches of similar length, so that little time goes into padding."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


@torch.no_grad()
def embed(encoder, encoded, batch_size=16):
    """
    Contextual embedding of each review: the mean of RoBERTa's last-layer token vectors.

    Returns:
    ndarray: One 768-dimensional row per review, in the input order.
    """
    encoder.to(DEVICE).eval()
    vectors = np.zeros((len(encoded), HIDDEN), dtype=np.float32)
    for rows in batches_by_length(encoded, batch_size):
        ids, mask = pad([encoded[i] for i in rows])
        hidden = encoder(ids, AttentionMask(mask)).last_hidden_layer_states
        vectors[rows] = mean_pool(hidden, mask).cpu().numpy()
    return vectors


@torch.no_grad()
def predict_proba(model, encoded, batch_size=32):
    """Probability of a positive review for each input, in the input order."""
    model.eval()
    proba = np.zeros(len(encoded), dtype=np.float32)
    for rows in batches_by_length(encoded, batch_size):
        ids, mask = pad([encoded[i] for i in rows])
        proba[rows] = torch.softmax(model(ids, mask), dim=-1)[:, 1].cpu().numpy()
    return proba


def training_batches(encoded, batch_size, rng):
    """
    Shuffled training batches of similar-length reviews: reviews are shuffled, sorted by length
    within large pools, cut into batches, and the batches are shuffled again.
    """
    order = list(range(len(encoded)))
    rng.shuffle(order)
    pool = batch_size * 50
    batches = []
    for start in range(0, len(order), pool):
        chunk = sorted(order[start:start + pool], key=lambda i: len(encoded[i]))
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def fine_tune(encoder, train_encoded, train_labels, val_encoded, val_labels, epochs=3, lr=3e-5,
              batch_size=16, frozen_layers=6, weight_decay=0.01, warmup=0.1, seed=0, log=print):
    """
    Fine-tune RoBERTa as a sentiment classifier and keep the epoch with the best validation
    accuracy.

    To keep training time reasonable on a CPU, the embedding layer and the bottom half of the
    transformer (6 of 12 layers) are not updated. The learning rate warms up over the first 10%
    of the steps, then decays linearly to zero.

    Parameters:
    encoder: A fresh RoBERTa encoder (it is copied, not modified).
    train_encoded, train_labels: Training reviews (token ids) and labels.
    val_encoded, val_labels: Validation reviews and labels, used to choose the epoch.
    epochs (int): Passes over the training data.
    lr (float): Peak learning rate.
    batch_size (int): Reviews per update.
    frozen_layers (int): Number of bottom transformer layers kept frozen.
    weight_decay (float): AdamW weight decay, not applied to biases and layer norms.
    warmup (float): Share of the steps used to warm up the learning rate.
    seed (int): Seed for the head, dropout and batch order.
    log (callable): Receives one progress line per epoch.

    Returns:
    tuple: The fine-tuned classifier (best epoch) and the history, one dict per epoch.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SentimentClassifier(copy.deepcopy(encoder)).to(DEVICE)
    blocks = model.encoder.curated_encoder
    for parameter in list(blocks.embeddings.parameters()) + [
            p for layer in blocks.layers[:frozen_layers] for p in layer.parameters()]:
        parameter.requires_grad = False

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    no_decay = ("bias", "norm")
    groups = [
        {"params": [p for n, p in trainable if not any(k in n.lower() for k in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in trainable if any(k in n.lower() for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    steps = epochs * -(-len(train_encoded) // batch_size)
    warmup_steps = max(1, int(warmup * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup_steps, max(0.0, (steps - step) / (steps - warmup_steps))))

    labels = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)
    best_accuracy, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for rows in training_batches(train_encoded, batch_size, rng):
            ids, mask = pad([train_encoded[i] for i in rows])
            loss = torch.nn.functional.cross_entropy(model(ids, mask), labels[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item() * len(rows)
        val_accuracy = float(((predict_proba(model, val_encoded) > 0.5) == np.asarray(val_labels)).mean())
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_encoded),
                        "val_accuracy": val_accuracy, "minutes": (time.time() - start) / 60})
        log(f"epoch {epoch}: loss {history[-1]['train_loss']:.4f}, validation accuracy {val_accuracy:.4f}, "
            f"{history[-1]['minutes']:.1f} min")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


EMBEDDINGS = "RoBERTa embeddings"
FINE_TUNED = "Fine-tuned RoBERTa"


def predictions_frame(model_name, train_domain, test, proba):
    """Predictions of one model on the test reviews, in the format saved to results/."""
    return pd.DataFrame({"model": model_name, "train_domain": train_domain, "review": test.index,
                         "domain": test["domain"].values, "label": test["label"].values,
                         "proba_positive": np.round(proba, 6)})


def run(out_dir=RESULTS_DIR, epochs=3, log=print):
    """
    Run both transformer experiments: for each product category, train on its training reviews
    and save the probability of a positive review for the test reviews of all four categories.

    This is the slow part of the project. On a CPU the embeddings take about half an hour and
    each fine-tuning run between 15 and 30 minutes; on a GPU everything takes minutes.
    Categories already fine-tuned are skipped when the command is run again.

    Parameters:
    out_dir (str or Path): Folder for the predictions and training histories.
    epochs (int): Fine-tuning epochs.
    log (callable): Receives progress messages.
    """
    from .data import DOMAINS, add_split, full_text, load_reviews
    from .models import tuned_logistic_regression

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reviews = add_split(load_reviews())
    test = reviews[reviews["split"] == "test"]
    encoder, tokenizer = load_encoder()
    encoded = encode(full_text(reviews), tokenizer)
    log(f"{len(reviews)} reviews, {np.mean([len(ids) for ids in encoded]):.0f} tokens on average, device {DEVICE}")

    cache = out / "roberta_embeddings.npy"
    if cache.exists():
        vectors = np.load(cache)
    else:
        start = time.time()
        vectors = embed(encoder, encoded)
        np.save(cache, vectors)
        log(f"embeddings computed in {(time.time() - start) / 60:.1f} min")

    frames = []
    for domain in DOMAINS:
        train = reviews[(reviews["domain"] == domain) & (reviews["split"] == "train")]
        model = tuned_logistic_regression(scale=True).fit(vectors[train.index], train["label"])
        frames.append(predictions_frame(EMBEDDINGS, domain, test, model.predict_proba(vectors[test.index])[:, 1]))
        log(f"{EMBEDDINGS}, {domain}: C = {model.best_params_['logisticregression__C']}")
    pd.concat(frames).to_csv(out / "roberta_embeddings_predictions.csv", index=False)

    for domain in DOMAINS:
        path = out / f"finetuned_{domain}_predictions.csv"
        if path.exists():
            log(f"{FINE_TUNED}, {domain}: already done")
            continue
        train = reviews[(reviews["domain"] == domain) & (reviews["split"] == "train")]
        fit_rows, val_rows = train_test_split(train.index, test_size=0.1, stratify=train["label"], random_state=0)
        log(f"{FINE_TUNED}, {domain}: {len(fit_rows)} training and {len(val_rows)} validation reviews")
        model, history = fine_tune(encoder, [encoded[i] for i in fit_rows], reviews.loc[fit_rows, "label"].tolist(),
                                   [encoded[i] for i in val_rows], reviews.loc[val_rows, "label"].tolist(),
                                   epochs=epochs, log=lambda line: log(f"  {domain} {line}"))
        proba = predict_proba(model, [encoded[i] for i in test.index])
        (out / f"finetuned_{domain}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        predictions_frame(FINE_TUNED, domain, test, proba).to_csv(path, index=False)
        log(f"{FINE_TUNED}, {domain}: in-category test accuracy "
            f"{((proba[test['domain'] == domain] > 0.5) == test.loc[test['domain'] == domain, 'label']).mean():.4f}")
        del model


def main(argv=None):
    """Command line: python -m review_sentiment.transformer"""
    parser = argparse.ArgumentParser(prog="python -m review_sentiment.transformer",
                                     description="RoBERTa embeddings and fine-tuning (slow on a CPU)")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="folder for the predictions")
    parser.add_argument("--epochs", type=int, default=3, help="fine-tuning epochs")
    args = parser.parse_args(argv)
    run(args.out, args.epochs, log=lambda line: print(time.strftime("%H:%M:%S"), line, flush=True))


if __name__ == "__main__":
    main()
