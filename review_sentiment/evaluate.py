"""
Training every model on each product category and scoring it on the test reviews of all four.

Every model is trained four times, once per category, and each time predicts the test reviews
of all four categories. That gives a 4 x 4 accuracy matrix per model: the diagonal is the
in-category accuracy, the rest shows what happens when the category changes.
"""

import numpy as np
import pandas as pd

from .data import DOMAINS, full_text
from .features import average_embeddings, tfidf, vader_compound
from .models import RANDOM_STATE, ThresholdClassifier, tuned_logistic_regression

VADER = "VADER lexicon"
TFIDF = "TF-IDF"
GLOVE = "GloVe"
EMBEDDINGS = "RoBERTa embeddings"
FINE_TUNED = "Fine-tuned RoBERTa"
MODELS = [VADER, TFIDF, GLOVE, EMBEDDINGS, FINE_TUNED]
DESCRIPTIONS = {
    VADER: "Rule-based sentiment lexicon, cut-off chosen on the training reviews",
    TFIDF: "Lemmas with negation marking, unigrams and bigrams, logistic regression",
    GLOVE: "Average of pretrained GloVe word vectors, logistic regression",
    EMBEDDINGS: "Average of frozen RoBERTa token vectors, logistic regression",
    FINE_TUNED: "RoBERTa fine-tuned as a classifier, top 6 of its 12 layers trained",
}


def predictions(model, train_domain, test, score, prediction):
    """Predictions of one model on the test reviews, one row per review."""
    return pd.DataFrame({"model": model, "train_domain": train_domain, "review": test.index,
                         "domain": test["domain"].values, "label": test["label"].values,
                         "score": np.asarray(score, dtype=float), "prediction": np.asarray(prediction, dtype=int)})


def run_classic_models(reviews, processed, glove_vectors, tfidf_column="lemmas_negation", ngram_range=(1, 2)):
    """
    Train the lexicon, TF-IDF and GloVe models on each category and predict all test reviews.

    Parameters:
    reviews (DataFrame): Reviews with a "split" column (data.add_split).
    processed (DataFrame): Output of preprocess.process for the same reviews.
    glove_vectors: Pretrained word vectors (features.load_glove).
    tfidf_column (str): Text column of `processed` used by TF-IDF.
    ngram_range (tuple): N-grams used by TF-IDF.

    Returns:
    DataFrame: Predictions of the three models, in the format of predictions().
    """
    test = reviews[reviews["split"] == "test"]
    vader = pd.Series(vader_compound(full_text(reviews)), index=reviews.index)
    glove = average_embeddings(processed["tokens"].tolist(), glove_vectors)
    frames = []
    for domain in DOMAINS:
        train = reviews[(reviews["domain"] == domain) & (reviews["split"] == "train")]
        cut = ThresholdClassifier().fit(vader[train.index], train["label"])
        frames.append(predictions(VADER, domain, test, vader[test.index], cut.predict(vader[test.index])))

        text = processed[tfidf_column]
        model = tuned_logistic_regression(tfidf(ngram_range)).fit(text[train.index], train["label"])
        proba = model.predict_proba(text[test.index])[:, 1]
        frames.append(predictions(TFIDF, domain, test, proba, proba > 0.5))

        model = tuned_logistic_regression(scale=True).fit(glove[train.index], train["label"])
        proba = model.predict_proba(glove[test.index])[:, 1]
        frames.append(predictions(GLOVE, domain, test, proba, proba > 0.5))
    return pd.concat(frames, ignore_index=True)


def load_transformer_predictions(results_dir):
    """
    Read the predictions saved by `python -m review_sentiment.transformer`.

    Returns:
    DataFrame: Predictions of the two RoBERTa models, in the format of predictions().
    """
    frames = []
    for path in sorted(results_dir.glob("*_predictions.csv")):
        saved = pd.read_csv(path)
        frames.append(saved.rename(columns={"proba_positive": "score"}).assign(
            prediction=lambda f: (f["score"] > 0.5).astype(int)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def accuracy_matrix(preds, model):
    """
    Accuracy of one model for every pair of training category (rows) and test category (columns).

    Returns:
    DataFrame: 4 x 4 accuracies.
    """
    rows = preds[preds["model"] == model]
    correct = rows.assign(correct=rows["prediction"] == rows["label"])
    return correct.pivot_table(index="train_domain", columns="domain", values="correct", aggfunc="mean").reindex(
        index=DOMAINS, columns=DOMAINS)


def summary(preds, models=MODELS):
    """
    In-category accuracy for each category and on average, the average accuracy across
    categories, and the drop between the two.

    Returns:
    DataFrame: One row per model.
    """
    rows = {}
    for model in models:
        if model not in set(preds["model"]):
            continue
        matrix = accuracy_matrix(preds, model)
        diagonal = pd.Series(np.diag(matrix), index=DOMAINS)
        off_diagonal = matrix.values[~np.eye(len(DOMAINS), dtype=bool)]
        rows[model] = {**diagonal.to_dict(), "in-category": diagonal.mean(),
                       "cross-category": off_diagonal.mean(), "drop": diagonal.mean() - off_diagonal.mean()}
    return pd.DataFrame(rows).T


def in_domain(preds, model):
    """The predictions of a model on test reviews of the category it was trained on."""
    rows = preds[preds["model"] == model]
    return rows[rows["train_domain"] == rows["domain"]]


def bootstrap_ci(preds, model, n_resamples=2000, level=0.95, random_state=RANDOM_STATE):
    """
    95% confidence interval of the average in-category accuracy: the test reviews of each
    category are resampled with replacement many times and the accuracy recomputed.

    Returns:
    tuple: Lower and upper bound.
    """
    rng = np.random.default_rng(random_state)
    rows = in_domain(preds, model)
    correct = {domain: (group["prediction"] == group["label"]).to_numpy() for domain, group in rows.groupby("domain")}
    scores = [np.mean([hits[rng.integers(0, len(hits), len(hits))].mean() for hits in correct.values()])
              for _ in range(n_resamples)]
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(scores, [tail, 100 - tail])
    return float(low), float(high)
