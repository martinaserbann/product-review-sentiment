"""
Linguistic preprocessing with spaCy: tokenization, lemmatization, part-of-speech tags and
negation scope.

Negation flips sentiment ("not good"), but a bag of words sees "good" either way. Following
Das and Chen (2001) and Pang et al. (2002), every word between a negation and the next
punctuation mark gets a NOT_ prefix, so "not good at all." becomes "not NOT_good NOT_at NOT_all".
"""

import warnings
from pathlib import Path

import pandas as pd

from .data import DATA_DIR, full_text

CACHE = DATA_DIR / "processed" / "linguistic.pkl"
NEGATION_WORDS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "nowhere",
                  "without", "hardly", "cannot"}
POS_COUNTED = ["ADJ", "ADV", "VERB", "NOUN"]


def load_spacy(name, **kwargs):
    """
    Load a spaCy pipeline. Loading also imports spaCy's transformer plugin, which calls
    torch.jit.script; recent PyTorch versions flag that as deprecated, a warning about the
    plugin rather than this project, so it is silenced here.
    """
    import spacy

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
        return spacy.load(name, **kwargs)


def load_nlp():
    """spaCy's small English pipeline: tokenizer, tagger, parser and lemmatizer (no entities)."""
    return load_spacy("en_core_web_sm", disable=["ner"])


def is_negation(token):
    """A negation, found by the parser (dependency "neg", as in "n't") or in a fixed word list."""
    return token.dep_ == "neg" or token.lower_ in NEGATION_WORDS


def analyse(doc):
    """
    Turn one parsed review into the features used by the models and the analysis.

    Parameters:
    doc (spacy.tokens.Doc): The parsed review.

    Returns:
    dict: Lowercase tokens and lemmas, each also with negation scope marked (joined by spaces),
    and counts of words, sentences, negations, exclamation marks and parts of speech.
    """
    tokens, lemmas, marked_tokens, marked_lemmas = [], [], [], []
    negated = False
    for token in doc:
        if token.is_space:
            continue
        if token.is_punct:
            negated = False
            continue
        lemma = token.lemma_.lower()
        tokens.append(token.lower_)
        lemmas.append(lemma)
        prefix = "NOT_" if negated and not is_negation(token) else ""
        marked_tokens.append(prefix + token.lower_)
        marked_lemmas.append(prefix + lemma)
        if is_negation(token):
            negated = True
    counts = {f"n_{pos.lower()}": sum(t.pos_ == pos for t in doc) for pos in POS_COUNTED}
    return {
        "tokens": " ".join(tokens),
        "lemmas": " ".join(lemmas),
        "tokens_negation": " ".join(marked_tokens),
        "lemmas_negation": " ".join(marked_lemmas),
        "n_words": len(tokens),
        "n_sentences": sum(1 for _ in doc.sents),
        "n_negations": sum(is_negation(t) for t in doc),
        "n_exclamations": sum(t.text == "!" for t in doc),
        **counts,
    }


def process(reviews, cache=CACHE, batch_size=64):
    """
    Parse every review with spaCy, or read the result from the cache.

    Parameters:
    reviews (DataFrame): Output of data.load_reviews.
    cache (str or Path, optional): Pickle file for the result; None disables caching.
    batch_size (int): Reviews per spaCy batch.

    Returns:
    DataFrame: One row per review, same index as `reviews`, with the columns of analyse().
    """
    if cache is not None and Path(cache).exists():
        processed = pd.read_pickle(cache)
        if processed.index.equals(reviews.index) and "tokens_negation" in processed.columns:
            return processed
    nlp = load_nlp()
    rows = [analyse(doc) for doc in nlp.pipe(full_text(reviews), batch_size=batch_size)]
    processed = pd.DataFrame(rows, index=reviews.index)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        processed.to_pickle(cache)
    return processed
