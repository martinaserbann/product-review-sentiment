"""
Which preprocessing steps help the TF-IDF model? Each variant is scored by 5-fold
cross-validation on the training reviews of each category; the test reviews are not used.
"""

import pandas as pd

from .data import DOMAINS
from .features import tfidf
from .models import tuned_logistic_regression

# Variant name mapped to (text column produced by preprocess.process, n-gram range)
VARIANTS = {
    "Words": ("tokens", (1, 1)),
    "Lemmas": ("lemmas", (1, 1)),
    "Words and bigrams": ("tokens", (1, 2)),
    "Lemmas and bigrams": ("lemmas", (1, 2)),
    "Words and bigrams, negation marked": ("tokens_negation", (1, 2)),
    "Lemmas and bigrams, negation marked": ("lemmas_negation", (1, 2)),
}


def preprocessing_experiments(reviews, processed, variants=VARIANTS):
    """
    Cross-validated accuracy of the TF-IDF model for each preprocessing variant and category.
    Each score is the mean accuracy over the 5 folds at the best regularisation strength.

    Returns:
    DataFrame: One row per variant, one column per category, plus the average.
    """
    scores = {}
    for name, (column, ngram_range) in variants.items():
        for domain in DOMAINS:
            train = reviews[(reviews["domain"] == domain) & (reviews["split"] == "train")]
            search = tuned_logistic_regression(tfidf(ngram_range)).fit(processed.loc[train.index, column], train["label"])
            scores.setdefault(name, {})[domain] = search.best_score_
    table = pd.DataFrame(scores).T
    table["average"] = table[DOMAINS].mean(axis=1)
    return table


def best_variant(table, variants=VARIANTS):
    """
    The variant with the best average cross-validated accuracy.

    Returns:
    tuple: Its name, text column and n-gram range.
    """
    name = table["average"].idxmax()
    return (name, *variants[name])
