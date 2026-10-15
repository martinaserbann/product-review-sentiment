"""The classifiers: a tuned logistic regression for every representation, and a threshold on VADER."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  # inverse regularisation strength
RANDOM_STATE = 0


def cross_validator():
    """5-fold cross-validation that keeps the share of positive reviews in every fold."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def tuned_logistic_regression(vectorizer=None, scale=False):
    """
    Logistic regression whose regularisation strength C is chosen by 5-fold cross-validation on
    the training reviews only.

    Parameters:
    vectorizer (optional): Turns raw text into features, e.g. a TF-IDF vectorizer. It sits in
    the pipeline, so its vocabulary is learned from the training folds only.
    scale (bool): Standardise the features first, for dense embeddings.

    Returns:
    GridSearchCV: The unfitted model.
    """
    steps = ([vectorizer] if vectorizer is not None else []) + ([StandardScaler()] if scale else [])
    pipeline = make_pipeline(*steps, LogisticRegression(max_iter=5000))
    return GridSearchCV(pipeline, {"logisticregression__C": C_GRID}, scoring="accuracy", cv=cross_validator())


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    """
    Turns a single sentiment score (such as VADER's) into a positive or negative label, with
    the cut-off that is most accurate on the training reviews.
    """

    def fit(self, scores, y):
        scores, y = np.ravel(scores), np.asarray(y)
        candidates = np.unique(np.concatenate([scores, [-np.inf]]))
        accuracy = [((scores > cut) == y).mean() for cut in candidates]
        self.threshold_ = float(candidates[int(np.argmax(accuracy))])
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, scores):
        return (np.ravel(scores) > self.threshold_).astype(int)
