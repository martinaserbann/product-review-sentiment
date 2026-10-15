import unittest
from collections import Counter

import numpy as np
import pandas as pd

from review_sentiment.analysis import document_counts, log_odds_z, vocabulary_overlap
from review_sentiment.data import DOMAINS
from review_sentiment.evaluate import accuracy_matrix, bootstrap_ci, summary


def fake_predictions(model="M"):
    """Every model is right on its own category and on half of the reviews elsewhere."""
    rows = []
    for train_domain in DOMAINS:
        for domain in DOMAINS:
            for review in range(10):
                label = review % 2
                right = train_domain == domain or review < 5
                rows.append({"model": model, "train_domain": train_domain, "review": review, "domain": domain,
                             "label": label, "score": 0.5, "prediction": label if right else 1 - label})
    return pd.DataFrame(rows)


class TestEvaluate(unittest.TestCase):
    def test_accuracy_matrix_and_summary(self):
        preds = fake_predictions()
        matrix = accuracy_matrix(preds, "M")
        self.assertTrue((np.diag(matrix) == 1.0).all())
        self.assertEqual(matrix.loc["books", "kitchen"], 0.5)
        row = summary(preds, ["M"]).loc["M"]
        self.assertEqual((row["in-category"], row["cross-category"], row["drop"]), (1.0, 0.5, 0.5))

    def test_bootstrap_interval_contains_the_accuracy(self):
        preds = fake_predictions()
        preds.loc[(preds["train_domain"] == preds["domain"]) & (preds["review"] < 3), "prediction"] ^= 1
        low, high = bootstrap_ci(preds, "M", n_resamples=300)
        self.assertLessEqual(low, 0.7)
        self.assertGreaterEqual(high, 0.7)


class TestAnalysis(unittest.TestCase):
    def test_words_count_once_per_document(self):
        self.assertEqual(document_counts(["good good bad", "good"]), Counter({"good": 2, "bad": 1}))

    def test_log_odds_sign_follows_the_group(self):
        a = Counter({"great": 30, "the": 50})
        b = Counter({"awful": 30, "the": 50})
        z = log_odds_z(a, b, a + b)
        self.assertGreater(z["great"], 2)
        self.assertLess(z["awful"], -2)
        self.assertAlmostEqual(z["the"], 0.0)

    def test_vocabulary_overlap(self):
        words = [f"w{i}" for i in range(4)]
        z = pd.DataFrame({"books": [3, -3, 1, 0], "dvd": [3, -3, 0, 1], "electronics": [0, 0, 3, -3],
                          "kitchen": [0, 1, 3, -3]}, index=words)
        for domain in DOMAINS:
            z[f"count {domain}"] = 20
        overlap = vocabulary_overlap(z, n=2)
        self.assertEqual(overlap.loc["books", "dvd"], 1.0)
        self.assertEqual(overlap.loc["books", "electronics"], 0.0)
        self.assertTrue((np.diag(overlap) == 1.0).all())


if __name__ == "__main__":
    unittest.main()
