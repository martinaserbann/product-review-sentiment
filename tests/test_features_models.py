import unittest

import numpy as np

from review_sentiment.features import average_embeddings, tfidf, vader_compound
from review_sentiment.models import ThresholdClassifier, tuned_logistic_regression


class FakeVectors:
    """The two attributes of gensim's KeyedVectors that average_embeddings uses."""

    def __init__(self, table):
        self.key_to_index = {word: i for i, word in enumerate(table)}
        self.matrix = np.array(list(table.values()), dtype=np.float32)
        self.vector_size = self.matrix.shape[1]

    def __getitem__(self, words):
        return self.matrix[[self.key_to_index[w] for w in words]]


class TestFeatures(unittest.TestCase):
    def test_average_skips_unknown_words_and_handles_empty_texts(self):
        vectors = FakeVectors({"good": [1.0, 0.0], "bad": [0.0, 1.0]})
        matrix = average_embeddings(["good bad unknown", "unknown", "good good"], vectors)
        np.testing.assert_allclose(matrix, [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]])

    def test_tfidf_keeps_negation_marks_and_bigrams(self):
        vectorizer = tfidf(min_df=1).fit(["not NOT_good", "very good"])
        vocabulary = set(vectorizer.get_feature_names_out())
        self.assertIn("NOT_good", vocabulary)
        self.assertIn("not NOT_good", vocabulary)

    def test_vader_reads_negation(self):
        scores = vader_compound(["This is great!", "This is not great at all."])
        self.assertGreater(scores[0], 0.5)
        self.assertLess(scores[1], 0)


class TestModels(unittest.TestCase):
    def test_threshold_is_the_most_accurate_cut(self):
        model = ThresholdClassifier().fit([-0.8, -0.2, 0.1, 0.3, 0.9], [0, 0, 0, 1, 1])
        self.assertEqual(model.threshold_, 0.1)
        np.testing.assert_array_equal(model.predict([0.0, 0.2]), [0, 1])

    def test_tuned_logistic_regression_learns_words(self):
        texts = ["great product", "love it great", "awful product", "broke awful"] * 5
        labels = [1, 1, 0, 0] * 5
        model = tuned_logistic_regression(tfidf(min_df=1)).fit(texts, labels)
        np.testing.assert_array_equal(model.predict(["great", "awful"]), [1, 0])


if __name__ == "__main__":
    unittest.main()
