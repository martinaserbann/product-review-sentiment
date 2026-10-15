import tempfile
import unittest
from pathlib import Path

import pandas as pd

from review_sentiment.data import DOMAINS, LABELS, add_split, full_text, parse, read_reviews

RAW = "<reviews><review>\tGreat blender\n\tIt works.\nVery well.\n</review><review>\tMeh\n\tBroke after a week\n</review>"


class TestParse(unittest.TestCase):
    def test_title_then_text(self):
        self.assertEqual(parse(RAW), [("Great blender", "It works.\nVery well."), ("Meh", "Broke after a week")])


class TestReadReviews(unittest.TestCase):
    def test_duplicates_within_a_category_are_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            for domain in DOMAINS:
                for label in LABELS:
                    reviews = RAW if label == "positive" else RAW + RAW  # every negative review twice
                    Path(folder, f"{domain}_{label}.parsed").write_text(reviews, encoding="utf-8")
            reviews = read_reviews(folder)
        # 2 positive and 2 unique negative reviews per category; duplicates across labels are kept
        self.assertEqual(len(reviews), 4 * len(DOMAINS) - 2 * len(DOMAINS))
        self.assertEqual(sorted(reviews.columns), ["domain", "label", "text", "title"])


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.reviews = pd.DataFrame({"domain": [d for d in DOMAINS for _ in range(100)],
                                     "label": [i % 2 for i in range(100 * len(DOMAINS))],
                                     "title": "t", "text": "x"})

    def test_twenty_percent_of_each_category_is_held_out_with_both_labels(self):
        split = add_split(self.reviews)
        for domain in DOMAINS:
            test = split[(split["domain"] == domain) & (split["split"] == "test")]
            self.assertEqual(len(test), 20)
            self.assertEqual(test["label"].sum(), 10)

    def test_split_is_reproducible(self):
        self.assertTrue(add_split(self.reviews)["split"].equals(add_split(self.reviews)["split"]))


class TestFullText(unittest.TestCase):
    def test_title_becomes_a_sentence(self):
        reviews = pd.DataFrame({"title": ["Great blender", "Awful!"], "text": ["It works.", "It broke."]})
        self.assertEqual(full_text(reviews), ["Great blender.\nIt works.", "Awful!\nIt broke."])


if __name__ == "__main__":
    unittest.main()
