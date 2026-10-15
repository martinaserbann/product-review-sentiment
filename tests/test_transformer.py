import unittest

import torch

from review_sentiment.transformer import BOS, EOS, PAD, encode, mean_pool, pad, training_batches


class FakeTokenizer:
    """Turns a text of n words into the ids 10, 11, ..., 10 + n - 1."""

    def encode_as_ids(self, text):
        return [10 + i for i in range(len(text.split()))]


class TestEncode(unittest.TestCase):
    def test_short_text_is_wrapped_in_start_and_end_tokens(self):
        self.assertEqual(encode(["a b c"], FakeTokenizer(), max_pieces=8), [[BOS, 10, 11, 12, EOS]])

    def test_long_text_keeps_its_beginning_and_end(self):
        ids = encode([" ".join("w" * 20)], FakeTokenizer(), max_pieces=8)[0]
        self.assertEqual(len(ids), 8)
        self.assertEqual(ids, [BOS, 10, 11, 12, 27, 28, 29, EOS])


class TestBatches(unittest.TestCase):
    def test_padding_and_mask(self):
        ids, mask = pad([[0, 5, 2], [0, 2]])
        self.assertEqual(ids.tolist(), [[0, 5, 2], [0, 2, PAD]])
        self.assertEqual(mask.tolist(), [[True, True, True], [True, True, False]])

    def test_mean_pool_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])
        self.assertEqual(mean_pool(hidden, mask).tolist(), [[2.0, 2.0]])

    def test_training_batches_cover_every_review_once(self):
        import random

        encoded = [[0] * n for n in range(1, 101)]
        batches = training_batches(encoded, 16, random.Random(0))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(100)))
        self.assertTrue(all(len(batch) <= 16 for batch in batches))


if __name__ == "__main__":
    unittest.main()
