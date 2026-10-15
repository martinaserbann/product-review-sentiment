import unittest

from review_sentiment.preprocess import analyse, load_nlp


class TestAnalyse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = load_nlp()

    def test_negation_scope_ends_at_punctuation(self):
        result = analyse(self.nlp("Not good at all, but cheap."))
        self.assertEqual(result["lemmas_negation"], "not NOT_good NOT_at NOT_all but cheap")
        self.assertEqual(result["n_negations"], 1)

    def test_words_keep_their_tense(self):
        result = analyse(self.nlp("It worked, but it doesn't work now."))
        self.assertEqual(result["tokens_negation"], "it worked but it does n't NOT_work NOT_now")
        self.assertEqual(result["lemmas"], "it work but it do not work now")

    def test_contracted_negation_is_found_by_the_parser(self):
        result = analyse(self.nlp("I wouldn't buy it again."))
        self.assertIn("NOT_buy", result["lemmas_negation"].split())

    def test_lemmas_and_counts(self):
        result = analyse(self.nlp("The pans were great! They heat evenly."))
        self.assertEqual(result["lemmas"], "the pan be great they heat evenly")
        self.assertEqual(result["tokens"], "the pans were great they heat evenly")
        self.assertEqual((result["n_words"], result["n_sentences"], result["n_exclamations"]), (7, 2, 1))
        self.assertEqual(result["n_adj"], 1)


if __name__ == "__main__":
    unittest.main()
