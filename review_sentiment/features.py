"""
Text representations: sparse TF-IDF vectors, averaged GloVe word embeddings and the VADER
sentiment lexicon. Contextual RoBERTa embeddings live in transformer.py.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

GLOVE = "glove-wiki-gigaword-300"  # Pennington et al. (2014): 400,000 words, 300 dimensions


def tfidf(ngram_range=(1, 2), min_df=2):
    """
    TF-IDF over already tokenized text (tokens joined by spaces). Term counts are
    log-scaled (sublinear tf), so a word repeated ten times does not weigh ten times more.

    Parameters:
    ngram_range (tuple): Smallest and largest n-gram.
    min_df (int): Ignore terms found in fewer training reviews than this.

    Returns:
    TfidfVectorizer: The unfitted vectorizer.
    """
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=ngram_range, min_df=min_df,
                           sublinear_tf=True)


def load_glove(name=GLOVE):
    """Download (the first time, about 380 MB) and load pretrained GloVe vectors with gensim."""
    import gensim.downloader

    return gensim.downloader.load(name)


def average_embeddings(texts, vectors):
    """
    Represent each text by the mean of its words' embeddings. Words missing from the
    embedding vocabulary are skipped; a text with no known word gets a vector of zeros.

    Parameters:
    texts (list): Tokenized texts, tokens joined by spaces.
    vectors: Word vectors with gensim's KeyedVectors interface.

    Returns:
    ndarray: One row per text.
    """
    matrix = np.zeros((len(texts), vectors.vector_size), dtype=np.float32)
    for row, text in enumerate(texts):
        known = [word for word in text.split() if word in vectors.key_to_index]
        if known:
            matrix[row] = vectors[known].mean(axis=0)
    return matrix


def vader_compound(texts):
    """
    VADER's compound score for each text, from -1 (most negative) to +1 (most positive).
    VADER (Hutto and Gilbert, 2014) is a hand-built sentiment lexicon with rules for
    negation, intensifiers ("very"), capitals and exclamation marks; nothing is trained.

    Returns:
    ndarray: One score per text.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return np.array([analyzer.polarity_scores(text)["compound"] for text in texts])
