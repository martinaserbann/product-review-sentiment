"""
Linguistic and error analysis: how reviews differ between categories and sentiments, which
words carry sentiment in each category, and which reviews the models get wrong.
"""

from collections import Counter

import numpy as np
import pandas as pd

from .data import DOMAINS, LABELS


def text_statistics(reviews, processed):
    """
    Average length and linguistic features per category and sentiment.

    Returns:
    DataFrame: One row per (category, sentiment): reviews, words, sentences, and negations,
    adjectives and exclamation marks per 100 words.
    """
    data = processed.assign(domain=reviews["domain"], sentiment=reviews["label"].map(dict(enumerate(LABELS))))
    grouped = data.groupby(["domain", "sentiment"])
    per_100 = lambda column: 100 * grouped[column].sum() / grouped["n_words"].sum()  # noqa: E731
    return pd.DataFrame({
        "reviews": grouped.size(),
        "words": grouped["n_words"].mean(),
        "sentences": grouped["n_sentences"].mean(),
        "negations per 100 words": per_100("n_negations"),
        "adjectives per 100 words": per_100("n_adj"),
        "exclamation marks per 100 words": per_100("n_exclamations"),
    })


def log_odds_z(counts_a, counts_b, prior):
    """
    Which words are typical of one group of texts rather than another: the log-odds ratio with
    an informative Dirichlet prior (Monroe, Colaresi and Quinn, 2008), divided by its standard
    deviation. Positive z-scores lean to group a, negative ones to group b; frequent words need
    a consistent difference to score high, and rare words cannot score high by chance.

    Parameters:
    counts_a, counts_b (Counter): Word counts of the two groups.
    prior (Counter): Background word counts, here from all the reviews.

    Returns:
    Series: z-score per word of the background vocabulary.
    """
    n_a, n_b, a0 = sum(counts_a.values()), sum(counts_b.values()), sum(prior.values())
    words = list(prior)
    y_a = np.array([counts_a[w] for w in words], dtype=float)
    y_b = np.array([counts_b[w] for w in words], dtype=float)
    alpha = np.array([prior[w] for w in words], dtype=float)
    delta = (np.log((y_a + alpha) / (n_a + a0 - y_a - alpha))
             - np.log((y_b + alpha) / (n_b + a0 - y_b - alpha)))
    return pd.Series(delta / np.sqrt(1 / (y_a + alpha) + 1 / (y_b + alpha)), index=words)


def document_counts(texts):
    """In how many texts each word appears (a word repeated in one review counts once)."""
    counts = Counter()
    for text in texts:
        counts.update(set(text.split()))
    return counts


def is_content_word(word):
    """Alphabetic, at least three letters and not a stop word such as "the" or "and"."""
    from spacy.lang.en.stop_words import STOP_WORDS

    return word.isalpha() and len(word) > 2 and word not in STOP_WORDS


def sentiment_z_scores(reviews, processed, column="lemmas"):
    """
    For each category, how strongly each word leans positive (z > 0) or negative (z < 0),
    computed on the training reviews only. Words are counted once per review, so a name
    repeated throughout one long review does not look like a sentiment word.

    Returns:
    DataFrame: One row per content word, one z-score column per category, and the number of
    training reviews of each category that contain the word.
    """
    train = reviews["split"] == "train"
    prior = document_counts(processed.loc[train, column])
    z, counts = {}, {}
    for domain in DOMAINS:
        in_domain = train & (reviews["domain"] == domain)
        positive = document_counts(processed.loc[in_domain & (reviews["label"] == 1), column])
        negative = document_counts(processed.loc[in_domain & (reviews["label"] == 0), column])
        z[domain] = log_odds_z(positive, negative, prior)
        counts[f"count {domain}"] = pd.Series({w: positive[w] + negative[w] for w in prior})
    table = pd.DataFrame(z).join(pd.DataFrame(counts))
    return table.loc[[w for w in table.index if is_content_word(w)]]


def top_words(z_scores, domain, n=10, min_count=10):
    """
    The n most positive and the n most negative words of one category, among words found in at
    least `min_count` of its training reviews.

    Returns:
    dict: Lists of words under "positive" and "negative".
    """
    frequent = z_scores[z_scores[f"count {domain}"] >= min_count][domain].sort_values()
    return {"positive": list(frequent.index[::-1][:n]), "negative": list(frequent.index[:n])}


def sentiment_vocabulary(z_scores, domain, n=100, min_count=10):
    """The n words of a category with the strongest sentiment, positive or negative."""
    frequent = z_scores[z_scores[f"count {domain}"] >= min_count][domain]
    return set(frequent.abs().sort_values(ascending=False).index[:n])


def vocabulary_overlap(z_scores, n=100, min_count=10):
    """
    How much two categories share their sentiment vocabulary: the share of one category's
    n strongest sentiment words that are also among the n strongest of the other.

    Returns:
    DataFrame: 4 x 4 shares, symmetric, with 1 on the diagonal.
    """
    vocabularies = {domain: sentiment_vocabulary(z_scores, domain, n, min_count) for domain in DOMAINS}
    return pd.DataFrame({a: {b: len(vocabularies[a] & vocabularies[b]) / n for b in DOMAINS} for a in DOMAINS})


def confident_errors(preds, reviews, model, n=8):
    """
    The test reviews a model gets wrong with the most confidence, in the category it was
    trained on.

    Returns:
    DataFrame: Category, true sentiment, predicted probability of positive, title and text.
    """
    rows = preds[(preds["model"] == model) & (preds["train_domain"] == preds["domain"])]
    wrong = rows[rows["prediction"] != rows["label"]].copy()
    wrong["confidence"] = (wrong["score"] - 0.5).abs()
    wrong = wrong.sort_values("confidence", ascending=False).head(n).set_index("review")
    return reviews.loc[wrong.index, ["domain", "title", "text"]].assign(
        sentiment=wrong["label"].map(dict(enumerate(LABELS))), score=wrong["score"])


def disagreements(preds, reviews, right_model, wrong_model):
    """
    Test reviews (in-category) that one model classifies correctly and another does not.

    Returns:
    DataFrame: The reviews, with both models' scores.
    """
    def in_domain(model):
        rows = preds[(preds["model"] == model) & (preds["train_domain"] == preds["domain"])]
        return rows.set_index("review")

    right, wrong = in_domain(right_model), in_domain(wrong_model)
    chosen = right.index[(right["prediction"] == right["label"]) & (wrong.loc[right.index, "prediction"] != right["label"])]
    return reviews.loc[chosen, ["domain", "title", "text"]].assign(
        sentiment=right.loc[chosen, "label"].map(dict(enumerate(LABELS))),
        **{right_model: right.loc[chosen, "score"], wrong_model: wrong.loc[chosen, "score"]})
