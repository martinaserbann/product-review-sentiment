"""
Download, parse and split the Multi-Domain Sentiment Dataset: Amazon product reviews in four
categories, each review labelled positive (4 or 5 stars) or negative (1 or 2 stars).

Blitzer, Dredze and Pereira (2007). Biographies, Bollywood, Boom-boxes and Blenders: Domain
Adaptation for Sentiment Classification. ACL 2007.
"""

import hashlib
import re
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DOMAINS = ["books", "dvd", "electronics", "kitchen"]
DOMAIN_NAMES = {"books": "Books", "dvd": "DVDs", "electronics": "Electronics", "kitchen": "Kitchen"}
LABELS = ["negative", "positive"]  # label 0 and label 1
RANDOM_STATE = 0
TEST_SIZE = 0.2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR.parent / "results"  # predictions saved by the transformer runs

# The labelled reviews (title and text) as published with Ziser and Reichart (2018), pinned to
# one commit so that the files never change. The original archive is linked in the README.
SOURCE = ("https://raw.githubusercontent.com/yftah89/PBLM-Domain-Adaptation/"
          "2eef6454720535ceac864d1c07852f61d61e963c/data/{domain}/{label}.parsed")
SHA256 = {
    ("books", "negative"): "c212e6f8601db51807baefc7db44e0c96c900fb53da16e57beaa7b96074a9c58",
    ("books", "positive"): "b740263ec09f4e8b070ba82dd160306d0bd870eb394bc8d6e191d76986c30b98",
    ("dvd", "negative"): "58c9e62eb5e97de4dbc7817d4549c6cd23fb947d6403eaddae1cebc7a8459b17",
    ("dvd", "positive"): "3258e0b744068d5cf493b585edba40bd9f29ff457f7ff01be8b94c3af792047c",
    ("electronics", "negative"): "f09930ad7cbf8e6e3be6add636c932c197ec7d55f351907d6c3cadc22bf4eeb0",
    ("electronics", "positive"): "f4eee889891058bee5c0773f7c6862f8dc9ea59ef182d170ee42de2ff2c4844e",
    ("kitchen", "negative"): "a73bb3dcbe165d18983aa37fa8afc1b745a643582b422b78d1d95ee4e0576318",
    ("kitchen", "positive"): "bf12d5495293d490c61665111f55e33243cfa7281b96e62e467bc84d3b7e7a58",
}


def raw_path(domain, label, raw_dir=RAW_DIR):
    """Local path of one raw file."""
    return Path(raw_dir) / f"{domain}_{label}.parsed"


def download(raw_dir=RAW_DIR):
    """
    Download the eight raw files (four categories, positive and negative) if they are missing,
    and check that each one is exactly the expected file.

    Parameters:
    raw_dir (str or Path): Folder for the raw files.
    """
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    for (domain, label), expected in SHA256.items():
        path = raw_path(domain, label, raw_dir)
        if not path.exists():
            urllib.request.urlretrieve(SOURCE.format(domain=domain, label=label), path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"{path.name} does not match the published file (sha256 {digest})")


def parse(text):
    """
    Parse one raw file. Each review sits between <review> tags: the first line is the title,
    the rest is the review itself.

    Parameters:
    text (str): Content of a raw file.

    Returns:
    list: (title, text) pairs.
    """
    reviews = []
    for chunk in re.findall(r"<review>(.*?)</review>", text, flags=re.S):
        lines = chunk.strip("\n").split("\n")
        reviews.append((lines[0].strip(), "\n".join(lines[1:]).strip()))
    return reviews


def load_reviews(raw_dir=RAW_DIR):
    """
    Load every review, downloading the data the first time.

    Returns:
    DataFrame: One row per unique review, with columns domain, label (0 negative, 1 positive),
    title and text.
    """
    download(raw_dir)
    return read_reviews(raw_dir)


def read_reviews(raw_dir=RAW_DIR):
    """
    Read the raw files of every category and sentiment into one table.

    Some reviews appear twice in the same category with the same text. Duplicates are removed
    before splitting, otherwise a copy of a test review could sit in the training data.

    Returns:
    DataFrame: One row per unique review, with columns domain, label, title and text.
    """
    rows = []
    for domain in DOMAINS:
        for label_id, label in enumerate(LABELS):
            content = raw_path(domain, label, raw_dir).read_text(encoding="utf-8")
            rows += [{"domain": domain, "label": label_id, "title": title, "text": text}
                     for title, text in parse(content)]
    reviews = pd.DataFrame(rows)
    return reviews.drop_duplicates(subset=["domain", "text"]).reset_index(drop=True)


def add_split(reviews, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Hold out 20% of each category as its test set, keeping the share of positive reviews.

    Every model is trained on the training part of one category and evaluated on the test part
    of all four, so the same test reviews are used for in-category and cross-category scores.

    Returns:
    DataFrame: A copy of the reviews with a "split" column, "train" or "test".
    """
    reviews = reviews.copy()
    reviews["split"] = "train"
    for domain in DOMAINS:
        rows = reviews.index[reviews["domain"] == domain]
        _, test_rows = train_test_split(rows, test_size=test_size, stratify=reviews.loc[rows, "label"],
                                        random_state=random_state)
        reviews.loc[test_rows, "split"] = "test"
    return reviews


def full_text(reviews):
    """
    The text given to the models: the title, as a sentence of its own, then the review.

    Returns:
    list: One string per review.
    """
    titles = reviews["title"].where(reviews["title"].str.contains(r"[.!?]$"), reviews["title"] + ".")
    return (titles + "\n" + reviews["text"]).tolist()
