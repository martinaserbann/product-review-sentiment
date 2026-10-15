"""
Published accuracies on the same dataset, for comparison. The setups differ slightly from this
project (see the README), so the comparison is indicative.
"""

import pandas as pd

from .data import DOMAINS

# Blitzer, Dredze and Pereira (2007), ACL: linear classifier on unigrams and bigrams,
# 1600 training and 400 test reviews per category (in-domain "gold standard", Figure 1).
BLITZER_IN_CATEGORY = {"books": 0.804, "dvd": 0.824, "electronics": 0.844, "kitchen": 0.877}

# Average accuracy over the 12 (training category, test category) pairs.
CROSS_CATEGORY = {
    "Blitzer et al. (2007), linear model, no adaptation": 0.745,  # averaged from Figure 1
    "Blitzer et al. (2007), SCL-MI, adapted with unlabelled target reviews": 0.780,  # averaged from Figure 1
    "Karouzos et al. (2021), BERT-base fine-tuned on the source category": 0.8993,  # Table 1, "SO BERT"
    "Karouzos et al. (2021), UDALM, adapted with unlabelled target reviews": 0.9174,  # Table 1
}


def in_category_table(summary, models):
    """
    In-category accuracy of this project's models next to Blitzer et al. (2007).

    Returns:
    DataFrame: One row per model, one column per category, plus the average.
    """
    rows = {"Blitzer et al. (2007), linear model on unigrams and bigrams": BLITZER_IN_CATEGORY}
    rows.update({f"This project: {model}": summary.loc[model, DOMAINS].to_dict() for model in models})
    table = pd.DataFrame(rows).T[DOMAINS]
    table["Average"] = table.mean(axis=1)
    return table


def cross_category_table(summary, models):
    """
    Average cross-category accuracy of this project's models next to published results.

    Returns:
    Series: Accuracy per method.
    """
    ours = {f"This project: {model}, no adaptation": summary.loc[model, "cross-category"] for model in models}
    return pd.Series({**CROSS_CATEGORY, **ours}, name="Average accuracy, 12 category pairs")
