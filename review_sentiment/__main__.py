"""
Run the analysis and the classic models, add the saved transformer predictions, and write every
table and figure: python -m review_sentiment

The transformer predictions come from `python -m review_sentiment.transformer` (slow on a CPU).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import analysis, plots, published  # noqa: E402
from .data import DOMAIN_NAMES, DOMAINS, RESULTS_DIR, add_split, load_reviews  # noqa: E402
from .evaluate import (DESCRIPTIONS, FINE_TUNED, MODELS, TFIDF, accuracy_matrix, bootstrap_ci,  # noqa: E402
                       load_transformer_predictions, run_classic_models, summary)
from .experiments import best_variant, preprocessing_experiments  # noqa: E402
from .features import load_glove  # noqa: E402
from .preprocess import process  # noqa: E402


def markdown_table(frame, percent=(), decimals=3):
    """
    Format a DataFrame as a Markdown table, with category names written out. Columns listed in
    `percent` are shown as percentages.
    """
    frame = frame.rename(columns=DOMAIN_NAMES, index=DOMAIN_NAMES)
    percent = [DOMAIN_NAMES.get(column, column) for column in percent]

    def cell(column, value):
        if isinstance(value, float):
            return f"{value:.1%}" if column in percent else f"{value:.{decimals}f}"
        return str(value)

    header = "| " + " | ".join([frame.index.name or ""] + [str(c) for c in frame.columns]) + " |"
    lines = [header, "|---" * (len(frame.columns) + 1) + "|"]
    for name, row in frame.iterrows():
        lines.append("| " + " | ".join([str(name)] + [cell(c, row[c]) for c in frame.columns]) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m review_sentiment", description=__doc__.split("\n")[1])
    parser.add_argument("--out", default="docs", help="folder for tables and figures")
    parser.add_argument("--results", default=str(RESULTS_DIR), help="folder with the transformer predictions")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reviews = add_split(load_reviews())
    processed = process(reviews)

    stats = analysis.text_statistics(reviews, processed)
    z_scores = analysis.sentiment_z_scores(reviews, processed)
    overlap = analysis.vocabulary_overlap(z_scores)
    experiments = preprocessing_experiments(reviews, processed)
    variant, column, ngram_range = best_variant(experiments)
    descriptions = {**DESCRIPTIONS, TFIDF: f"{variant}, logistic regression"}

    preds = pd.concat([run_classic_models(reviews, processed, load_glove(), column, ngram_range),
                       load_transformer_predictions(Path(args.results))], ignore_index=True)
    models = [m for m in MODELS if m in set(preds["model"])]
    table = summary(preds, models)
    table["95% CI"] = [" to ".join(f"{v:.1%}" for v in bootstrap_ci(preds, m)) for m in table.index]
    best = table["in-category"].idxmax()
    matrices = {m: accuracy_matrix(preds, m) for m in (TFIDF, FINE_TUNED) if m in models}
    pairs = [(a, b) for a in DOMAINS for b in DOMAINS if a != b]
    correlations = {m: np.corrcoef([overlap.loc[a, b] for a, b in pairs], [matrix.loc[a, b] for a, b in pairs])[0, 1]
                    for m, matrix in matrices.items()}

    accuracy = table[DOMAINS + ["in-category", "95% CI", "cross-category"]].rename(
        columns={"in-category": "Same category", "cross-category": "Other categories"})
    # the drop between the two rounded percentages, so that the table adds up
    accuracy["Drop, points"] = (100 * table["in-category"]).round(1) - (100 * table["cross-category"]).round(1)
    accuracy.index.name = "Model"
    accuracy.insert(0, "Representation", [descriptions[model] for model in accuracy.index])
    words = pd.DataFrame({d: {s.capitalize(): ", ".join(w) for s, w in analysis.top_words(z_scores, d).items()}
                          for d in DOMAINS}).T
    words.index.name = "Category"
    errors = analysis.confident_errors(preds, reviews, best)
    report = [
        "## Test accuracy",
        "",
        "Each model is trained on one category and tested on the held-out reviews of all four. "
        "\"Same category\" averages the four in-category accuracies, with a 95% bootstrap interval; "
        "\"other categories\" averages the 12 cross-category ones.",
        "",
        markdown_table(accuracy, percent=DOMAINS + ["Same category", "Other categories"], decimals=1),
        "",
        "## Compared with published results",
        "",
        markdown_table(published.in_category_table(table, models).rename_axis("Same category"),
                       percent=DOMAINS + ["Average"]),
        "",
        markdown_table(published.cross_category_table(table, models).to_frame().rename_axis("Other categories"),
                       percent=["Average accuracy, 12 category pairs"]),
        "",
        "## Preprocessing experiments (TF-IDF, cross-validation on the training reviews)",
        "",
        markdown_table(experiments.rename(columns={"average": "Average"}).rename_axis("Variant"),
                       percent=DOMAINS + ["Average"]),
        "",
        f"The TF-IDF model uses the best variant: {variant}.",
        "",
        "## Words with the strongest sentiment, per category (training reviews)",
        "",
        markdown_table(words),
        "",
        "## Shared sentiment vocabulary between categories (top 100 words)",
        "",
        markdown_table(overlap.rename_axis("Category"), percent=DOMAINS),
        "",
        "Correlation between shared vocabulary and cross-category accuracy over the 12 pairs: "
        + ", ".join(f"{m} r = {r:.2f}" for m, r in correlations.items()),
        "",
        "## Text statistics",
        "",
        markdown_table(stats.reset_index().set_index("domain").rename_axis("Category").rename(
            columns=lambda c: c[0].upper() + c[1:]), decimals=1),
        "",
        f"## Most confident errors of the best model ({best})",
        "",
    ]
    for review, row in errors.iterrows():
        excerpt = " ".join(row["text"].split())[:280]
        report.append(f"- **{row['domain']}, {row['sentiment']}, P(positive) = {row['score']:.2f}**: "
                      f"*{row['title']}*. {excerpt}...")
    (out / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    accuracy.to_csv(out / "accuracy.csv")
    experiments.to_csv(out / "preprocessing.csv")
    overlap.to_csv(out / "vocabulary_overlap.csv")
    stats.to_csv(out / "text_statistics.csv")

    plots.plot_accuracy(table, out / "accuracy.png")
    plots.plot_accuracy_matrices(matrices, out / "accuracy_matrices.png")
    plots.plot_sentiment_words(z_scores, path=out / "sentiment_words.png")
    plots.plot_overlap_vs_transfer(overlap, matrices[TFIDF], TFIDF, out / "vocabulary_transfer.png")
    plots.plot_text_statistics(stats, "negations per 100 words", out / "negations.png")
    plt.close("all")
    print((out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
