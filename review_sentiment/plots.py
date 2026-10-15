"""Figures for the README and the notebook, in one consistent style."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

from .data import DOMAIN_NAMES, DOMAINS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # first two series colours, safe for colour-blind readers
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"  # the two poles of a diverging scale
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SHORT = {"books": "B", "dvd": "D", "electronics": "E", "kitchen": "K"}


def _style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _finish(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_accuracy(summary, path=None):
    """
    Accuracy of every model, on the category it was trained on and on the other three.

    Parameters:
    summary (DataFrame): Output of evaluate.summary.
    path (str or Path, optional): Where to save the PNG.

    Returns:
    Figure: The matplotlib figure.
    """
    ordered = summary.sort_values("in-category")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(ordered) + 1.3), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.hlines(y, ordered["cross-category"], ordered["in-category"], color=GRID, linewidth=2, zorder=1)
    for column, label, color, align, offset in (("cross-category", "Other categories", ORANGE, "right", -0.006),
                                                ("in-category", "Same category", BLUE, "left", 0.006)):
        ax.scatter(ordered[column], y, s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=2, label=label)
        for value, row in zip(ordered[column], y):
            ax.text(value + offset, row, f"{value:.1%}", va="center", ha=align, fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(y, ordered.index, color=INK, fontsize=9)
    ax.set_xlim(0.6, 1.0)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("Test accuracy, average over the four categories", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Accuracy by text representation", loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finish(fig, path)


def plot_accuracy_matrices(matrices, path=None, vmin=0.65, vmax=0.95):
    """
    Side-by-side 4 x 4 accuracy matrices: rows are the training category, columns the test
    category, so the diagonal is in-category accuracy.

    Parameters:
    matrices (dict): Model name mapped to the output of evaluate.accuracy_matrix.
    path (str or Path, optional): Where to save the PNG.
    vmin, vmax (float): Range of the shared colour scale.

    Returns:
    Figure: The matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.4 * len(matrices), 4.2), dpi=150, facecolor=SURFACE)
    for ax, (name, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        values = matrix.loc[DOMAINS, DOMAINS].to_numpy()
        ax.imshow(values, cmap=BLUES, vmin=vmin, vmax=vmax)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(DOMAINS)):
            for j in range(len(DOMAINS)):
                dark = (values[i, j] - vmin) / (vmax - vmin) > 0.55
                ax.text(j, i, f"{values[i, j]:.1%}", ha="center", va="center", fontsize=9,
                        color="white" if dark else INK, fontweight="bold" if i == j else "normal")
        names = [DOMAIN_NAMES[d] for d in DOMAINS]
        ax.set_xticks(range(len(DOMAINS)), names, color=INK_SECONDARY, fontsize=8)
        ax.set_yticks(range(len(DOMAINS)), names, color=INK_SECONDARY, fontsize=8)
        ax.tick_params(length=0)
        ax.set_xlabel("Tested on", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("Trained on", color=INK_SECONDARY, fontsize=9)
        ax.set_title(name, loc="left", color=INK, fontsize=10, fontweight="bold")
    return _finish(fig, path)


def plot_sentiment_words(z_scores, n=8, min_count=10, path=None):
    """
    The words that lean most positive and most negative in each category (log-odds z-scores on
    the training reviews), one panel per category.

    Parameters:
    z_scores (DataFrame): Output of analysis.sentiment_z_scores.
    n (int): Words per side.
    min_count (int): Minimum number of training reviews containing the word.
    path (str or Path, optional): Where to save the PNG.

    Returns:
    Figure: The matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(13, 0.32 * 2 * n + 1.4), dpi=150, facecolor=SURFACE)
    for ax, domain in zip(axes, DOMAINS):
        frequent = z_scores[z_scores[f"count {domain}"] >= min_count][domain].sort_values()
        chosen = list(frequent.index[:n]) + list(frequent.index[-n:])
        values = frequent[chosen]
        y = np.arange(len(chosen))
        _style(ax)
        ax.barh(y, values, height=0.7, color=[POSITIVE if v > 0 else NEGATIVE for v in values])
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_yticks(y, chosen, color=INK, fontsize=8)
        ax.set_title(DOMAIN_NAMES[domain], loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.set_xlabel("z-score", color=INK_SECONDARY, fontsize=8)
    fig.suptitle("Words that signal sentiment in each category (blue positive, red negative)", x=0.01, ha="left",
                 color=INK, fontsize=11, fontweight="bold")
    return _finish(fig, path)


def plot_overlap_vs_transfer(overlap, matrix, model, path=None):
    """
    For each pair of categories, shared sentiment vocabulary against the accuracy of a model
    trained on one category and tested on the other.

    Parameters:
    overlap (DataFrame): Output of analysis.vocabulary_overlap.
    matrix (DataFrame): Output of evaluate.accuracy_matrix for one model.
    model (str): Model name, for the axis label.
    path (str or Path, optional): Where to save the PNG.

    Returns:
    Figure: The matplotlib figure.
    """
    pairs = [(a, b) for a in DOMAINS for b in DOMAINS if a != b]
    x = np.array([overlap.loc[a, b] for a, b in pairs])
    y = np.array([matrix.loc[a, b] for a, b in pairs])
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150, facecolor=SURFACE)
    _style(ax, grid_axis="both")
    ax.scatter(x, y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    for (a, b), xi, yi in zip(pairs, x, y):
        ax.annotate(f"{SHORT[a]}→{SHORT[b]}", (xi, yi), xytext=(5, 3), textcoords="offset points",
                    fontsize=8, color=INK_SECONDARY)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("Shared sentiment words (top 100 of each category)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(f"{model}: accuracy on the other category", color=INK_SECONDARY, fontsize=9)
    ax.set_title(f"Shared vocabulary predicts transfer (r = {np.corrcoef(x, y)[0, 1]:.2f})", loc="left", color=INK,
                 fontsize=11, fontweight="bold")
    return _finish(fig, path)


def plot_text_statistics(stats, column, path=None):
    """
    One linguistic statistic per category, for negative and positive reviews.

    Parameters:
    stats (DataFrame): Output of analysis.text_statistics.
    column (str): The statistic to plot.
    path (str or Path, optional): Where to save the PNG.

    Returns:
    Figure: The matplotlib figure.
    """
    table = stats[column].unstack("sentiment").loc[DOMAINS]
    x = np.arange(len(DOMAINS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150, facecolor=SURFACE)
    _style(ax, grid_axis="y")
    for offset, sentiment, color in ((-width / 2 - 0.01, "negative", NEGATIVE), (width / 2 + 0.01, "positive", POSITIVE)):
        bars = ax.bar(x + offset, table[sentiment], width=width, color=color, label=f"{sentiment.capitalize()} reviews")
        for bar, value in zip(bars, table[sentiment]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8,
                    color=INK_SECONDARY)
    ax.set_xticks(x, [DOMAIN_NAMES[d] for d in DOMAINS], color=INK, fontsize=9)
    ax.set_ylim(0, table.to_numpy().max() * 1.3)  # room for the legend
    ax.set_ylabel(column.capitalize(), color=INK_SECONDARY, fontsize=9)
    ax.set_title(column.capitalize(), loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finish(fig, path)
