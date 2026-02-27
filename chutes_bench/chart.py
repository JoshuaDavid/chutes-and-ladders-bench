"""Generate a leaderboard bar chart from Elo ratings."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt


def make_elo_chart(
    ratings: dict[str, float],
    output_path: str = "elo_leaderboard.png",
    title: str = "Chutes & Ladders Elo Leaderboard",
) -> str:
    """Create a horizontal bar chart of Elo ratings, sorted descending.

    Returns the path to the saved PNG.
    """
    sorted_items = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    names = [name for name, _ in sorted_items]
    scores = [score for _, score in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.7)))
    bars = ax.barh(names, scores, color="#4A90D9", edgecolor="white")

    # Annotate bars with rating values
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            f"{score:.0f}",
            va="center", fontsize=11, fontweight="bold",
        )

    ax.set_xlabel("Elo Rating")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # highest on top
    ax.set_xlim(left=min(scores) - 50, right=max(scores) + 60)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
