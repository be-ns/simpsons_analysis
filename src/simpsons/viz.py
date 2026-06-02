"""Figure generation for the analysis. Every figure is built from honest,
leak-free outputs produced in :mod:`simpsons.modeling`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D
from . import modeling as M

# A restrained palette nodding to the show without being garish.
INK = "#2b2b2b"
GOLD = "#FED41D"
BLUE = "#3267a8"
RED = "#c1492f"
GREY = "#9aa0a6"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#ececec",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })


def _rolling(y: np.ndarray, window: int = 15) -> np.ndarray:
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()


def rating_over_time(df: pd.DataFrame, path: Path) -> None:
    """The headline figure: 27 years of decline, with the golden era marked."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = df["number_in_series"].to_numpy()
    y = df["imdb_rating"].to_numpy()
    ax.scatter(x, y, s=14, c=GREY, alpha=0.55, label="Episode rating", linewidths=0)
    ax.plot(x, _rolling(y), color=RED, lw=2.6, label="15-episode rolling mean")

    cutoff = df.loc[df["season"] <= 10, "number_in_series"].max()
    ax.axvspan(x.min(), cutoff, color=GOLD, alpha=0.18)
    ax.text(cutoff / 2, 4.8, "Golden era (S1–10)", ha="center", color="#8a6d00",
            fontsize=11, fontweight="bold")
    ax.text((cutoff + x.max()) / 2, 4.8, "Later seasons (S11+)", ha="center",
            color=INK, fontsize=11, fontweight="bold")

    early, late = df[df.season <= 10].imdb_rating.mean(), df[df.season > 10].imdb_rating.mean()
    ax.set_title(f"The Simpsons' IMDb decline  ·  S1–10 avg {early:.1f}  →  S11+ avg {late:.1f}")
    ax.set_xlabel("Episode number in series")
    ax.set_ylabel("IMDb rating (1–10)")
    ax.set_xlim(x.min(), x.max())
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def model_comparison(results: list[M.CVResult], path: Path) -> None:
    """Bar chart of honest 5-fold CV RMSE across feature sets."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r.name for r in results]
    means = [r.rmse_mean for r in results]
    stds = [r.rmse_std for r in results]
    colors = [GREY, RED, BLUE, GOLD]
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors,
                  edgecolor=INK, linewidth=0.8)
    baseline = results[0].rmse_mean
    ax.axhline(baseline, ls="--", color=INK, lw=1, alpha=0.6)
    ax.text(len(names) - 0.5, baseline + 0.006, "naive baseline", ha="right",
            fontsize=9, color=INK)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.012, f"{m:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cross-validated RMSE  (lower is better)")
    ax.set_title("Where the signal lives: scripts add nothing beyond air date")
    ax.set_ylim(0, max(means) * 1.18)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def permutation_importance_fig(imp: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    imp = imp.iloc[::-1]
    colors = [GOLD if f in ("number_in_series", "season") else BLUE for f in imp["feature"]]
    ax.barh(imp["feature"], imp["rmse_increase"], xerr=imp["std"], capsize=3,
            color=colors, edgecolor=INK, linewidth=0.7)
    ax.set_xlabel("Increase in RMSE when feature is shuffled")
    ax.set_title("Permutation importance: chronology dominates, scripts are noise")
    ax.axvline(0, color=INK, lw=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def predicted_vs_actual(df: pd.DataFrame, oof: np.ndarray, rmse: float, path: Path) -> None:
    """Out-of-fold predictions overlaid on the true rating timeline."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = df["number_in_series"].to_numpy()
    ax.plot(x, df["imdb_rating"], color=GREY, lw=1.2, alpha=0.7, label="Actual rating")
    ax.plot(x, _rolling(oof, 9), color=BLUE, lw=2.4, label="Predicted (out-of-fold)")
    ax.set_title(f"Out-of-fold predictions track the trend, not the episode  ·  RMSE {rmse:.3f}")
    ax.set_xlabel("Episode number in series")
    ax.set_ylabel("IMDb rating (1–10)")
    ax.set_xlim(x.min(), x.max())
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def autoresearch_leaderboard(board: pd.DataFrame, path: Path) -> None:
    """Heatmap of CV RMSE across every feature set x model from autoresearch."""
    _style()
    order_fs = ["chrono_only", "engineered", "engineered+chrono", "linguistic+chrono",
                "lsa_only", "lsa+chrono", "all_text+chrono"]
    order_m = ["ridge", "elasticnet", "knn", "random_forest", "hist_gbm", "mlp"]
    piv = board.pivot(index="feature_set", columns="model", values="cv_rmse")
    piv = piv.reindex([f for f in order_fs if f in piv.index])[[m for m in order_m if m in piv.columns]]
    data = piv.values

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0.43, vmax=0.75)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white" if (v > 0.62 or v < 0.445) else "black")
    ax.set_title("Autoresearch leaderboard — 5-fold CV RMSE (lower = greener)")
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("CV RMSE")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def script_features_vs_rating(df: pd.DataFrame, path: Path) -> None:
    """Four script features vs. rating — visibly flat clouds."""
    _style()
    feats = ["family_line_ratio", "avg_line_len", "n_locations", "max_line_len"]
    labels = ["Family screen-time ratio", "Avg words per line",
              "Distinct locations", "Longest line (words)"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, f, lab in zip(axes.ravel(), feats, labels):
        ax.scatter(df[f], df["imdb_rating"], s=12, c=BLUE, alpha=0.45, linewidths=0)
        r = np.corrcoef(df[f], df["imdb_rating"])[0, 1]
        ax.set_title(f"{lab}   (r = {r:+.2f})", fontsize=11)
        ax.set_xlabel(lab)
        ax.set_ylabel("IMDb rating")
    fig.suptitle("Script features carry almost no signal about quality",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
