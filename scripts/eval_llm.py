"""Evaluate the blind LLM rubric features against chronology — honestly.

Produces reports/llm_evaluation.json. Run after scoring episodes into
data/llm_features/ (see simpsons.llm_features / the subagent extraction).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simpsons import data as D  # noqa: E402
from simpsons import llm_features as L  # noqa: E402

CV = KFold(n_splits=5, shuffle=True, random_state=42)
MIN_WORDS = 100  # drop episodes whose source transcript is empty/near-empty


def _model():
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_depth=3, max_iter=400,
        l2_regularization=1.0, random_state=42,
    )


def _rmse(X, y):
    return float(np.sqrt(-cross_val_score(_model(), X, y, cv=CV,
                                          scoring="neg_mean_squared_error")).mean())


def main() -> None:
    sc = D.load_script_lines()
    words = sc[sc["is_speaking"]].groupby("episode_id")["spoken_words"].apply(
        lambda s: sum(len(str(x).split()) for x in s if isinstance(x, str)))

    ep = D.load_episodes().dropna(subset=["imdb_rating"]).set_index("id")
    df = L.load_llm_features().join(ep[["imdb_rating", "season", "number_in_series"]], how="inner")
    df["words"] = words.reindex(df.index).fillna(0)
    df = df[df["words"] >= MIN_WORDS]

    y = df["imdb_rating"].to_numpy()
    chrono = df[["number_in_series", "season"]].to_numpy()
    llm = df[L.FEATURE_COLUMNS].to_numpy()
    both = np.hstack([llm, chrono])

    trend = cross_val_predict(_model(), chrono, y, cv=CV)
    resid = y - trend
    craft = (df[["joke_quality", "story_coherence", "story_originality", "satire_sharpness",
                 "memorable_moment_strength", "emotional_resonance"]].mean(axis=1)
             - df[["guest_star_reliance", "mean_spiritedness"]].mean(axis=1))

    out = {
        "n_episodes": int(len(df)),
        "season_span": [int(df.season.min()), int(df.season.max())],
        "rmse": {
            "baseline_mean": round(float(df.imdb_rating.std()), 3),
            "chronology_only": round(_rmse(chrono, y), 3),
            "llm_only": round(_rmse(llm, y), 3),
            "llm_plus_chronology": round(_rmse(both, y), 3),
        },
        "residual_std": round(float(resid.std()), 3),
        "llm_residual_r2": round(float(cross_val_score(_model(), llm, resid, cv=CV, scoring="r2").mean()), 3),
        "corr_craft_rating": round(float(np.corrcoef(craft, y)[0, 1]), 3),
        "corr_craft_residual": round(float(np.corrcoef(craft, resid)[0, 1]), 3),
    }
    (ROOT / "reports" / "llm_evaluation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
