"""The decisive test: can any feature set explain rating *beyond* chronology?

Raw RMSE is the wrong yardstick — chronology dominates it, so a feature can be
genuinely informative and still barely move the total. The right question is
whether a feature set explains the **detrended residual**: the part of an
episode's rating left over after removing the smooth decline-over-time trend.
A positive out-of-fold R² on that residual means the feature beats chronology.

    python scripts/beat_chronology.py

With the full LLM rubric extracted (simpsons.llm_features), this is the headline
number for "did we beat chronology?". With only the committed demonstration
sample, it falls back to reporting correlations on the covered episodes.
"""
from __future__ import annotations

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
from simpsons import text_features as T  # noqa: E402

CV = KFold(n_splits=5, shuffle=True, random_state=42)


def _model():
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_depth=3, max_iter=400,
        l2_regularization=1.0, random_state=42,
    )


def chronology_residual(df: pd.DataFrame) -> np.ndarray:
    """Rating minus the out-of-fold chronology prediction (leak-free detrend)."""
    X = df[["number_in_series", "season"]].to_numpy()
    y = df[D.TARGET].to_numpy()
    trend = cross_val_predict(_model(), X, y, cv=CV)
    return y - trend


def residual_r2(features: np.ndarray, residual: np.ndarray) -> float:
    """Out-of-fold R² of a feature set predicting the residual. >0 beats chronology."""
    return float(cross_val_score(_model(), features, residual, cv=CV, scoring="r2").mean())


def main() -> None:
    df = T.build_text_dataset(n_components=100)
    resid = chronology_residual(df)
    print(f"{len(df)} episodes. Residual std after detrending chronology: {resid.std():.3f}\n")

    feature_sets = {
        "engineered": D.SCRIPT_FEATURES,
        "linguistic": df.attrs["linguistic_cols"],
        "lsa(100d)": df.attrs["lsa_cols"],
    }
    print("Out-of-fold R² explaining the chronology residual (>0 = beats chronology):")
    for name, cols in feature_sets.items():
        r2 = residual_r2(df[cols].to_numpy(), resid)
        print(f"  {name:14s} R² = {r2:+.3f}")

    # LLM rubric features — full-coverage CV if available, else a demo correlation.
    llm = L.load_llm_features()
    if llm.empty:
        print("\nNo LLM features cached. Run: python -m simpsons.llm_features extract")
        return

    idx = df.set_index("id")
    covered = idx.index.isin(llm.index)
    if covered.all():
        feats = idx.join(llm)[L.FEATURE_COLUMNS].to_numpy()
        r2 = residual_r2(feats, resid)
        print(f"  {'llm_rubric':14s} R² = {r2:+.3f}   <-- full-coverage out-of-fold result")
    else:
        sub = idx.loc[covered].join(llm)
        sub_resid = pd.Series(resid, index=idx.index)[covered]
        craft = (sub[["joke_quality", "story_coherence", "story_originality",
                      "satire_sharpness", "memorable_moment_strength",
                      "emotional_resonance"]].mean(axis=1)
                 - sub[["guest_star_reliance", "mean_spiritedness"]].mean(axis=1))
        print(f"\nLLM features cover {covered.sum()}/{len(df)} episodes "
              f"(demonstration sample — extract all to get the CV result above).")
        print(f"  corr(LLM craft, rating)            = "
              f"{np.corrcoef(craft, sub[D.TARGET])[0, 1]:+.3f}")
        print(f"  corr(LLM craft, chronology residual)= "
              f"{np.corrcoef(craft, sub_resid)[0, 1]:+.3f}  <-- signal beyond chronology")


if __name__ == "__main__":
    main()
