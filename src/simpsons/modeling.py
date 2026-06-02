"""Leak-free model evaluation and training.

Everything here answers one question honestly: *how much can we predict an
episode's IMDb rating, and from what?* We compare a naive baseline against
models trained on script-only, time-only, and combined feature sets using a
single shared cross-validation splitter so the numbers are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split

from . import data as D

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_model() -> HistGradientBoostingRegressor:
    """The model used throughout.

    A single, modern gradient-boosting regressor replaces the original
    AdaBoost->GradientBoosting stack. The stack added complexity and a leak
    (the meta-model was fed in-sample base predictions); honest out-of-fold
    evaluation shows a single tuned booster matches it without the fragility.
    """
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_depth=3,
        max_iter=400,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


@dataclass
class CVResult:
    name: str
    rmse_mean: float
    rmse_std: float

    def __str__(self) -> str:
        return f"{self.name:<28} RMSE {self.rmse_mean:.3f} +/- {self.rmse_std:.3f}"


def _rmse_cv(model, X, y) -> tuple[float, float]:
    scores = cross_val_score(model, X, y, cv=CV, scoring="neg_root_mean_squared_error")
    return float(-scores.mean()), float(scores.std())


def compare_feature_sets(df: pd.DataFrame) -> list[CVResult]:
    """Quantify how much signal lives in scripts vs. chronology."""
    y = df[D.TARGET].to_numpy()
    runs = [
        ("Baseline (predict mean)", DummyRegressor(strategy="mean"), D.TIME_FEATURES),
        ("Script features only", make_model(), D.SCRIPT_FEATURES),
        ("Time features only", make_model(), D.TIME_FEATURES),
        ("Script + Time (full)", make_model(), D.ALL_FEATURES),
    ]
    results = []
    for name, model, feats in runs:
        mean, std = _rmse_cv(model, df[feats].to_numpy(), y)
        results.append(CVResult(name, mean, std))
    return results


def out_of_fold_predictions(df: pd.DataFrame, features=D.ALL_FEATURES) -> np.ndarray:
    """Honest predicted ratings: each episode predicted by a fold it was held out of."""
    X, y = D.xy(df, features)
    return cross_val_predict(make_model(), X, y, cv=CV)


def permutation_importances(df: pd.DataFrame, features=D.ALL_FEATURES) -> pd.DataFrame:
    """Permutation importance on a held-out split, in RMSE units."""
    X, y = D.xy(df, features)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    model = make_model().fit(X_tr, y_tr)
    pi = permutation_importance(
        model, X_te, y_te, n_repeats=30, random_state=0,
        scoring="neg_root_mean_squared_error",
    )
    return (
        pd.DataFrame({
            "feature": features,
            "rmse_increase": pi.importances_mean,
            "std": pi.importances_std,
        })
        .sort_values("rmse_increase", ascending=False)
        .reset_index(drop=True)
    )
