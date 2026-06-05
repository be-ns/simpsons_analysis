"""Feature sets, a model zoo, and the search primitives used by autoresearch.

The design follows Andrej Karpathy's "recipe for training neural nets":
establish a dead-simple baseline first, change one thing at a time, and above
all *don't fool yourself* — every number here is an honest cross-validated
estimate, and the eventual winner is re-scored with nested CV so the
hyperparameter search cannot leak into the reported RMSE. (Peeking at the
holdout is precisely what inflated the original project's 0.351.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D

RANDOM_STATE = 42
OUTER = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
INNER = KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Feature sets — built lazily so we only compute the expensive text ones once. #
# --------------------------------------------------------------------------- #
def feature_sets() -> dict[str, pd.DataFrame]:
    """Return {name: DataFrame} of candidate feature matrices sharing an index.

    All frames are aligned to the rated-episode index and carry a ``__target__``
    column so callers never have to re-join.
    """
    from . import text_features as T

    df = T.build_text_dataset(n_components=100)
    y = df[D.TARGET]
    ling = df.attrs["linguistic_cols"]
    lsa = df.attrs["lsa_cols"]
    chrono = D.TIME_FEATURES
    engineered = D.SCRIPT_FEATURES

    def pack(cols):
        out = df[cols].copy()
        out["__target__"] = y.values
        return out

    sets = {
        "chrono_only": pack(chrono),
        "engineered": pack(engineered),
        "engineered+chrono": pack(engineered + chrono),
        "linguistic+chrono": pack(ling + chrono),
        "lsa_only": pack(lsa),
        "lsa+chrono": pack(lsa + chrono),
        "all_text+chrono": pack(engineered + ling + lsa + chrono),
    }

    # Register LLM rubric features — but only when they cover every rated episode.
    # The committed demonstration sample covers a handful, so this stays out of the
    # leaderboard until you run the full batch extraction (see simpsons.llm_features).
    from . import llm_features as Lf

    llm = Lf.load_llm_features()
    covered = df["id"].isin(llm.index)
    if covered.all() and len(llm.columns):
        joined = df.set_index("id").join(llm)
        for col in Lf.FEATURE_COLUMNS:
            df[col] = joined[col].to_numpy()
        sets["llm_only"] = pack(Lf.FEATURE_COLUMNS)
        sets["llm+chrono"] = pack(Lf.FEATURE_COLUMNS + chrono)
        sets["all+llm"] = pack(engineered + ling + lsa + Lf.FEATURE_COLUMNS + chrono)
    return sets


# --------------------------------------------------------------------------- #
# Model zoo — each entry: (estimator factory, param distribution, needs_scaling)#
# --------------------------------------------------------------------------- #
def model_zoo() -> dict[str, tuple]:
    return {
        "ridge": (Ridge(), {"model__alpha": loguniform(1e-2, 1e3)}, True),
        "elasticnet": (
            ElasticNet(max_iter=5000),
            {"model__alpha": loguniform(1e-3, 1e1), "model__l1_ratio": [0.1, 0.5, 0.9]},
            True,
        ),
        "knn": (
            KNeighborsRegressor(),
            {"model__n_neighbors": randint(3, 30), "model__weights": ["uniform", "distance"]},
            True,
        ),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": randint(200, 600), "model__max_depth": randint(2, 8),
             "model__min_samples_leaf": randint(1, 8)},
            False,
        ),
        "hist_gbm": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__learning_rate": loguniform(1e-2, 3e-1), "model__max_depth": randint(2, 5),
             "model__max_iter": randint(150, 600), "model__l2_regularization": loguniform(1e-2, 1e1)},
            False,
        ),
        "mlp": (  # a small neural net
            MLPRegressor(random_state=RANDOM_STATE, max_iter=1500, early_stopping=True),
            {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
             "model__alpha": loguniform(1e-5, 1e-1),
             "model__learning_rate_init": loguniform(1e-4, 1e-2)},
            True,
        ),
    }


def _pipeline(estimator, needs_scaling: bool) -> Pipeline:
    steps = []
    if needs_scaling:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


@dataclass
class Trial:
    feature_set: str
    model: str
    cv_rmse: float
    best_params: dict


def search_one(X: np.ndarray, y: np.ndarray, estimator, params: dict,
               needs_scaling: bool, n_iter: int = 25) -> tuple[float, dict, object]:
    """Randomized hyperparameter search; returns (cv_rmse, best_params, fitted)."""
    pipe = _pipeline(estimator, needs_scaling)
    search = RandomizedSearchCV(
        pipe, params, n_iter=n_iter, cv=INNER, n_jobs=-1,
        scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE,
    )
    search.fit(X, y)
    return -search.best_score_, search.best_params_, search.best_estimator_


def nested_cv_rmse(X: np.ndarray, y: np.ndarray, estimator, params: dict,
                   needs_scaling: bool, n_iter: int = 25) -> tuple[float, float]:
    """Unbiased RMSE: hyperparameter search runs *inside* each outer fold."""
    pipe = _pipeline(estimator, needs_scaling)
    search = RandomizedSearchCV(
        pipe, params, n_iter=n_iter, cv=INNER, n_jobs=-1,
        scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE,
    )
    scores = cross_val_score(search, X, y, cv=OUTER, scoring="neg_root_mean_squared_error")
    return float(-scores.mean()), float(scores.std())
