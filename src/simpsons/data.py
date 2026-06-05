"""Data loading and feature engineering for the Simpsons ratings analysis.

The single most important rule in this module: features are built only from
information that would be available *before* an episode airs (script content +
air date). The target (``imdb_rating``) is never imputed or used as an input,
and episodes with a missing rating are dropped rather than filled — imputing the
target was one of the leaks in the original 2017 pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EPISODES_CSV = DATA_DIR / "simpsons_episodes.csv"
SCRIPT_LINES_CSV = DATA_DIR / "simpsons_script_lines.csv"

# Core family surname used to compute the "screen time" ratio.
FAMILY = "Simpson"

# Feature groups, kept explicit so we can evaluate them in isolation.
SCRIPT_FEATURES = [
    "avg_line_len",
    "max_line_len",
    "n_speaking_lines",
    "n_locations",
    "n_characters",
    "family_line_ratio",
    "title_len",
    "election_year",
]
TIME_FEATURES = ["number_in_series", "season"]
ALL_FEATURES = SCRIPT_FEATURES + TIME_FEATURES
TARGET = "imdb_rating"


def load_episodes() -> pd.DataFrame:
    """Load the episode-level table with parsed dates."""
    ep = pd.read_csv(EPISODES_CSV)
    ep["original_air_date"] = pd.to_datetime(ep["original_air_date"], errors="coerce")
    return ep


def load_script_lines() -> pd.DataFrame:
    """Load the 158K-row script-line table, coercing the messy columns.

    The raw file has a handful of rows shifted by stray double quotes, which
    leaves non-numeric junk in ``word_count`` and free text in
    ``speaking_line``. We coerce rather than crash, so a few corrupt rows simply
    contribute nothing instead of derailing the pipeline.
    """
    sc = pd.read_csv(SCRIPT_LINES_CSV, low_memory=False)
    sc["word_count"] = pd.to_numeric(sc["word_count"], errors="coerce")
    sc["is_speaking"] = sc["speaking_line"].astype(str).str.lower().eq("true")
    sc["raw_character_text"] = sc["raw_character_text"].astype("string")
    return sc


def _episode_script_features(sc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate script lines into one row of features per episode."""
    spoken = sc[sc["is_speaking"]].copy()
    grp = spoken.groupby("episode_id")

    feats = pd.DataFrame(
        {
            "avg_line_len": grp["word_count"].mean(),
            "max_line_len": grp["word_count"].max(),
            "n_speaking_lines": grp.size(),
            "n_locations": grp["location_id"].nunique(),
            "n_characters": grp["character_id"].nunique(),
        }
    )

    # Share of speaking lines delivered by the core family — the headline
    # "screen time" feature from the original project.
    is_family = spoken["raw_character_text"].str.contains(FAMILY, case=False, na=False)
    family_lines = spoken.assign(is_family=is_family).groupby("episode_id")["is_family"].sum()
    feats["family_line_ratio"] = (family_lines / feats["n_speaking_lines"]).fillna(0.0)
    return feats


def build_dataset() -> pd.DataFrame:
    """Return one modelling-ready row per rated episode.

    Episodes with no IMDb rating are dropped. Feature NaNs (a handful of
    episodes with no usable script rows) are filled with the column median;
    crucially the *target* is never imputed.
    """
    ep = load_episodes()
    sc = load_script_lines()
    feats = _episode_script_features(sc)

    df = ep.merge(feats, left_on="id", right_index=True, how="left")
    df["title_len"] = df["title"].str.len()
    # US presidential elections fall on years divisible by four.
    df["election_year"] = (df["original_air_date"].dt.year % 4 == 0).astype(int)

    df = df.dropna(subset=[TARGET]).copy()
    for col in SCRIPT_FEATURES + TIME_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df.sort_values("number_in_series").reset_index(drop=True)


def xy(df: pd.DataFrame, features: list[str] = ALL_FEATURES):
    """Split a built dataset into a feature matrix and target vector."""
    return df[features].to_numpy(), df[TARGET].to_numpy()


def era(season: "pd.Series | np.ndarray", cutoff: int = 10) -> np.ndarray:
    """Label episodes as 'Golden era (S1-10)' or 'Later (S11+)'."""
    season = np.asarray(season)
    return np.where(season <= cutoff, f"Golden era (S1-{cutoff})", f"Later (S{cutoff + 1}+)")
