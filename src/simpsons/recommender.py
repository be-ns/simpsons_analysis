"""Transparent, content-based episode recommender.

This replaces the original "preference funnel" — a chain of ``sort_values``
slices (top-40 then top-20 then top-5) whose output depended entirely on the
order the filters happened to be applied. That design quietly dropped episodes
and was impossible to explain.

The rebuild is a cold-start, content-based ranker: every episode gets a
transparent match score that is a weighted sum of normalized preference
signals. There are no hidden cut-offs, every component is inspectable, and the
ranking is fully reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as D

# Keywords used to score "political" episodes from the raw dialogue.
POLITICS_TERMS = [
    "president", "politics", "political", "election", "congress", "senator",
    "immigration", "campaign", "vote", "mayor", "government", "war",
]
SONG_TERMS = ["sing", "singing", "song", "musical"]


def _normalize(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]; constant columns map to 0."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def build_episode_profiles() -> pd.DataFrame:
    """One row per episode with the signals the recommender ranks on."""
    ep = D.load_episodes()
    sc = D.load_script_lines()
    spoken = sc[sc["is_speaking"]].copy()
    text = spoken["raw_text"].astype(str).str.lower()
    char = spoken["raw_character_text"].astype(str).str.lower()

    song_hits = text.str.contains("|".join(SONG_TERMS)) | char.str.contains("singer")
    politics_hits = text.str.contains("|".join(POLITICS_TERMS))

    by_ep = spoken.assign(_song=song_hits.values, _pol=politics_hits.values).groupby("episode_id")
    profiles = pd.DataFrame({
        "n_lines": by_ep.size(),
        "song_score": by_ep["_song"].sum(),
        "politics_score": by_ep["_pol"].sum(),
    })
    profiles["song_density"] = profiles["song_score"] / profiles["n_lines"]
    profiles["politics_density"] = profiles["politics_score"] / profiles["n_lines"]

    out = ep.merge(profiles, left_on="id", right_index=True, how="inner")
    out = out.dropna(subset=["imdb_rating"]).reset_index(drop=True)
    # Cache the line table so per-character queries don't re-read 158K rows.
    out.attrs["spoken"] = spoken[["episode_id", "raw_character_text", "raw_location_text"]]
    return out


def _entity_share(spoken: pd.DataFrame, column: str, query: str) -> pd.Series:
    """Share of each episode's lines whose ``column`` contains ``query``."""
    col = spoken[column].astype(str).str.lower()
    hit = col.str.contains(query.lower().strip(), na=False)
    grp = spoken.assign(_hit=hit.values).groupby("episode_id")
    return (grp["_hit"].sum() / grp.size()).rename("share")


@dataclass
class Preferences:
    """A user's cold-start preferences. Any field left ``None`` is ignored."""
    character: str | None = None
    location: str | None = None
    wants_song: bool | None = None
    wants_politics: bool | None = None
    # How much to weight a high IMDb rating vs. preference match (0..1).
    quality_weight: float = 0.25
    weights: dict = field(default_factory=lambda: {
        "character": 1.0, "location": 0.8, "song": 0.5, "politics": 0.5,
    })


def recommend(prefs: Preferences, profiles: pd.DataFrame | None = None, top_n: int = 5) -> pd.DataFrame:
    """Rank episodes by transparent preference match, best first.

    Returns the top ``top_n`` episodes with a ``match_score`` plus every
    component that fed it, so any recommendation can be explained at a glance.
    """
    if profiles is None:
        profiles = build_episode_profiles()
    spoken = profiles.attrs["spoken"]
    df = profiles.copy()

    components: dict[str, pd.Series] = {}
    if prefs.character:
        share = _entity_share(spoken, "raw_character_text", prefs.character)
        components["character"] = df["id"].map(share).fillna(0.0)
    if prefs.location:
        share = _entity_share(spoken, "raw_location_text", prefs.location)
        components["location"] = df["id"].map(share).fillna(0.0)
    if prefs.wants_song is not None:
        s = _normalize(df["song_density"])
        components["song"] = s if prefs.wants_song else (1 - s)
    if prefs.wants_politics is not None:
        s = _normalize(df["politics_density"])
        components["politics"] = s if prefs.wants_politics else (1 - s)

    pref_score = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for name, comp in components.items():
        w = prefs.weights.get(name, 1.0)
        df[f"match_{name}"] = comp.values
        pref_score += w * _normalize(comp).values
        total_w += w
    pref_score = pref_score / total_w if total_w else pref_score

    quality = _normalize(df["imdb_rating"])
    df["match_score"] = (1 - prefs.quality_weight) * pref_score + prefs.quality_weight * quality

    cols = ["title", "season", "number_in_season", "imdb_rating", "match_score"]
    cols += [c for c in df.columns if c.startswith("match_") and c != "match_score"]
    return df.sort_values("match_score", ascending=False).head(top_n)[cols].reset_index(drop=True)
