"""LLM-extracted qualitative features — "can Opus read the script and beat chronology?"

The shallow features (line lengths, location counts) and even LSA embeddings
carry almost no signal about quality once you know the air date. The open
question is whether a *reader* — a model that actually understands the dialogue —
can extract the qualitative things critics actually talk about: joke quality,
heart, satire, mean-spiritedness, story coherence, over-reliance on guest stars.

This module sends each episode's transcript to Claude Opus and asks for a
structured rubric of 0–10 scores. Design notes:

* **Model:** ``claude-opus-4-8`` — we want the strongest reader available.
* **Batch API:** 600 episodes is a textbook batch job (50% cheaper, async).
* **Prompt caching:** the long rubric/system prompt is identical for every
  episode, so it is marked ``cache_control`` once and read ~600 times.
* **Structured outputs:** ``output_config.format`` with a JSON schema pins the
  response shape so parsing never fails. (JSON-schema numeric bounds aren't
  enforced server-side, so we clamp to 0–10 on the way in.)
* **Disk cache:** every result is written to ``data/llm_features/<id>.json`` so
  the (paid) extraction is done exactly once and is fully resumable.

If you have no Anthropic credentials/network, the extraction calls raise a clear
error; the cached JSON files (including the Opus-authored demonstration sample
committed to the repo) are still readable via :func:`load_llm_features`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from . import data as D
from .text_features import episode_documents

MODEL = "claude-opus-4-8"
CACHE_DIR = D.DATA_DIR / "llm_features"

# The rubric. Each dimension is hypothesized to relate to reception *and* to vary
# within an era — i.e. to carry signal chronology cannot. Keep names stable: they
# become feature columns and JSON keys.
RUBRIC: dict[str, str] = {
    "joke_density": "How packed the episode is with jokes/gags (0 sparse … 10 relentless).",
    "joke_quality": "How clever and genuinely funny the humor is, independent of how much there is.",
    "emotional_resonance": "Heart, poignancy, earned emotional beats (the 'Lisa's Substitute' factor).",
    "satire_sharpness": "Quality and bite of social/political/cultural satire.",
    "story_coherence": "How tight and well-constructed the A/B plot is (0 incoherent … 10 airtight).",
    "story_originality": "Freshness of the premise vs. retread/formulaic.",
    "family_centricity": "How much the core Simpson family drives the story vs. side characters.",
    "guest_star_reliance": "How much the episode leans on celebrity guest voices/cameos for value.",
    "cultural_reference_density": "Density of topical/pop-culture references (can date an episode).",
    "mean_spiritedness": "Cynicism/cruelty/characters acting nasty (frequently cited in the decline).",
    "absurdity_level": "Zaniness/cartoon logic vs. grounded character comedy (later seasons skew high).",
    "memorable_moment_strength": "Presence of a standout iconic scene, line, or song.",
}

SYSTEM_PROMPT = (
    "You are a television critic and story analyst with encyclopedic knowledge of "
    "The Simpsons. You will be given the full spoken-dialogue transcript of one "
    "episode. Score it on each rubric dimension as an integer from 0 to 10, judging "
    "only from the transcript (you will not be told the rating, season, or air date — "
    "do not guess them or let them influence you). Be discriminating: use the full "
    "0–10 range across episodes rather than clustering around the middle.\n\nRubric:\n"
    + "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
)

# JSON schema for structured outputs: every rubric key as an integer, plus a
# short rationale. (additionalProperties:false is required for strict schemas.)
_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "integer", "description": v} for k, v in RUBRIC.items()},
        "rationale": {"type": "string", "description": "One sentence justifying the scores."},
    },
    "required": list(RUBRIC) + ["rationale"],
    "additionalProperties": False,
}

FEATURE_COLUMNS = list(RUBRIC)


def _client():
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError("pip install anthropic to run LLM feature extraction") from e
    return anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / auth from env


def _user_prompt(doc: str) -> str:
    # Transcripts are long; trim defensively so a single episode never blows the
    # request size. Opus 4.8 has a 1M window so this is generous.
    return f"Episode transcript:\n\n{doc[:120_000]}"


def _clamp(record: dict) -> dict:
    return {k: max(0, min(10, int(record.get(k, 5)))) for k in RUBRIC} | {
        "rationale": str(record.get("rationale", ""))
    }


# --------------------------------------------------------------------------- #
# Extraction — Batch API (the right tool for 600 one-shot calls)              #
# --------------------------------------------------------------------------- #
def extract_all(limit: int | None = None, only_missing: bool = True) -> Path:
    """Submit every episode to the Batch API and cache results to disk.

    Returns the cache directory. Safe to re-run: by default it skips episodes
    already cached, so an interrupted run resumes for free.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    docs = episode_documents()
    if limit:
        docs = docs.iloc[:limit]

    shared_system = [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
    ]
    requests = []
    for episode_id, doc in docs.items():
        if only_missing and (CACHE_DIR / f"{episode_id}.json").exists():
            continue
        requests.append(Request(
            custom_id=f"ep-{episode_id}",
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=1024,
                system=shared_system,                     # cached across all 600
                output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
                messages=[{"role": "user", "content": _user_prompt(doc)}],
            ),
        ))
    if not requests:
        print("Nothing to extract — all episodes already cached.")
        return CACHE_DIR

    client = _client()
    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch {batch.id} with {len(requests)} episodes. "
          f"Poll with: python -m simpsons.llm_features collect {batch.id}")
    return CACHE_DIR


def collect(batch_id: str) -> int:
    """Fetch a completed batch's results into the disk cache. Returns count written."""
    client = _client()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            record = _clamp(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            continue
        episode_id = int(result.custom_id.removeprefix("ep-"))
        (CACHE_DIR / f"{episode_id}.json").write_text(json.dumps(record, indent=2))
        written += 1
    print(f"Wrote {written} cached feature files to {CACHE_DIR}")
    return written


# --------------------------------------------------------------------------- #
# Loading cached features for the autoresearch harness                         #
# --------------------------------------------------------------------------- #
def load_llm_features() -> pd.DataFrame:
    """Read all cached per-episode JSON into a DataFrame indexed by episode id.

    Returns an empty frame (with the right columns) if nothing is cached yet, so
    the autoresearch registry can simply skip the LLM feature set.
    """
    if not CACHE_DIR.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    rows = {}
    for path in sorted(CACHE_DIR.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rows[int(path.stem)] = {k: rec.get(k) for k in FEATURE_COLUMNS}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FEATURE_COLUMNS)


if __name__ == "__main__":  # tiny CLI: extract / collect <batch_id>
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if cmd == "extract":
        extract_all(limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "collect":
        collect(sys.argv[2])
    else:
        print(__doc__)
