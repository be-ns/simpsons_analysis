# LLM-extracted qualitative features

One JSON file per episode (`<id>.json`): 12 rubric scores (0–10) plus a rationale,
matching the schema in `src/simpsons/llm_features.py`.

These **299 files** were produced by **Claude subagents scoring each transcript blind**
— the scorer was given only the spoken dialogue, never the rating, season, or air
date — across a sample spread evenly over all 28 seasons (282 are used in the
evaluation after dropping ~17 empty upstream transcripts). Evaluate them with
`python scripts/eval_llm.py` (see `reports/llm_evaluation.json`).

To regenerate at full coverage with a single consistent scorer, use the hosted
Batch API path: `python -m simpsons.llm_features extract` then `... collect <batch_id>`
(requires Anthropic credentials). A few source transcripts are empty in the
upstream data and are dropped during evaluation.
