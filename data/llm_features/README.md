# LLM-extracted qualitative features

Per-episode rubric scores (0-10) produced by `simpsons.llm_features`.

The 12 files currently here are an **Opus-authored demonstration sample** (`claude-opus-4-8`), scored from the actual episode transcripts as a proof that the rubric discriminates between acclaimed and weak episodes. To extract all ~564 episodes via the Batch API, run `python -m simpsons.llm_features extract` with Anthropic credentials, then `... collect <batch_id>`.
