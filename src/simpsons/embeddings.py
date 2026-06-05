"""SOTA semantic embeddings for episode transcripts — pluggable backends.

Why this is a *separate, optional* module: a June-2026 state-of-the-art text
embedding (a sentence-transformer or a hosted embedding API) needs either model
weights downloaded from Hugging Face or an API key. In a locked-down/offline
environment neither is available, so the reproducible analysis falls back to the
LSA embeddings in :mod:`simpsons.text_features`. This module gives you a drop-in
upgrade path that feeds the *exact same* autoresearch harness the moment you run
it somewhere with network access.

Usage::

    from simpsons.embeddings import embed_episodes
    emb = embed_episodes(backend="sentence-transformers",
                         model="BAAI/bge-large-en-v1.5")   # or any 2026 SOTA model
    # emb is a DataFrame indexed by episode_id; pass it into experiments.feature_sets

Backends
--------
* ``sentence-transformers`` — local inference with any model on the MTEB
  leaderboard (e.g. ``BAAI/bge-large-en-v1.5``, ``nvidia/NV-Embed-v2``,
  ``Alibaba-NLP/gte-Qwen2-7B-instruct``). Requires ``pip install
  sentence-transformers`` and weight download.
* ``openai`` — hosted ``text-embedding-3-large`` (or 2026 successor) via the
  ``openai`` SDK and ``OPENAI_API_KEY``.
* ``voyage`` — ``voyage-3``-class embeddings via the ``voyageai`` SDK.

All backends return L2-normalized vectors so cosine geometry is preserved.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .text_features import episode_documents


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def embed_episodes(backend: str = "sentence-transformers",
                   model: str = "BAAI/bge-large-en-v1.5",
                   batch_size: int = 16) -> pd.DataFrame:
    """Return a DataFrame of episode embeddings, indexed by ``episode_id``.

    Raises a clear error (not a silent fallback) if the backend's dependency or
    credentials are missing, so experiments never quietly run on the wrong
    representation.
    """
    docs = episode_documents()
    texts = docs.tolist()

    if backend == "sentence-transformers":
        vecs = _embed_sentence_transformers(texts, model, batch_size)
    elif backend == "openai":
        vecs = _embed_openai(texts, model)
    elif backend == "voyage":
        vecs = _embed_voyage(texts, model)
    else:
        raise ValueError(f"Unknown backend {backend!r}")

    vecs = _normalize(np.asarray(vecs, dtype=np.float32))
    cols = [f"emb_{i}" for i in range(vecs.shape[1])]
    return pd.DataFrame(vecs, index=docs.index, columns=cols)


def _embed_sentence_transformers(texts, model, batch_size):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pip install sentence-transformers to use the local SOTA backend"
        ) from e
    encoder = SentenceTransformer(model)
    return encoder.encode(texts, batch_size=batch_size, show_progress_bar=True,
                          normalize_embeddings=False)


def _embed_openai(texts, model):  # pragma: no cover - needs network + key
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai to use the OpenAI backend") from e
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("set OPENAI_API_KEY for the OpenAI backend")
    client = OpenAI()
    model = model if model.startswith("text-embedding") else "text-embedding-3-large"
    out = []
    for i in range(0, len(texts), 64):
        chunk = [t[:8000] for t in texts[i:i + 64]]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend(d.embedding for d in resp.data)
    return out


def _embed_voyage(texts, model):  # pragma: no cover - needs network + key
    try:
        import voyageai
    except ImportError as e:
        raise RuntimeError("pip install voyageai to use the Voyage backend") from e
    client = voyageai.Client()
    model = model if model.startswith("voyage") else "voyage-3"
    out = []
    for i in range(0, len(texts), 64):
        res = client.embed(texts[i:i + 64], model=model, input_type="document")
        out.extend(res.embeddings)
    return out
