"""Text representations of each episode's dialogue.

Two families are produced here, both computed only from the spoken dialogue so
they remain leak-free:

1. **Linguistic features** — interpretable signals (lexical diversity, question
   and exclamation rates, sentiment proxy, dialogue pacing).
2. **LSA "semantic" embeddings** — TF-IDF over the full episode transcript
   reduced with Truncated SVD. This is the classical, fully-offline cousin of a
   modern sentence-transformer embedding, and serves as the reproducible
   baseline that the SOTA embedder in :mod:`simpsons.embeddings` is benchmarked
   against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from . import data as D

# Tiny, dependency-free polarity lexicon — enough for a coarse sentiment proxy
# without pulling NLTK/VADER data over a blocked network.
_POS = {
    "love", "great", "good", "happy", "wonderful", "best", "fun", "nice",
    "amazing", "beautiful", "win", "hope", "friend", "smile", "laugh", "yes",
}
_NEG = {
    "hate", "bad", "terrible", "awful", "worst", "sad", "angry", "stupid",
    "kill", "die", "no", "never", "ugly", "lose", "cry", "horrible", "fear",
}


def episode_documents() -> pd.Series:
    """Concatenate each episode's normalized dialogue into a single document."""
    sc = D.load_script_lines()
    spoken = sc[sc["is_speaking"]].copy()
    text = spoken["normalized_text"].astype("string").fillna("")
    return text.groupby(spoken["episode_id"]).apply(lambda s: " ".join(s)).rename("doc")


def linguistic_features(docs: pd.Series | None = None) -> pd.DataFrame:
    """Interpretable per-episode linguistic signals."""
    if docs is None:
        docs = episode_documents()
    sc = D.load_script_lines()
    spoken = sc[sc["is_speaking"]]

    rows = {}
    for eid, doc in docs.items():
        words = doc.split()
        n = len(words) or 1
        uniq = len(set(words))
        rows[eid] = {
            "type_token_ratio": uniq / n,
            "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
            "pos_rate": sum(w in _POS for w in words) / n,
            "neg_rate": sum(w in _NEG for w in words) / n,
        }
    feats = pd.DataFrame.from_dict(rows, orient="index")

    raw = sc[sc["is_speaking"]]["raw_text"].astype(str)
    by = raw.groupby(spoken["episode_id"])
    feats["question_rate"] = by.apply(lambda s: s.str.contains(r"\?").mean())
    feats["exclaim_rate"] = by.apply(lambda s: s.str.contains("!").mean())
    feats["sentiment"] = feats["pos_rate"] - feats["neg_rate"]
    return feats


def lsa_embeddings(docs: pd.Series | None = None, n_components: int = 100,
                   random_state: int = 42) -> pd.DataFrame:
    """TF-IDF + Truncated SVD (LSA) embedding of each episode transcript."""
    if docs is None:
        docs = episode_documents()
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_features=20000, ngram_range=(1, 2),
        min_df=3, max_df=0.6, stop_words="english",
    )
    svd = make_pipeline(
        TruncatedSVD(n_components=n_components, random_state=random_state),
        Normalizer(copy=False),
    )
    tfidf = vectorizer.fit_transform(docs.values)
    emb = svd.fit_transform(tfidf)
    cols = [f"lsa_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=docs.index, columns=cols)


def build_text_dataset(n_components: int = 100) -> pd.DataFrame:
    """Episodes joined to linguistic + LSA features, target retained, no leak."""
    base = D.build_dataset().set_index("id")
    docs = episode_documents()
    ling = linguistic_features(docs)
    lsa = lsa_embeddings(docs, n_components=n_components)
    out = base.join(ling, how="left").join(lsa, how="left")
    feat_cols = list(ling.columns) + list(lsa.columns)
    out[feat_cols] = out[feat_cols].fillna(out[feat_cols].median())
    out.attrs["linguistic_cols"] = list(ling.columns)
    out.attrs["lsa_cols"] = list(lsa.columns)
    return out.reset_index()
