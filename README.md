# Can a Simpsons script tell you how good the episode is?

**Short answer: no — and proving that rigorously is the interesting part.**

This repository started life in 2017 as a portfolio project that claimed to
*predict an episode's IMDb rating from its script* with an RMSE of 0.351. This
is a 2026 rebuild. With leak-free methodology and an automated model search, the
honest answer turns out to be very different — and, I'd argue, far more
interesting than the original headline.

> **Headline finding.** Across **42 model × feature-set combinations** — shallow
> engineered features, hand-built linguistic features, and TF-IDF/LSA *semantic
> embeddings* of the full dialogue, run through linear models, tree ensembles
> and a neural net — the best honest (nested-CV) error is **RMSE ≈ 0.44**. A
> model that knows *only when the episode aired* scores **0.436**. Everything the
> scripts add on top of that is **< 0.005 RMSE — statistical noise.** The
> Simpsons' rating is almost entirely a function of *its decline over time*, not
> the content of any individual episode.

![IMDb rating over 27 years](reports/figures/rating_over_time.png)

---

## TL;DR for the busy reviewer

| | Original (2017) | This rebuild (2026) |
|---|---|---|
| Reported RMSE | 0.351 | **0.439 ± 0.050** (nested CV) |
| How it was obtained | overnight `while` loop saving the best **holdout** draw | nested cross-validation; the search never sees its own test fold |
| Target leakage | `imdb_rating` NaNs imputed with the mean, then scored | episodes with no rating are **dropped**, never imputed |
| Stacking | AdaBoost → GBM fed **in-sample** base predictions | single tuned booster; stacking gave no honest lift |
| Headline claim | "scripts predict ratings" | scripts add **< 0.005 RMSE** beyond air date |
| Runs today? | ❌ imports the long-removed `sklearn.externals.joblib` | ✅ scikit-learn ≥ 1.5, one `pip install` |

The original code is preserved unchanged in [`legacy/`](legacy/) so the before/after is auditable.

---

## What the data actually says

The dataset is [The Simpsons by the Data](https://data.world/data-society/the-simpsons-by-the-data):
**600 episodes** (597 with an IMDb rating) and **158,314 script lines**.

Three facts drive everything:

1. **Ratings are dominated by time.** Rating correlates **−0.75** with episode
   order. The "golden era" (S1–10) averages **8.1**; season 11 onward averages **7.0**.
2. **The naive baseline is already hard to beat.** Ratings have a standard
   deviation of **0.73**, so "always predict the mean" gives RMSE ≈ 0.73. Any
   honest model has only ~0.73 of headroom to work with.
3. **Script-derived features barely move that baseline.** On their own they get
   to ~0.68 — a rounding error better than guessing.

![Where the signal lives](reports/figures/model_comparison.png)

Permutation importance makes it unambiguous: shuffle `number_in_series` and the
model falls apart; shuffle any script feature and nothing happens.

![Permutation importance](reports/figures/permutation_importance.png)

The script features themselves are flat clouds against rating — there is simply
no relationship to learn:

![Script features vs rating](reports/figures/script_features_vs_rating.png)

---

## Autoresearch: an automated, honest model search

Per Andrej Karpathy's *[A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)* —
establish a dumb baseline, change one thing at a time, and **don't fool
yourself** — `scripts/autoresearch.py` sweeps every feature representation
against every model family, tuning each with an inner cross-validated search,
then **re-scores the winner with nested CV** so hyperparameter selection can't
leak into the reported number. (That guard is exactly what the original
overnight loop lacked.)

**7 feature sets × 6 model families = 42 honestly-scored pipelines:**

![Autoresearch leaderboard](reports/figures/autoresearch_leaderboard.png)

Read the heatmap top-to-bottom and the conclusion jumps out:

- **Anything green requires chronology.** The `engineered`-only row (no time) is
  uniformly orange/red. The moment you add `chrono`, every text variant
  collapses into the same ~0.44 band.
- **Semantic embeddings do extract real signal — just redundant signal.** LSA
  embeddings *alone* reach **0.536**, comfortably beating the 0.73 baseline and
  the 0.68 of shallow features. So dialogue genuinely carries information about
  quality — but it's information chronology already encodes, so it adds nothing
  on top.
- **The neural net is the worst model on the board** (0.49 → 1.12). With only
  564 examples and 100-dim inputs, an MLP overfits; this is a small-data regime
  where Karpathy's "don't be a hero" rule favours regularized linear models and
  boosted trees.
- **Winner: `engineered + chrono` + HistGradientBoosting**, inner-CV RMSE
  **0.435**, **nested-CV RMSE 0.439 ± 0.050**. `chrono_only` scores 0.436. The
  gap between them — and between the optimistic 0.435 and the honest 0.439 — is
  the search-optimism the original project mistook for a real result.

![Out-of-fold predictions](reports/figures/predicted_vs_actual.png)

The out-of-fold predictions track the multi-year *trend* beautifully and are
essentially blind to episode-to-episode variation — visual confirmation that the
model is a chronology estimator wearing a script-analysis costume.

---

## State-of-the-art embeddings (a pluggable upgrade path)

`src/simpsons/embeddings.py` ships a drop-in interface for June-2026 SOTA text
embeddings — local `sentence-transformers` (e.g. `BAAI/bge-large-en-v1.5`,
`nvidia/NV-Embed-v2`), or hosted `text-embedding-3-large` / `voyage-3`. They feed
the *exact same* autoresearch harness:

```python
from simpsons.embeddings import embed_episodes
emb = embed_episodes(backend="sentence-transformers", model="BAAI/bge-large-en-v1.5")
```

The reproducible analysis above uses **TF-IDF/LSA embeddings** rather than a
transformer for one honest reason: this environment's network policy blocks the
Hugging Face and embedding-API hosts, so transformer weights can't be fetched
here. The result it would test, though, is well-supported by what we *can* run:
classical semantic embeddings already recover all the signal chronology
provides, so a heavier encoder is overwhelmingly likely to confirm the same
ceiling — a great hypothesis to validate the moment you run it somewhere with
network access.

---

## The recommender, rebuilt

The original "recommender" was a **preference funnel**: a chain of `sort_values`
slices (`[:40]` → `[:20]` → `[:5]`) whose output depended on the order the
filters happened to be applied, silently dropping episodes along the way.

[`src/simpsons/recommender.py`](src/simpsons/recommender.py) replaces it with a
transparent, cold-start, content-based ranker. Every episode gets a `match_score`
that is an inspectable weighted blend of how well it matches your stated
preferences (favourite character, location, songs, politics) plus a small nudge
toward higher-rated episodes — **no hidden cut-offs, every component exposed.**

```text
Preferences(character="Lisa", wants_song=True, wants_politics=True)
  →  title                 season  imdb  match_score  char  song  politics
     My Sister, My Sitter      8    8.1     0.581     0.32  0.02   0.06
     Lisa's Wedding            6    8.3     0.577     0.30  0.01   0.13
     Sideshow Bob Roberts      6    8.3     0.576     0.16  0.05   0.93
```

A minimal Flask demo (`web_app.py`) serves it live; it scores episodes on the
fly, so there's no opaque pre-computed hash table to keep in sync.

---

## Reproduce everything

```bash
pip install -r requirements.txt

python scripts/run_analysis.py     # honest metrics + 5 core figures  → reports/
python scripts/autoresearch.py     # the 42-pipeline search           → reports/
python scripts/train_model.py      # persist the rating model         → models/
python web_app.py                  # the recommender demo at :8080
```

Everything is deterministic given the fixed seeds. Machine-readable results land
in [`reports/metrics.json`](reports/metrics.json),
[`reports/autoresearch.json`](reports/autoresearch.json), and
[`reports/autoresearch_leaderboard.csv`](reports/autoresearch_leaderboard.csv).

```
src/simpsons/
  data.py          leak-free loading + feature engineering
  text_features.py linguistic features + LSA semantic embeddings
  embeddings.py    SOTA embedding backends (sentence-transformers / OpenAI / Voyage)
  modeling.py      honest CV, baselines, permutation importance
  experiments.py   feature sets + model zoo + nested-CV search
  recommender.py   transparent content-based ranker
  viz.py           figure generation
scripts/           run_analysis.py · autoresearch.py · train_model.py
reports/           metrics, leaderboard, figures
legacy/            the original 2017 project, untouched
```

---

## Honest limitations & next steps

- **The ceiling is real, not a modelling failure.** Per-episode quality is
  dominated by writing, voice acting, and direction — none of which survive in a
  line-delimited transcript. No feature set here will break ~0.44 because the
  information isn't in the data.
- **Detrend, then ask the real question.** The genuinely interesting target is
  the *residual* after removing the time trend: *given its era, what made an
  episode over- or under-perform?* That's where SOTA embeddings might finally earn
  their keep, and it's the experiment I'd run next.
- **Validate with a transformer encoder** on a networked machine via the
  embeddings module, and add guest-star and writer/director metadata (not in the
  current dataset) — plausibly the only features with real residual signal.

---

*Data: [The Simpsons by the Data](https://data.world/data-society/the-simpsons-by-the-data).
Original concept inspired by Todd Schneider's
[The Simpsons by the Data](https://toddwschneider.com/posts/the-simpsons-by-the-data/).
The 2017 implementation is preserved in [`legacy/`](legacy/).*
