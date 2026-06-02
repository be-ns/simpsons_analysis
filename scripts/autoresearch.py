"""Autoresearch: a Karpathy-style automated search for the lowest honest RMSE.

    python scripts/autoresearch.py

The loop, in the spirit of "A Recipe for Training Neural Networks":

  1. Start with a dumb baseline (predict the mean) and a strong simple baseline
     (chronology only) so we always know what "good" means.
  2. Sweep every feature representation x every model family, tuning each with an
     inner cross-validated randomized search. One thing changes at a time.
  3. Rank the full leaderboard by cross-validated RMSE.
  4. Re-score the winner with *nested* CV — the only number that hasn't seen its
     own hyperparameter search — so we don't fool ourselves. This is the guard
     the original overnight while-loop lacked.

Outputs reports/autoresearch_leaderboard.csv and reports/autoresearch.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simpsons import experiments as E  # noqa: E402

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def main() -> None:
    t0 = time.time()
    print("Building feature representations (incl. LSA embeddings)...")
    fsets = E.feature_sets()
    any_set = next(iter(fsets.values()))
    y = any_set["__target__"].to_numpy()

    dumb = -cross_val_score(DummyRegressor(strategy="mean"),
                            np.zeros((len(y), 1)), y,
                            cv=E.OUTER, scoring="neg_root_mean_squared_error").mean()
    print(f"Dumb baseline (predict mean): RMSE {dumb:.3f}\n")

    zoo = E.model_zoo()
    trials: list[E.Trial] = []
    print(f"Searching {len(fsets)} feature sets x {len(zoo)} models "
          f"(n_iter={N_ITER} each)...")
    for fname, frame in fsets.items():
        X = frame.drop(columns="__target__").to_numpy()
        for mname, (est, params, scale) in zoo.items():
            rmse, best, _ = E.search_one(X, y, est, params, scale, n_iter=N_ITER)
            trials.append(E.Trial(fname, mname, rmse, best))
            print(f"  {fname:<18} {mname:<14} CV-RMSE {rmse:.3f}")

    board = (
        pd.DataFrame([{"feature_set": t.feature_set, "model": t.model,
                       "cv_rmse": round(t.cv_rmse, 4)} for t in trials])
        .sort_values("cv_rmse").reset_index(drop=True)
    )
    board.to_csv(ROOT / "reports" / "autoresearch_leaderboard.csv", index=False)
    try:
        from simpsons import viz
        (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
        viz.autoresearch_leaderboard(board, ROOT / "reports" / "figures" / "autoresearch_leaderboard.png")
    except Exception as exc:  # plotting is a nicety, never fail the search over it
        print(f"  (skipped leaderboard figure: {exc})")

    print("\n=== Leaderboard (top 8 by inner-CV RMSE) ===")
    print(board.head(8).to_string(index=False))

    # Honest re-scoring of the winner with nested CV.
    best_trial = min(trials, key=lambda t: t.cv_rmse)
    Xw = fsets[best_trial.feature_set].drop(columns="__target__").to_numpy()
    est, params, scale = zoo[best_trial.model]
    print(f"\nNested-CV re-scoring winner: {best_trial.feature_set} + {best_trial.model} ...")
    nested_mean, nested_std = E.nested_cv_rmse(Xw, y, est, params, scale, n_iter=N_ITER)

    summary = {
        "dumb_baseline_rmse": round(float(dumb), 3),
        "winner": {"feature_set": best_trial.feature_set, "model": best_trial.model},
        "winner_inner_cv_rmse": round(best_trial.cv_rmse, 3),
        "winner_nested_cv_rmse": round(nested_mean, 3),
        "winner_nested_cv_std": round(nested_std, 3),
        "leaderboard": board.to_dict(orient="records"),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (ROOT / "reports" / "autoresearch.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWinner: {best_trial.feature_set} + {best_trial.model}")
    print(f"  inner-CV RMSE  {best_trial.cv_rmse:.3f}   "
          f"nested-CV RMSE {nested_mean:.3f} +/- {nested_std:.3f}")
    print(f"Done in {summary['runtime_sec']}s. Wrote reports/autoresearch.json")


if __name__ == "__main__":
    main()
