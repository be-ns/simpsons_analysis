"""Reproduce the entire analysis: metrics + figures, from raw CSVs.

    python scripts/run_analysis.py

Writes figures to reports/figures/ and a machine-readable summary to
reports/metrics.json. Everything is deterministic given the fixed seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simpsons import data as D  # noqa: E402
from simpsons import modeling as M  # noqa: E402
from simpsons import viz  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Building dataset from raw script lines...")
    df = D.build_dataset()
    print(f"  {len(df)} rated episodes, {len(D.ALL_FEATURES)} features")

    print("Comparing feature sets (5-fold CV)...")
    results = M.compare_feature_sets(df)
    for r in results:
        print("  ", r)

    print("Out-of-fold predictions + permutation importance...")
    oof = M.out_of_fold_predictions(df)
    oof_rmse = float(np.sqrt(mean_squared_error(df[D.TARGET], oof)))
    imp = M.permutation_importances(df)

    print("Rendering figures...")
    viz.rating_over_time(df, FIG_DIR / "rating_over_time.png")
    viz.model_comparison(results, FIG_DIR / "model_comparison.png")
    viz.permutation_importance_fig(imp, FIG_DIR / "permutation_importance.png")
    viz.predicted_vs_actual(df, oof, oof_rmse, FIG_DIR / "predicted_vs_actual.png")
    viz.script_features_vs_rating(df, FIG_DIR / "script_features_vs_rating.png")

    metrics = {
        "n_episodes": int(len(df)),
        "target_std": float(df[D.TARGET].std()),
        "rating_vs_chronology_corr": float(
            np.corrcoef(df["number_in_series"], df[D.TARGET])[0, 1]
        ),
        "golden_era_mean": float(df[df.season <= 10][D.TARGET].mean()),
        "later_era_mean": float(df[df.season > 10][D.TARGET].mean()),
        "cv_rmse": {r.name: round(r.rmse_mean, 3) for r in results},
        "cv_rmse_std": {r.name: round(r.rmse_std, 3) for r in results},
        "out_of_fold_rmse": round(oof_rmse, 3),
        "permutation_importance": {
            row.feature: round(float(row.rmse_increase), 4) for row in imp.itertuples()
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {METRICS_PATH.relative_to(ROOT)} and {len(list(FIG_DIR.glob('*.png')))} figures.")


if __name__ == "__main__":
    main()
