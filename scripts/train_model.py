"""Train the rating model on all rated episodes and persist it.

    python scripts/train_model.py

Writes models/rating_model.joblib. The web app loads this artifact; the model
is small and trains in seconds, so the artifact is a convenience, not a
checked-in binary blob that hides how it was made.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simpsons import data as D  # noqa: E402
from simpsons import modeling as M  # noqa: E402

MODEL_PATH = ROOT / "models" / "rating_model.joblib"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = D.build_dataset()
    X, y = D.xy(df, D.ALL_FEATURES)
    model = M.make_model().fit(X, y)
    joblib.dump({"model": model, "features": D.ALL_FEATURES}, MODEL_PATH)
    print(f"Trained on {len(df)} episodes -> {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
