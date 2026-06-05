"""Deep-learning autoresearch, Karpathy-style.

Follows "A Recipe for Training Neural Networks":
  1. Baselines are already known (mean 0.725; gradient boosting ~0.43-0.44).
  2. SANITY: overfit a tiny batch first — if the net can't drive train RMSE to
     ~0 on 48 samples, the training loop is broken; stop and fix it.
  3. REGULARIZE: standardized inputs, dropout, weight decay, early stopping on a
     held-out val split inside every fold.
  4. TUNE: random search over depth / width / dropout / weight-decay / lr, scored
     by the same honest 5-fold CV used everywhere else in this repo.

Small-data reality check: with ~300-600 rows this is the regime where Karpathy's
"don't be a hero" applies. We run it anyway and let the CV decide.

    python scripts/autoresearch_dl.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simpsons import data as D  # noqa: E402
from simpsons import llm_features as L  # noqa: E402
from simpsons import text_features as T  # noqa: E402

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)
DEVICE = "cpu"


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train(model, Xtr, ytr, Xval, yval, lr, weight_decay, batch_size, max_epochs=300, patience=25):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.MSELoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xval_t, yval_t = torch.tensor(Xval), torch.tensor(yval)
    n = len(Xtr_t)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xval_t), yval_t).item()
        if v < best_val - 1e-5:
            best_val, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def _predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def sanity_overfit(X, y) -> float:
    """Karpathy step 3: can the net memorize a tiny batch? (train RMSE should -> ~0)"""
    torch.manual_seed(SEED)
    Xs = StandardScaler().fit_transform(X[:48]).astype(np.float32)
    ys = y[:48].astype(np.float32)
    model = MLP(Xs.shape[1], (128, 64), dropout=0.0).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    lossf = nn.MSELoss()
    Xt, yt = torch.tensor(Xs), torch.tensor(ys)
    for _ in range(1500):
        model.train(); opt.zero_grad()
        lossf(model(Xt), yt).backward(); opt.step()
    return float(np.sqrt(np.mean((_predict(model, Xs) - ys) ** 2)))


def cv_rmse(X, y, hidden, dropout, weight_decay, lr, batch_size) -> float:
    rmses = []
    for tr, te in CV.split(X):
        torch.manual_seed(SEED)
        scaler = StandardScaler().fit(X[tr])
        Xtr_all = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)
        ytr_all = y[tr].astype(np.float32)
        Xtr, Xval, ytr, yval = train_test_split(Xtr_all, ytr_all, test_size=0.2, random_state=SEED)
        model = MLP(X.shape[1], hidden, dropout).to(DEVICE)
        _train(model, Xtr, ytr, Xval, yval, lr, weight_decay, batch_size)
        pred = _predict(model, Xte)
        rmses.append(np.sqrt(np.mean((pred - y[te].astype(np.float32)) ** 2)))
    return float(np.mean(rmses))


SEARCH = {
    "hidden": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
    "dropout": [0.0, 0.1, 0.25, 0.5],
    "weight_decay": [1e-5, 1e-4, 1e-3, 1e-2],
    "lr": [3e-4, 1e-3, 3e-3],
    "batch_size": [16, 32],
}


def random_search(X, y, n_iter=16):
    rng = np.random.default_rng(SEED)
    best = None
    for _ in range(n_iter):
        cfg = {k: v[rng.integers(len(v))] for k, v in SEARCH.items()}
        r = cv_rmse(X, y, **cfg)
        if best is None or r < best[0]:
            best = (r, cfg)
        print(f"    hidden={str(cfg['hidden']):14s} drop={cfg['dropout']:.2f} "
              f"wd={cfg['weight_decay']:.0e} lr={cfg['lr']:.0e} bs={cfg['batch_size']:>2}  RMSE {r:.3f}")
    return best


def run(name, X, y):
    X = X.astype(np.float32)
    print(f"\n=== {name}: {X.shape[0]} episodes, {X.shape[1]} features ===")
    print(f"  sanity (overfit 48 samples) train RMSE: {sanity_overfit(X, y):.3f}  (want ~0)")
    best_rmse, best_cfg = random_search(X, y)
    print(f"  >> best DL CV RMSE: {best_rmse:.3f}  with {best_cfg}")
    return best_rmse


def main():
    df = T.build_text_dataset(n_components=100)
    y_full = df[D.TARGET].to_numpy()

    # Full data (597): the strongest non-LLM feature set.
    cols = D.SCRIPT_FEATURES + df.attrs["lsa_cols"] + D.TIME_FEATURES
    run("all_text+chrono (full 597)", df[cols].to_numpy(), y_full)

    # 282-episode subset with the blind LLM rubric features.
    llm = L.load_llm_features()
    sub = df.set_index("id").join(llm, how="inner").dropna(subset=L.FEATURE_COLUMNS)
    yl = sub[D.TARGET].to_numpy()
    run("llm+chrono (282)", sub[L.FEATURE_COLUMNS + D.TIME_FEATURES].to_numpy(), yl)

    print("\nReference (gradient boosting, same harness): chrono 0.438 · best 0.430 · baseline 0.725")


if __name__ == "__main__":
    main()
