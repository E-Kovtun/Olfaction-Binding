"""Reusable MP baselines (MLP + XGBoost) over in-memory embedding dicts.

Shared by scripts/modeling/eval/eval_mp_table.py and notebooks/. Works on dicts so a
notebook can pass *modified* protein embeddings (e.g. transformed ESM-2) without
touching files.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from . import dataset as D


def make_xy(pairs, prot, mol, random_prot=False, seed=0):
    """X = [molecule || protein] for the pairs that have both embeddings."""
    mask = pairs["receptor"].isin(prot) & pairs["inchikey"].isin(mol)
    p = pairs[mask].reset_index(drop=True)
    Xm = np.stack([mol[i] for i in p["inchikey"]]).astype(np.float32)
    if random_prot:
        dim = next(iter(prot.values())).shape[0]
        rng = np.random.default_rng(seed)
        Xp = rng.standard_normal((len(p), dim)).astype(np.float32)  # per-row noise floor
    else:
        Xp = np.stack([prot[r] for r in p["receptor"]]).astype(np.float32)
    X = np.concatenate([Xm, Xp], axis=1)
    y = p["label"].to_numpy().astype(np.float32)
    return X, y, p


def train_mlp(Xtr, ytr, Xte, seed=42, hidden=(512, 128), dropout=0.3, lr=1e-3, epochs=100, batch=256):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr, ytr, Xte = torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)])
    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d = h
    layers += [nn.Linear(d, 1)]
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]; opt.zero_grad()
            loss_fn(model(Xtr[b]).squeeze(-1), ytr[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte).squeeze(-1)).numpy()


def train_boost(Xtr, ytr, Xte, seed=42):
    import xgboost as xgb
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            eval_metric="aucpr", tree_method="hist", device=device,
                            n_jobs=-1, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


HEADS = {"mlp": train_mlp, "boost": train_boost}


def run_one(pairs, prot, mol, head, split, random_prot=False, seed=42, test_size=0.2):
    """assemble -> split -> train head -> metrics. Returns (metrics, info)."""
    X, y, p = make_xy(pairs, prot, mol, random_prot=random_prot, seed=seed)
    tr, te = D.split(p, y, kind=split, test_size=test_size, seed=seed)
    pred = HEADS[head](X[tr], y[tr], X[te], seed=seed)
    info = {"train": int(tr.sum()), "test": int(te.sum()), "test_pos": int(y[te].sum())}
    return D.metrics(y[te], pred), info
