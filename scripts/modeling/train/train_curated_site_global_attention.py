"""Curated mixed-granularity attention experiment.

The molecule is represented by variable-length per-site GIN tokens (300 dims),
while the receptor is represented by one mean ESM vector (1280 dims).  The
attention model lets the receptor token query all molecule-site tokens.  Two
controls use exactly the same splits: XGBoost on mean-pooled GIN + ESM, and a
small DeepSets-style MLP using masked mean/max pooling instead of attention.

Run one GPU job, or use ``run_curated_site_global_attention.sh`` for the
resumable multi-seed grid.  A completed (regime, model, seed) is skipped unless
``--force`` is supplied.  Test predictions are written per run so the plotting
notebook can build confidence intervals and ROC/PR curves without retraining.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from orbind import dataset as D

REGIMES = ("transductive", "inductive_molecule")
MODELS = ("site_global_attention", "site_pool_mlp", "xgb_mean")
METRICS = ("AUROC", "AUPRC", "MCC", "F1", "precision", "recall")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(pairs: pd.DataFrame, regime: str, seed: int,
                  test_size: float, val_size: float):
    """Return train/validation/test row indices for one independent seed."""
    y = pairs.label.to_numpy(dtype=np.float32)
    if regime == "transductive":
        trval, te = D.split(pairs, y, kind="stratified", test_size=test_size, seed=seed)
        tv = pairs.iloc[np.flatnonzero(trval)].reset_index(drop=True)
        tr_local, va_local = D.split(
            tv, y[trval], kind="stratified",
            test_size=val_size / (1.0 - test_size), seed=seed + 10000)
        tv_idx = np.flatnonzero(trval)
        return tv_idx[tr_local], tv_idx[va_local], np.flatnonzero(te)
    if regime == "inductive_molecule":
        trval, te = D.split(pairs, y, kind="group_molecule", test_size=test_size, seed=seed)
        tv_idx = np.flatnonzero(trval)
        tv = pairs.iloc[tv_idx].reset_index(drop=True)
        tr_local, va_local = D.split(
            tv, y[trval], kind="group_molecule",
            test_size=val_size / (1.0 - test_size), seed=seed + 10000)
        return tv_idx[tr_local], tv_idx[va_local], np.flatnonzero(te)
    raise ValueError(regime)


class PairDataset(Dataset):
    def __init__(self, pairs, indices, proteins, molecule_sites):
        self.pairs, self.indices = pairs, np.asarray(indices)
        self.proteins, self.molecule_sites = proteins, molecule_sites

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.pairs.iloc[self.indices[i]]
        return (np.asarray(self.proteins[row.receptor], dtype=np.float32),
                np.asarray(self.molecule_sites[row.inchikey], dtype=np.float32),
                np.float32(row.label))


def collate(batch):
    proteins, sites, labels = zip(*batch)
    lengths = [len(x) for x in sites]
    x = torch.zeros(len(sites), max(lengths), sites[0].shape[-1])
    mask = torch.ones(len(sites), max(lengths), dtype=torch.bool)
    for i, site in enumerate(sites):
        x[i, :len(site)] = torch.from_numpy(site)
        mask[i, :len(site)] = False
    return torch.from_numpy(np.stack(proteins)), x, torch.tensor(labels), mask


def masked_mean_max(x, mask):
    keep = (~mask).unsqueeze(-1)
    denom = keep.sum(1).clamp_min(1).to(x.dtype)
    mean = (x * keep).sum(1) / denom
    maxv = x.masked_fill(mask.unsqueeze(-1), -torch.inf).max(1).values
    return mean, maxv


class MixedAttention(nn.Module):
    def __init__(self, dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.protein = nn.Sequential(nn.Linear(1280, dim), nn.LayerNorm(dim), nn.GELU())
        self.site = nn.Sequential(nn.Linear(300, dim), nn.LayerNorm(dim), nn.GELU())
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * dim, 1))

    def forward(self, protein, sites, mask):
        q = self.protein(protein).unsqueeze(1)
        s = self.site(sites)
        attended, _ = self.attn(q, s, s, key_padding_mask=mask, need_weights=False)
        q = self.norm(q + attended).squeeze(1)
        mean, maxv = masked_mean_max(s, mask)
        return self.head(torch.cat([q, mean, maxv], dim=-1)).squeeze(-1)


class PoolMLP(nn.Module):
    def __init__(self, dim=64, dropout=0.1):
        super().__init__()
        self.protein = nn.Sequential(nn.Linear(1280, dim), nn.LayerNorm(dim), nn.GELU())
        self.site = nn.Sequential(nn.Linear(300, dim), nn.LayerNorm(dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * dim, 1))

    def forward(self, protein, sites, mask):
        mean, maxv = masked_mean_max(self.site(sites), mask)
        return self.head(torch.cat([self.protein(protein), mean, maxv], dim=-1)).squeeze(-1)


def metric_at(y, p, threshold=0.5):
    m = D.metrics(y, p)
    pred = (p >= threshold).astype(np.int8)
    from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
    m.update(MCC=matthews_corrcoef(y, pred), F1=f1_score(y, pred, zero_division=0),
             precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred, zero_division=0))
    return m


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); ys, ps = [], []
    for protein, sites, y, mask in loader:
        z = model(protein.to(device), sites.to(device), mask.to(device))
        ys.append(y.numpy()); ps.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_torch(model, train_loader, val_loader, test_loader, args, device):
    model.to(device)
    pos = sum(float(y.sum()) for _, _, y, _ in train_loader)
    n = len(train_loader.dataset)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(n - pos) / max(pos, 1)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_ap, best_state, best_epoch, stale = -np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for protein, sites, y, mask in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(protein.to(device), sites.to(device), mask.to(device)), y.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad); opt.step()
        yv, pv = predict(model, val_loader, device)
        ap = D.metrics(yv, pv)["AUPRC"]
        print(f"    epoch {epoch:03d}: val_AUPRC={ap:.4f}", flush=True)
        if ap > best_ap + args.min_delta:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state)
    yt, pt = predict(model, test_loader, device)
    return yt, pt, best_epoch, model


def run_xgb(X, y, tr, va, te, args):
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=args.xgb_estimators, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
                          eval_metric="logloss", tree_method="hist", random_state=args.seed,
                          n_jobs=args.xgb_jobs)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    return y[te], model.predict_proba(X[te])[:, 1], 0, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["all", *REGIMES], default="all")
    ap.add_argument("--model", choices=["all", *MODELS], default="all")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    ap.add_argument("--protein", default="data/embeddings/proteins/esm2_650m_mean.npz")
    ap.add_argument("--molecule-sites", default="data/embeddings/molecules/gin_supervised_contextpred_all_m2or_per_atom.npz")
    ap.add_argument("--molecule-mean", default="data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz")
    ap.add_argument("--out-dir", default="results/attention/curated/site_global")
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=1.0); ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--test-size", type=float, default=0.2); ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--device", default="auto"); ap.add_argument("--xgb-estimators", type=int, default=500)
    ap.add_argument("--xgb-jobs", type=int, default=0); ap.add_argument("--force", action="store_true")
    args = ap.parse_args(); seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(ROOT / args.pairs)
    proteins, sites, means = D.load_npz_dict(ROOT / args.protein), D.load_npz_dict(ROOT / args.molecule_sites), D.load_npz_dict(ROOT / args.molecule_mean)
    mask = pairs.receptor.isin(proteins) & pairs.inchikey.isin(sites) & pairs.inchikey.isin(means)
    pairs = pairs.loc[mask].reset_index(drop=True)
    y = pairs.label.to_numpy(np.float32)
    model_list = list(MODELS) if args.model == "all" else [args.model]
    regime_list = list(REGIMES) if args.regime == "all" else [args.regime]
    records = []
    metrics_path = out / "metrics.csv"
    if metrics_path.exists(): records = pd.read_csv(metrics_path).to_dict("records")
    for regime in regime_list:
        tr, va, te = split_indices(pairs, regime, args.seed, args.test_size, args.val_size)
        loader = lambda idx, shuffle=False: DataLoader(PairDataset(pairs, idx, proteins, sites), batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate, pin_memory=device.type == "cuda")
        for model_name in model_list:
            key = (regime, model_name, args.seed)
            if any((r["regime"], r["model"], int(r["seed"])) == key for r in records) and not args.force:
                print(f"SKIP cached {key}"); continue
            start = time.time(); print(f"RUN {regime} {model_name} seed={args.seed} device={device}")
            if model_name == "xgb_mean":
                X = np.stack([np.concatenate([means[r.inchikey], proteins[r.receptor]]) for _, r in pairs.iterrows()]).astype(np.float32)
                yt, pt, best_epoch, fitted = run_xgb(X, y, tr, va, te, args)
            else:
                fitted = MixedAttention(args.dim, args.heads, args.dropout) if model_name == "site_global_attention" else PoolMLP(args.dim, args.dropout)
                yt, pt, best_epoch, fitted = train_torch(fitted, loader(tr, True), loader(va), loader(te), args, device)
            rec = {"regime": regime, "model": model_name, "seed": args.seed, **metric_at(yt, pt), "best_epoch": best_epoch, "seconds": round(time.time()-start, 1), "n_test": len(te)}
            records = [r for r in records if (r["regime"], r["model"], int(r["seed"])) != key] + [rec]
            pd.DataFrame(records).sort_values(["regime", "model", "seed"]).to_csv(metrics_path, index=False)
            np.savez_compressed(out / f"pred_{regime}_{model_name}_seed{args.seed}.npz", y=yt, p=pt)
            if model_name != "xgb_mean": torch.save({"state_dict": fitted.state_dict(), "config": vars(args), "regime": regime, "model": model_name, "seed": args.seed}, out / f"model_{regime}_{model_name}_seed{args.seed}.pt")
            else: fitted.save_model(out / f"model_{regime}_{model_name}_seed{args.seed}.json")
            print("  " + " ".join(f"{k}={rec[k]:.3f}" for k in ("AUROC", "AUPRC", "MCC", "F1")), flush=True)


if __name__ == "__main__": main()
