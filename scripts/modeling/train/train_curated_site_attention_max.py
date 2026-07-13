"""Curated attention-as-score experiment.

This is a constrained variant of the mixed-granularity attention screen.  A mean
ESM protein vector queries per-site GIN molecule tokens, the normalized attention
weights are computed explicitly, and the final prediction is the maximum
attention weight.  Training combines BCE on that max weight with a margin loss:
negative pairs should have all weights below ``neg_threshold`` and positive pairs
should have at least one weight above ``pos_threshold``.
"""
from __future__ import annotations

import argparse
import copy
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
MODEL_NAME = "site_attention_max"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(pairs: pd.DataFrame, regime: str, seed: int, test_size: float, val_size: float):
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


class MaxAttentionScore(nn.Module):
    def __init__(self, dim=64, temperature=1.0, dropout=0.1):
        super().__init__()
        self.protein = nn.Sequential(nn.Linear(1280, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.site = nn.Sequential(nn.Linear(300, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.temperature = temperature
        self.scale = dim ** -0.5

    def attention_weights(self, protein, sites, mask):
        q = self.protein(protein)
        k = self.site(sites)
        logits = torch.einsum("bd,bld->bl", q, k) * self.scale / self.temperature
        logits = logits.masked_fill(mask, -torch.inf)
        return torch.softmax(logits, dim=1)

    def forward(self, protein, sites, mask):
        weights = self.attention_weights(protein, sites, mask)
        return weights.max(dim=1).values, weights


def attention_loss(p, y, args):
    p_safe = p.clamp(1e-6, 1.0 - 1e-6)
    pos = y.sum()
    neg = y.numel() - pos
    pos_weight = (neg / pos.clamp_min(1.0)).detach()
    bce = nn.functional.binary_cross_entropy(p_safe, y, weight=torch.where(y > 0, pos_weight, torch.ones_like(y)))
    neg_margin = ((1.0 - y) * torch.relu(p - args.neg_threshold)).mean()
    pos_margin = (y * torch.relu(args.pos_threshold - p)).mean()
    return bce + args.margin_weight * (neg_margin + pos_margin)


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
    model.eval()
    ys, ps, entropies = [], [], []
    for protein, sites, y, mask in loader:
        p, w = model(protein.to(device), sites.to(device), mask.to(device))
        valid = (~mask.to(device)).float()
        entropy = -(w.clamp_min(1e-12).log() * w * valid).sum(1) / valid.sum(1).log().clamp_min(1e-6)
        ys.append(y.numpy())
        ps.append(p.cpu().numpy())
        entropies.append(entropy.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(entropies)


def train_model(model, train_loader, val_loader, test_loader, args, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_ap, best_state, best_epoch, stale = -np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for protein, sites, y, mask in train_loader:
            opt.zero_grad(set_to_none=True)
            p, _ = model(protein.to(device), sites.to(device), mask.to(device))
            loss = attention_loss(p, y.to(device), args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        yv, pv, _ = predict(model, val_loader, device)
        ap = D.metrics(yv, pv)["AUPRC"]
        print(f"    epoch {epoch:03d}: loss={np.mean(losses):.4f} val_AUPRC={ap:.4f}", flush=True)
        if ap > best_ap + args.min_delta:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    yt, pt, ent = predict(model, test_loader, device)
    return yt, pt, ent, best_epoch, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["all", *REGIMES], default="all")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    ap.add_argument("--protein", default="data/embeddings/proteins/esm2_650m_mean_full.npz")
    ap.add_argument("--molecule-sites", default="data/embeddings/molecules/gin_supervised_contextpred_per_atom.npz")
    ap.add_argument("--out-dir", default="results/attention/curated/site_max")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--neg-threshold", type=float, default=0.15)
    ap.add_argument("--pos-threshold", type=float, default=0.25)
    ap.add_argument("--margin-weight", type=float, default=1.0)
    ap.add_argument("--decision-threshold", type=float, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.neg_threshold >= args.pos_threshold:
        raise ValueError("neg-threshold must be lower than pos-threshold")
    if args.decision_threshold is None:
        args.decision_threshold = args.pos_threshold
    seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(ROOT / args.pairs)
    proteins = D.load_npz_dict(ROOT / args.protein)
    sites = D.load_npz_dict(ROOT / args.molecule_sites)
    mask = pairs.receptor.isin(proteins) & pairs.inchikey.isin(sites)
    pairs = pairs.loc[mask].reset_index(drop=True)

    regime_list = list(REGIMES) if args.regime == "all" else [args.regime]
    records = []
    metrics_path = out / "metrics.csv"
    if metrics_path.exists():
        records = pd.read_csv(metrics_path).to_dict("records")

    for regime in regime_list:
        key = (regime, MODEL_NAME, args.seed, args.neg_threshold, args.pos_threshold)
        if any((r["regime"], r["model"], int(r["seed"]), float(r["neg_threshold"]), float(r["pos_threshold"])) == key for r in records) and not args.force:
            print(f"SKIP cached {key}")
            continue
        start = time.time()
        print(f"RUN {regime} {MODEL_NAME} seed={args.seed} device={device} neg_t={args.neg_threshold} pos_t={args.pos_threshold}", flush=True)
        tr, va, te = split_indices(pairs, regime, args.seed, args.test_size, args.val_size)
        loader = lambda idx, shuffle=False: DataLoader(
            PairDataset(pairs, idx, proteins, sites), batch_size=args.batch_size, shuffle=shuffle,
            collate_fn=collate, pin_memory=device.type == "cuda")
        model = MaxAttentionScore(args.dim, args.temperature, args.dropout)
        yt, pt, entropy, best_epoch, fitted = train_model(model, loader(tr, True), loader(va), loader(te), args, device)
        rec = {"regime": regime, "model": MODEL_NAME, "seed": args.seed,
               "neg_threshold": args.neg_threshold, "pos_threshold": args.pos_threshold,
               "margin_weight": args.margin_weight, "decision_threshold": args.decision_threshold,
               **metric_at(yt, pt, args.decision_threshold),
               "mean_max_weight": float(np.mean(pt)), "pos_mean_max_weight": float(np.mean(pt[yt == 1])) if np.any(yt == 1) else np.nan,
               "neg_mean_max_weight": float(np.mean(pt[yt == 0])) if np.any(yt == 0) else np.nan,
               "mean_attention_entropy": float(np.mean(entropy)),
               "best_epoch": best_epoch, "seconds": round(time.time() - start, 1), "n_test": len(te)}
        records = [r for r in records if not ((r["regime"], r["model"], int(r["seed"]), float(r["neg_threshold"]), float(r["pos_threshold"])) == key)] + [rec]
        pd.DataFrame(records).sort_values(["regime", "model", "seed"]).to_csv(metrics_path, index=False)
        suffix = f"{regime}_{MODEL_NAME}_seed{args.seed}_neg{args.neg_threshold:g}_pos{args.pos_threshold:g}"
        np.savez_compressed(out / f"pred_{suffix}.npz", y=yt, p=pt, attention_entropy=entropy)
        torch.save({"state_dict": fitted.state_dict(), "config": vars(args), "regime": regime,
                    "model": MODEL_NAME, "seed": args.seed}, out / f"model_{suffix}.pt")
        print("  " + " ".join(f"{k}={rec[k]:.3f}" for k in ("AUROC", "AUPRC", "MCC", "F1")), flush=True)


if __name__ == "__main__":
    main()