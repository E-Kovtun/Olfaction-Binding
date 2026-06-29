"""Train four attention interaction baselines on curated M2OR.

Each completed setup/split is written immediately to
``results/tables/attention_results.{csv,md}``, so baseline_screening.ipynb can
display partial results while the remaining runs are still training.

Examples
--------
Run the full curated grid (2 splits x 4 setups)::

    uv run python scripts/modeling/train/train_attention.py

Run one setup, or make a cheap smoke run::

    uv run python scripts/modeling/train/train_attention.py --setup site_cross
    uv run python scripts/modeling/train/train_attention.py --setup flat_cross \
        --split stratified --epochs 1 --limit-pairs 256 --force
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
from torch.utils.data import DataLoader, Dataset

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind import dataset as D
from orbind.attention import InteractionAttention, SETUPS

SPLITS = ("stratified", "group_molecule")
METRICS = ("AUROC", "AUPRC", "MCC", "F1", "precision", "recall")
DISPLAY = {
    "flat_cross": "Attn·flat-cross",
    "flat_self": "Attn·flat-self",
    "site_cross": "Attn·site-cross",
    "site_self": "Attn·site-self",
}


class PairDataset(Dataset):
    def __init__(self, pairs, indices, protein, molecule, site):
        self.pairs = pairs
        self.indices = np.asarray(indices)
        self.protein = protein
        self.molecule = molecule
        self.site = site

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.pairs.iloc[self.indices[i]]
        p = np.asarray(self.protein[row.receptor], dtype=np.float32)
        m = np.asarray(self.molecule[row.inchikey], dtype=np.float32)
        return p, m, np.float32(row.label)


def _pad(arrays):
    lengths = [len(x) for x in arrays]
    out = torch.zeros(len(arrays), max(lengths), arrays[0].shape[-1], dtype=torch.float32)
    mask = torch.ones(len(arrays), max(lengths), dtype=torch.bool)
    for i, x in enumerate(arrays):
        n = len(x)
        out[i, :n] = torch.from_numpy(x)
        mask[i, :n] = False
    return out, mask


def collate_flat(batch):
    p, m, y = zip(*batch)
    return (torch.from_numpy(np.stack(p)), torch.from_numpy(np.stack(m)),
            torch.tensor(y), None, None)


def collate_site(batch):
    p, m, y = zip(*batch)
    pt, pmask = _pad(p)
    mt, mmask = _pad(m)
    return pt, mt, torch.tensor(y), pmask, mmask


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def display_path(path):
    """Prefer a short repo-relative path, but allow temp/custom output paths."""
    return path.relative_to(_root) if path.is_relative_to(_root) else path


def make_indices(pairs, split, test_size, val_size, seed):
    y = pairs.label.to_numpy(dtype=np.float32)
    trainval, test = D.split(pairs, y, kind=split, test_size=test_size, seed=seed)
    tv_idx = np.flatnonzero(trainval)
    tv_pairs = pairs.iloc[tv_idx].reset_index(drop=True)
    tv_y = y[tv_idx]
    relative_val = val_size / (1.0 - test_size)
    train_local, val_local = D.split(
        tv_pairs, tv_y, kind=split, test_size=relative_val, seed=seed + 1)
    return tv_idx[train_local], tv_idx[val_local], np.flatnonzero(test)


def _move(batch, device):
    p, m, y, pmask, mmask = batch
    return (p.to(device), m.to(device), y.to(device),
            None if pmask is None else pmask.to(device),
            None if mmask is None else mmask.to(device))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    ys, scores = [], []
    for batch in loader:
        p, m, y, pmask, mmask = _move(batch, device)
        scores.append(torch.sigmoid(model(p, m, pmask, mmask)).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores)


def train_one(setup, split, pairs, protein, molecule, args, device):
    site = setup.startswith("site")
    tr, va, te = make_indices(pairs, split, args.test_size, args.val_size, args.seed)
    collate = collate_site if site else collate_flat
    batch_size = args.batch_size or ({
        "flat_cross": 4, "flat_self": 2, "site_cross": 16, "site_self": 8,
    }[setup])

    def loader(indices, shuffle=False):
        return DataLoader(PairDataset(pairs, indices, protein, molecule, site),
                          batch_size=batch_size, shuffle=shuffle, num_workers=0,
                          collate_fn=collate, pin_memory=device.type == "cuda")

    train_loader, val_loader, test_loader = loader(tr, True), loader(va), loader(te)
    model = InteractionAttention(setup, dim=args.dim, heads=args.heads,
                                 layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = float(pairs.iloc[tr].label.sum())
    pos_weight = torch.tensor([(len(tr) - pos) / max(pos, 1.0)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_ap, best_epoch, best_state, stale = -np.inf, 0, None, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            p, m, y, pmask, mmask = _move(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(p, m, pmask, mmask)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            total_loss += float(loss) * len(y)

        yv, pv = predict(model, val_loader, device)
        vm = D.metrics(yv, pv)
        print(f"  epoch {epoch:02d} loss={total_loss/len(tr):.4f} "
              f"val_AUROC={vm['AUROC']:.3f} val_AUPRC={vm['AUPRC']:.3f}", flush=True)
        if vm["AUPRC"] > best_ap + args.min_delta:
            best_ap, best_epoch = vm["AUPRC"], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop (best epoch {best_epoch})", flush=True)
                break

    model.load_state_dict(best_state)
    yt, pt = predict(model, test_loader, device)
    metrics = D.metrics(yt, pt)
    elapsed = time.time() - started
    checkpoint = {
        "setup": setup, "split": split, "seed": args.seed,
        "model_args": {"dim": args.dim, "heads": args.heads, "layers": args.layers,
                       "dropout": args.dropout},
        "state_dict": best_state, "best_epoch": best_epoch,
        "test_indices": te, "test_labels": yt, "test_scores": pt,
    }
    return metrics, checkpoint, {
        "train": len(tr), "val": len(va), "test": len(te),
        "test_pos": int(yt.sum()), "best_epoch": best_epoch,
        "seconds": round(elapsed, 1), "batch_size": batch_size,
    }


def write_results(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        table = pd.read_csv(path)
        same = ((table.setup == record["setup"]) & (table.split == record["split"]) &
                (table.seed == record["seed"]))
        table = table.loc[~same]
        table = pd.concat([table, pd.DataFrame([record])], ignore_index=True)
    else:
        table = pd.DataFrame([record])
    order = {name: i for i, name in enumerate(SETUPS)}
    split_order = {name: i for i, name in enumerate(SPLITS)}
    table["_setup"] = table.setup.map(order)
    table["_split"] = table.split.map(split_order)
    table = table.sort_values(["_setup", "_split", "seed"]).drop(columns=["_setup", "_split"])

    tmp = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)  # atomic: notebook sees either the old or the complete new CSV

    cols = ["method", "split", *METRICS, "best_epoch", "seconds"]
    lines = ["# Curated attention results", "",
             "| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for _, row in table.iterrows():
        values = []
        for c in cols:
            v = row[c]
            values.append(f"{v:.3f}" if c in METRICS else str(v))
        lines.append("| " + " | ".join(values) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _limit_pairs(pairs, n, seed):
    if not n or n >= len(pairs):
        return pairs
    # Debug-only subsample retaining positives and negatives.
    frac = n / len(pairs)
    out = (pairs.groupby("label", group_keys=False)
           .sample(frac=frac, random_state=seed))
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["all", *SETUPS], default="all")
    ap.add_argument("--split", choices=["all", *SPLITS], default="all")
    ap.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    ap.add_argument("--protein-mean", default="data/embeddings/proteins/esm2_650m_mean_curated.npz")
    ap.add_argument("--protein-sites", default="data/embeddings/proteins/esm2_650m_per_residue_curated.npz")
    ap.add_argument("--molecule-mean", default="data/embeddings/molecules/gin_supervised_contextpred.npz")
    ap.add_argument("--molecule-sites", default="data/embeddings/molecules/gin_supervised_contextpred_per_atom.npz")
    ap.add_argument("--out", default=None,
                    help="CSV path (default: production table; a separate smoke table with --limit-pairs)")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    ap.add_argument("--limit-pairs", type=int, default=None, help="Debug-only pair subsample")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}", flush=True)

    pairs = _limit_pairs(pd.read_csv(_root / args.pairs), args.limit_pairs, args.seed)
    setup_list = list(SETUPS) if args.setup == "all" else [args.setup]
    split_list = list(SPLITS) if args.split == "all" else [args.split]
    default_out = ("results/tables/attention_smoke_results.csv" if args.limit_pairs
                   else "results/tables/attention_results.csv")
    default_ckpt = ("results/checkpoints/attention_smoke" if args.limit_pairs
                    else "results/checkpoints")
    out_path = _root / (args.out or default_out)
    ckpt_dir = _root / (args.checkpoint_dir or default_ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    caches = {}
    def load(path):
        if path not in caches:
            print(f"loading {path} ...", flush=True)
            caches[path] = D.load_npz_dict(_root / path)
        return caches[path]

    for setup in setup_list:
        site = setup.startswith("site")
        p_path = args.protein_sites if site else args.protein_mean
        m_path = args.molecule_sites if site else args.molecule_mean
        if site and not (_root / m_path).exists():
            raise FileNotFoundError(
                f"missing {m_path}; generate it with:\n"
                "uv run python scripts/embedding_generation/molecules/embed_molecules_gin.py "
                "--molecules data/processed/molecules/molecule_smiles.csv "
                "--node-out data/embeddings/molecules/gin_supervised_contextpred_per_atom.npz")
        protein, molecule = load(p_path), load(m_path)
        mask = pairs.receptor.isin(protein) & pairs.inchikey.isin(molecule)
        p = pairs.loc[mask].reset_index(drop=True)
        if (~mask).sum():
            print(f"  dropped {(~mask).sum()} pairs lacking embeddings")

        for split in split_list:
            already = (not existing.empty and
                       ((existing.setup == setup) & (existing.split == split) &
                        (existing.seed == args.seed)).any())
            if already and not args.force and args.limit_pairs is None:
                print(f"[{setup} | {split}] cached -> skip", flush=True)
                continue
            print(f"\n[{setup} | {split}] pairs={len(p)}", flush=True)
            metrics, checkpoint, info = train_one(
                setup, split, p, protein, molecule, args, device)
            ckpt_path = ckpt_dir / f"attention_{setup}_{split}.pt"
            torch.save(checkpoint, ckpt_path)
            record = {
                "dataset": "curated", "setup": setup, "method": DISPLAY[setup],
                "split": split, "seed": args.seed,
                **{k: round(float(v), 4) for k, v in metrics.items()}, **info,
            }
            write_results(out_path, record)
            existing = pd.read_csv(out_path)
            print("  " + " ".join(f"{k}={metrics[k]:.3f}" for k in METRICS))
            print(f"  saved -> {display_path(out_path)}; checkpoint -> {display_path(ckpt_path)}",
                  flush=True)


if __name__ == "__main__":
    main()
