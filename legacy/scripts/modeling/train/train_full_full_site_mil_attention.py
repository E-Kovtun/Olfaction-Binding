"""Full_full LoRaX mixed-granularity MIL/attention screen.

The molecule is represented by per-atom/site GIN tokens keyed by InChIKey.
The receptor is represented by one mean LoRaX ESM-1b vector keyed by sequence.

Splits:
  * transductive: genuine LoRaX rand_split_1..5;
  * inductive_molecule: fold-1 full pool with cold-molecule split seeds.

The script writes one atomic run directory.  Use
``legacy/scripts/queues/run_full_full_site_mil_attention.sh`` for the resumable
multi-run dashboard.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from orbind import dataset as D
from orbind.baselines import train_boost

REGIMES = ("transductive", "inductive_molecule")
MODELS = (
    "boost_gin_mean",
    "mil_max",
    "mil_noisy_or",
    "mil_lse",
    "attention_mil",
    "attention_excess_max",
)
METRICS = ("AUROC", "AUPRC", "MCC", "F1", "precision", "recall")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz_dict(path: pathlib.Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    if set(z.files) >= {"ids", "emb"} and len(z.files) <= 3:
        return {str(k): np.asarray(v) for k, v in zip(z["ids"], z["emb"])}
    return {str(k): np.asarray(z[k]) for k in z.files}


def metric_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    m = D.metrics(y, p)
    pred = (p >= threshold).astype(np.int8)
    m.update(
        MCC=matthews_corrcoef(y, pred),
        F1=f1_score(y, pred, zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
    )
    return {k: float(m[k]) for k in METRICS}


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    if len(grid) == 0:
        return 0.5
    vals = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(vals))])


def read_lorax_split(base: pathlib.Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(base / f"{split}_df.csv")
    return df.rename(columns={"SMILES": "smiles", "Protein sequence": "protein", "output": "label"})


@dataclass
class FullFullData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    coverage: dict[str, int | float | str]


def filter_and_annotate(
    df: pd.DataFrame,
    smiles_to_inchikey: dict[str, str],
    mol_keys: set[str],
    protein_keys: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    out = df.copy()
    out["inchikey"] = out["smiles"].astype(str).map(smiles_to_inchikey)
    out["label"] = out["label"].astype(np.float32)
    mask = out["inchikey"].isin(mol_keys) & out["protein"].astype(str).isin(protein_keys)
    out = out.loc[mask].reset_index(drop=True)
    return out, {
        "rows_before": int(before),
        "rows_after": int(len(out)),
        "rows_dropped": int(before - len(out)),
        "n_molecules": int(out["inchikey"].nunique()),
        "n_proteins": int(out["protein"].nunique()),
        "n_pos": int(out["label"].sum()),
        "n_neg": int(len(out) - out["label"].sum()),
    }


def load_full_full_data(args, regime: str, repeat: int, mol_keys: set[str], protein_keys: set[str]) -> FullFullData:
    lorax_root = ROOT / args.lorax_dir
    bridge = pd.read_csv(ROOT / args.lorax_bridge)
    smiles_to_inchikey = dict(zip(bridge["smiles_lorax"].astype(str), bridge["inchikey"].astype(str)))

    if regime == "transductive":
        base = lorax_root / f"rand_split_{repeat}"
        raw = {s: read_lorax_split(base, s) for s in ("train", "val", "test")}
        filtered, stats = {}, {}
        for split, df in raw.items():
            filtered[split], stats[split] = filter_and_annotate(df, smiles_to_inchikey, mol_keys, protein_keys)
        coverage = {
            "regime": regime,
            "repeat": repeat,
            "split_protocol": "lorax_rand_split",
            **{f"{sp}_{k}": v for sp, d in stats.items() for k, v in d.items()},
        }
        return FullFullData(filtered["train"], filtered["val"], filtered["test"], coverage)

    if regime != "inductive_molecule":
        raise ValueError(regime)

    parts = []
    for split in ("train", "val", "test"):
        df = read_lorax_split(lorax_root / "rand_split_1", split)
        df["_source_split"] = split
        parts.append(df)
    pool = pd.concat(parts, ignore_index=True).drop_duplicates(["smiles", "protein", "_DataQuality"]).reset_index(drop=True)
    pool, pool_stats = filter_and_annotate(pool, smiles_to_inchikey, mol_keys, protein_keys)
    ec50 = pool.loc[pool["_DataQuality"].eq("ec50")].reset_index(drop=True)
    mol_tbl = ec50.groupby("smiles", sort=False)["label"].max().reset_index(name="has_pos")
    mols = mol_tbl["smiles"].to_numpy()
    strat = mol_tbl["has_pos"].astype(int).to_numpy()
    try:
        _, held = train_test_split(mols, test_size=args.inductive_holdout_fraction, random_state=repeat, stratify=strat)
        held_tbl = mol_tbl[mol_tbl["smiles"].isin(held)]
        val_mols, test_mols = train_test_split(
            held_tbl["smiles"].to_numpy(),
            test_size=args.inductive_test_fraction_within_holdout,
            random_state=repeat + 17,
            stratify=held_tbl["has_pos"].astype(int).to_numpy(),
        )
    except ValueError:
        _, held = train_test_split(mols, test_size=args.inductive_holdout_fraction, random_state=repeat)
        val_mols, test_mols = train_test_split(
            held, test_size=args.inductive_test_fraction_within_holdout, random_state=repeat + 17
        )
    held_set = set(val_mols) | set(test_mols)
    train = pool.loc[~pool["smiles"].isin(held_set)].reset_index(drop=True)
    val = ec50.loc[ec50["smiles"].isin(val_mols)].reset_index(drop=True)
    test = ec50.loc[ec50["smiles"].isin(test_mols)].reset_index(drop=True)
    coverage = {
        "regime": regime,
        "repeat": repeat,
        "split_protocol": "fold1_cold_molecule_seed",
        **{f"pool_{k}": v for k, v in pool_stats.items()},
        "train_rows_after": int(len(train)),
        "val_rows_after": int(len(val)),
        "test_rows_after": int(len(test)),
        "train_n_pos": int(train["label"].sum()),
        "val_n_pos": int(val["label"].sum()),
        "test_n_pos": int(test["label"].sum()),
        "train_n_molecules": int(train["inchikey"].nunique()),
        "val_n_molecules": int(val["inchikey"].nunique()),
        "test_n_molecules": int(test["inchikey"].nunique()),
        "train_n_proteins": int(train["protein"].nunique()),
        "val_n_proteins": int(val["protein"].nunique()),
        "test_n_proteins": int(test["protein"].nunique()),
    }
    return FullFullData(train, val, test, coverage)


class PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, proteins: dict[str, np.ndarray], sites: dict[str, np.ndarray]):
        self.pairs = pairs.reset_index(drop=True)
        self.proteins = proteins
        self.sites = sites

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        return (
            np.asarray(self.proteins[row.protein], dtype=np.float32),
            np.asarray(self.sites[row.inchikey], dtype=np.float32),
            np.float32(row.label),
        )


def collate(batch):
    proteins, sites, labels = zip(*batch)
    lengths = [len(x) for x in sites]
    site_dim = sites[0].shape[-1]
    x = torch.zeros(len(sites), max(lengths), site_dim, dtype=torch.float32)
    mask = torch.ones(len(sites), max(lengths), dtype=torch.bool)
    for i, site in enumerate(sites):
        x[i, : len(site)] = torch.from_numpy(site)
        mask[i, : len(site)] = False
    return torch.from_numpy(np.stack(proteins)), x, torch.tensor(labels, dtype=torch.float32), mask


def masked_mean_max(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keep = (~mask).unsqueeze(-1)
    denom = keep.sum(1).clamp_min(1).to(x.dtype)
    mean = (x * keep).sum(1) / denom
    maxv = x.masked_fill(mask.unsqueeze(-1), -torch.inf).max(1).values
    return mean, maxv


class SiteScorer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.protein = nn.Sequential(nn.Linear(1280, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.site = nn.Sequential(nn.Linear(300, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, 1))

    def site_logits(self, protein: torch.Tensor, sites: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p = self.protein(protein)
        s = self.site(sites)
        p_rep = p.unsqueeze(1).expand_as(s)
        logits = self.head(torch.cat([s, p_rep, s * p_rep], dim=-1)).squeeze(-1)
        return logits.masked_fill(mask, -torch.inf)


class MILMax(SiteScorer):
    def forward(self, protein, sites, mask):
        return self.site_logits(protein, sites, mask).max(1).values


class MILNoisyOR(SiteScorer):
    def forward(self, protein, sites, mask):
        logits = self.site_logits(protein, sites, mask)
        probs = torch.sigmoid(logits).masked_fill(mask, 0.0)
        log_no = torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(1)
        p = 1.0 - torch.exp(log_no)
        return torch.logit(p.clamp(1e-6, 1.0 - 1e-6))


class MILLogSumExp(SiteScorer):
    def __init__(self, dim: int, dropout: float, temperature: float):
        super().__init__(dim, dropout)
        self.temperature = temperature

    def forward(self, protein, sites, mask):
        logits = self.site_logits(protein, sites, mask)
        n = (~mask).sum(1).clamp_min(1).to(logits.dtype)
        return self.temperature * (torch.logsumexp(logits / self.temperature, dim=1) - n.log())


class AttentionMIL(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.protein = nn.Sequential(nn.Linear(1280, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.site = nn.Sequential(nn.Linear(300, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.attn_v = nn.Sequential(nn.Linear(2 * dim, dim), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.attn_w = nn.Linear(dim, 1)
        self.head = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, 1))

    def forward(self, protein, sites, mask):
        p = self.protein(protein)
        s = self.site(sites)
        p_rep = p.unsqueeze(1).expand_as(s)
        a_in = torch.cat([s, p_rep], dim=-1)
        a = self.attn_w(self.attn_v(a_in) * self.attn_u(a_in)).squeeze(-1).masked_fill(mask, -torch.inf)
        w = torch.softmax(a, dim=1)
        bag = torch.einsum("bl,bld->bd", w, s)
        mean, _ = masked_mean_max(s, mask)
        return self.head(torch.cat([p, bag, mean], dim=-1)).squeeze(-1)


class ExcessAttentionMax(nn.Module):
    def __init__(self, dim: int, dropout: float, temperature: float):
        super().__init__()
        self.protein = nn.Sequential(nn.Linear(1280, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.site = nn.Sequential(nn.Linear(300, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.temperature = temperature
        self.scale = dim ** -0.5

    def weights(self, protein, sites, mask):
        q = self.protein(protein)
        k = self.site(sites)
        logits = torch.einsum("bd,bld->bl", q, k) * self.scale / self.temperature
        logits = logits.masked_fill(mask, -torch.inf)
        return torch.softmax(logits, dim=1)

    def score(self, protein, sites, mask):
        w = self.weights(protein, sites, mask)
        raw_max = w.max(1).values
        n = (~mask).sum(1).to(w.dtype).clamp_min(1.0)
        uniform = 1.0 / n
        return ((raw_max - uniform) / (1.0 - uniform).clamp_min(1e-6)).clamp(0.0, 1.0)

    def forward(self, protein, sites, mask):
        p = self.score(protein, sites, mask)
        return torch.logit(p.clamp(1e-6, 1.0 - 1e-6))


def build_model(name: str, args) -> nn.Module:
    if name == "mil_max":
        return MILMax(args.dim, args.dropout)
    if name == "mil_noisy_or":
        return MILNoisyOR(args.dim, args.dropout)
    if name == "mil_lse":
        return MILLogSumExp(args.dim, args.dropout, args.temperature)
    if name == "attention_mil":
        return AttentionMIL(args.dim, args.dropout)
    if name == "attention_excess_max":
        return ExcessAttentionMax(args.dim, args.dropout, args.temperature)
    raise ValueError(name)


def make_loader(pairs, proteins, sites, args, device, train=False):
    ds = PairDataset(pairs, proteins, sites)
    sampler = None
    shuffle = train
    if train and args.pos_fraction is not None:
        y = pairs["label"].to_numpy(dtype=np.int64)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos and n_neg:
            weights = np.where(y == 1, args.pos_fraction / n_pos, (1.0 - args.pos_fraction) / n_neg)
            gen = torch.Generator()
            gen.manual_seed(args.seed + 1009)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double), num_samples=len(y), replacement=True, generator=gen
            )
            shuffle = False
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
    )


@torch.inference_mode()
def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for protein, sites, y, mask in loader:
        logits = model(protein.to(device), sites.to(device), mask.to(device))
        ys.append(y.numpy())
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_torch_model(model_name: str, data: FullFullData, proteins, sites, args, device):
    train_loader = make_loader(data.train, proteins, sites, args, device, train=True)
    val_loader = make_loader(data.val, proteins, sites, args, device, train=False)
    test_loader = make_loader(data.test, proteins, sites, args, device, train=False)
    model = build_model(model_name, args).to(device)
    y_train = data.train["label"].to_numpy(dtype=np.float32)
    pos = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - pos) / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_ap, best_state, best_epoch, stale = -np.inf, None, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for protein, sites_b, y, mask in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(protein.to(device), sites_b.to(device), mask.to(device))
            if model_name == "attention_excess_max":
                p = torch.sigmoid(logits)
                bce = loss_fn(logits, y.to(device))
                neg_margin = ((1.0 - y.to(device)) * torch.relu(p - args.neg_threshold)).mean()
                pos_margin = (y.to(device) * torch.relu(args.pos_threshold - p)).mean()
                loss = bce + args.margin_weight * (neg_margin + pos_margin)
            else:
                loss = loss_fn(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        yv, pv = predict_torch(model, val_loader, device)
        ap = D.metrics(yv, pv)["AUPRC"]
        print(f"    epoch {epoch:03d}: loss={np.mean(losses):.4f} val_AUPRC={ap:.4f}", flush=True)
        if ap > best_ap + args.min_delta:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    yv, pv = predict_torch(model, val_loader, device)
    yt, pt = predict_torch(model, test_loader, device)
    threshold = args.decision_threshold
    if threshold is None:
        threshold = args.pos_threshold if model_name == "attention_excess_max" and args.use_margin_threshold else best_f1_threshold(yv, pv)
    return yt, pt, yv, pv, float(threshold), int(best_epoch), model


def run_boost(data: FullFullData, proteins, mol_mean, args):
    def xy(df):
        X = np.stack(
            [np.concatenate([mol_mean[row.inchikey], proteins[row.protein]]) for row in df.itertuples()],
            axis=0,
        ).astype(np.float32)
        y = df["label"].to_numpy(dtype=np.float32)
        return X, y

    xtr, ytr = xy(data.train)
    xva, yva = xy(data.val)
    xte, yte = xy(data.test)
    # Current shared baseline helper trains on train only.  For threshold tuning we still use val.
    pte = train_boost(xtr, ytr, xte, seed=args.boost_seed)
    pva = train_boost(xtr, ytr, xva, seed=args.boost_seed)
    threshold = args.decision_threshold if args.decision_threshold is not None else best_f1_threshold(yva, pva)
    return yte, pte, yva, pva, float(threshold), 0, None


def save_predictions(path: pathlib.Path, df: pd.DataFrame, y: np.ndarray, p: np.ndarray, kind: str) -> None:
    cols = ["smiles", "inchikey", "protein", "label"]
    out = df.loc[:, cols].copy()
    out["split"] = kind
    out["y"] = y
    out["p"] = p
    out.to_csv(path, index=False)


def run_one(model_name: str, data: FullFullData, proteins, sites, mol_mean, args, device):
    start = time.time()
    if model_name == "boost_gin_mean":
        yt, pt, yv, pv, threshold, best_epoch, fitted = run_boost(data, proteins, mol_mean, args)
    else:
        yt, pt, yv, pv, threshold, best_epoch, fitted = train_torch_model(model_name, data, proteins, sites, args, device)
    rec = {
        "regime": args.regime,
        "repeat": args.repeat,
        "model": model_name,
        "seed": args.seed,
        "boost_seed": args.boost_seed,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "seconds": round(time.time() - start, 1),
        "n_train": len(data.train),
        "n_val": len(data.val),
        "n_test": len(data.test),
        "train_pos": int(data.train["label"].sum()),
        "val_pos": int(data.val["label"].sum()),
        "test_pos": int(data.test["label"].sum()),
        "embedding_set": "GIN_all_m2or_per_atom/mean + LoRaX_ESM1b_mean",
        **metric_at(yt, pt, threshold),
    }
    return rec, yt, pt, yv, pv, fitted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES, required=True)
    ap.add_argument("--repeat", type=int, required=True, help="LoRaX fold for transductive; cold seed for inductive")
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--seed", type=int, default=42, help="neural/model seed")
    ap.add_argument("--boost-seed", type=int, default=1042)
    ap.add_argument("--out-dir", default="results/attention/full_full/site_mil")
    ap.add_argument("--lorax-dir", default="data/splits_indexes/lorax_m2or")
    ap.add_argument("--lorax-bridge", default="data/processed/molecules/lorax_smiles_to_inchikey.csv")
    ap.add_argument("--protein", default="data/embeddings/proteins/esm1b_650m_mean.npz")
    ap.add_argument("--molecule-sites", default="data/embeddings/molecules/gin_supervised_contextpred_all_m2or_per_atom.npz")
    ap.add_argument("--molecule-mean", default="data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--pos-fraction", type=float, default=0.5)
    ap.add_argument("--neg-threshold", type=float, default=0.2)
    ap.add_argument("--pos-threshold", type=float, default=0.4)
    ap.add_argument("--margin-weight", type=float, default=1.0)
    ap.add_argument("--decision-threshold", type=float, default=None)
    ap.add_argument("--use-margin-threshold", action="store_true")
    ap.add_argument("--inductive-holdout-fraction", type=float, default=0.30)
    ap.add_argument("--inductive-test-fraction-within-holdout", type=float, default=2.0 / 3.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.regime == "transductive" and args.repeat not in (1, 2, 3, 4, 5):
        raise ValueError("transductive repeat must be a LoRaX fold in 1..5")
    if args.pos_fraction is not None and not (0.0 < args.pos_fraction < 1.0):
        raise ValueError("--pos-fraction must be in (0, 1)")
    seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    out = ROOT / args.out_dir
    run_id = f"{args.regime}_rep{args.repeat}_{args.model}_seed{args.seed}_boost{args.boost_seed}"
    run_dir = out / "runs" / run_id
    pred_dir = out / "predictions"
    ckpt_dir = out / "checkpoints"
    for d in (run_dir, pred_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"SKIP cached {run_id} -> {metrics_path}", flush=True)
        return

    proteins = load_npz_dict(ROOT / args.protein)
    sites = load_npz_dict(ROOT / args.molecule_sites)
    mol_mean = load_npz_dict(ROOT / args.molecule_mean)
    mol_keys = set(sites) & set(mol_mean)
    data = load_full_full_data(args, args.regime, args.repeat, mol_keys, set(proteins))

    print(
        f"RUN {run_id} device={device} train={len(data.train)} val={len(data.val)} test={len(data.test)} "
        f"test_pos={int(data.test.label.sum())}",
        flush=True,
    )
    rec, yt, pt, yv, pv, fitted = run_one(args.model, data, proteins, sites, mol_mean, args, device)
    rec.update(data.coverage)
    rec["run_id"] = run_id
    rec["config"] = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    pd.DataFrame([{k: v for k, v in rec.items() if k != "config"}]).to_csv(run_dir / "metrics.csv", index=False)
    save_predictions(pred_dir / f"test_{run_id}.csv", data.test, yt, pt, "test")
    save_predictions(pred_dir / f"val_{run_id}.csv", data.val, yv, pv, "val")
    if fitted is not None:
        torch.save(
            {"state_dict": fitted.state_dict(), "config": vars(args), "run_id": run_id, "metrics": rec},
            ckpt_dir / f"{run_id}.pt",
        )
    print("  " + " ".join(f"{k}={rec[k]:.3f}" for k in ("AUROC", "AUPRC", "MCC", "F1")), flush=True)
    print(f"UNIT_DONE {run_id}", flush=True)


if __name__ == "__main__":
    main()
