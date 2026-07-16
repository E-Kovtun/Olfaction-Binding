"""Two standalone, self-contained pair-level "cls" extractors for
orbind.ensemble.run_ensemble: MilNoisyOrExtractor and MilLseExtractor.

Per this project's own site-MIL screen, these two per-site pooling rules
(noisy-OR and log-sum-exp over per-site logits) are the current leaders.
Each wraps its own small torch model over protein mean-ESM + per-atom
molecule-GIN embeddings -- the architecture is written out directly here,
not imported from scripts/modeling/train/train_full_full_site_mil_attention.py.
Only generic, model-agnostic plumbing (dataset/collate/training-loop/OOF
driver) is shared between the two, so each model's own definition stays
fully independent and directly editable.

Both are supervised, pair-level sources: each trains its own model on
train_idx's labels (via a scalar noisy-OR/LSE prediction head, BCE loss).
The *feature* handed back to the boosting stage, though, is not that scalar
prediction -- it's the pooled 2*dim hidden representation the head would
otherwise collapse to one number (`model.embed`, weighted by the same
noisy-OR/LSE pooling weights), so the boosting stage gets an actual
embedding ("cls token") rather than a degenerate 1-d re-statement of the
model's own prediction. To avoid leaking a row's own label into that
embedding for train rows, train embeddings are produced out-of-fold
(stratified K-fold retraining internally, each fold gets its own inner
train/val split for early stopping); val/test embeddings come from one
model fit on the whole of train_idx (which never touches val/test, so no
leakage there either).
"""
from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import dataset as D


# --------------------------------------------------------------------------- generic data plumbing (no model logic)

class _PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, proteins: dict, sites: dict):
        self.pairs = pairs.reset_index(drop=True)
        self.proteins = proteins
        self.sites = sites

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        return (np.asarray(self.proteins[row.receptor], dtype=np.float32),
                np.asarray(self.sites[row.inchikey], dtype=np.float32),
                np.float32(row.label))


def _collate(batch):
    proteins, sites, labels = zip(*batch)
    lengths = [len(x) for x in sites]
    site_dim = sites[0].shape[-1]
    x = torch.zeros(len(sites), max(lengths), site_dim, dtype=torch.float32)
    mask = torch.ones(len(sites), max(lengths), dtype=torch.bool)
    for i, site in enumerate(sites):
        x[i, :len(site)] = torch.from_numpy(site)
        mask[i, :len(site)] = False
    return torch.from_numpy(np.stack(proteins)), x, torch.tensor(labels, dtype=torch.float32), mask


def _make_loader(df, proteins, sites, batch_size, pos_fraction, seed, train):
    ds = _PairDataset(df, proteins, sites)
    sampler, shuffle = None, train
    if train and pos_fraction is not None:
        y = df["label"].to_numpy(dtype=np.int64)
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        if n_pos and n_neg:
            weights = np.where(y == 1, pos_fraction / n_pos, (1.0 - pos_fraction) / n_neg)
            gen = torch.Generator()
            gen.manual_seed(seed + 1009)
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                             num_samples=len(y), replacement=True, generator=gen)
            shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                       sampler=sampler, collate_fn=_collate)


@torch.inference_mode()
def _predict(model, loader, device):
    """Scalar probabilities -- used only for AUPRC-based early stopping during
    training, never handed back as the extractor's feature (see `_embed`)."""
    model.eval()
    ps = []
    for protein, sites, _, mask in loader:
        logits = model(protein.to(device), sites.to(device), mask.to(device))
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(ps)


@torch.inference_mode()
def _embed(model, loader, device):
    """Pooled pre-head embedding (model.embed) -- this is the actual feature
    handed back to the boosting stage, not the scalar prediction."""
    model.eval()
    es = []
    for protein, sites, _, mask in loader:
        e = model.embed(protein.to(device), sites.to(device), mask.to(device))
        es.append(e.detach().cpu().numpy())
    return np.concatenate(es, axis=0)


def _train_and_predict(build_model, train_df, val_df, test_df, proteins, sites, hp, device):
    """Generic epoch loop (BCE loss, AdamW, early stopping on val AUPRC).
    `build_model()` supplies the extractor-specific nn.Module -- this
    function itself has no opinion on architecture."""
    train_loader = _make_loader(train_df, proteins, sites, hp["batch_size"], hp["pos_fraction"], hp["seed"], train=True)
    val_loader = _make_loader(val_df, proteins, sites, hp["batch_size"], None, hp["seed"], train=False)
    test_loader = _make_loader(test_df, proteins, sites, hp["batch_size"], None, hp["seed"], train=False)

    model = build_model().to(device)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    pos = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - pos) / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    y_val = val_df["label"].to_numpy(dtype=np.float32)

    best_ap, best_state, stale = -np.inf, None, 0
    for _ in range(hp["epochs"]):
        model.train()
        for protein, sites_b, y, mask in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(protein.to(device), sites_b.to(device), mask.to(device))
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["clip_grad"])
            opt.step()
        pv = _predict(model, val_loader, device)
        ap = D.metrics(y_val, pv)["AUPRC"]
        if ap > best_ap + hp["min_delta"]:
            best_ap, stale = ap, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= hp["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return _embed(model, val_loader, device), _embed(model, test_loader, device), model


def _run_oof(ext, pairs: pd.DataFrame, train_idx, val_idx, test_idx, seed: int, checkpoint_dir=None):
    """Shared OOF driver: val/test from one whole-train fit; train from
    stratified K-fold out-of-fold refits. `ext` supplies `_build_model()`,
    `_hp(seed)`, `_proteins`/`_sites`, `n_folds`, `inner_val_fraction`,
    `seed_offset` -- identical for both extractors below, just parameterized
    by which model class `_build_model` returns.

    The whole-train fit and all K OOF folds are independent trainings (each
    builds its own fresh model, no shared state) and are always run
    concurrently via a thread pool -- not an option, since every future
    pair-level extractor needs this same honest-OOF shape.

    If `checkpoint_dir` is given, the whole-train-fit model (the one that
    produced val/test -- never the throwaway per-fold OOF models, which only
    exist to give train rows a leak-free feature) is saved there as
    `attn_{ext.name}.pt` (a plain `state_dict`)."""
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            missing = int((~mask).sum())
            raise KeyError(
                f"{ext.name}: {missing}/{len(idx)} {split_name} rows missing a protein or "
                f"per-atom molecule embedding (protein={ext.protein_path}, sites={ext.molecule_sites_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = pairs.iloc[train_idx], pairs.iloc[val_idx], pairs.iloc[test_idx]
    y_train = train_df["label"].to_numpy()

    def whole_train_job():
        return _train_and_predict(ext._build_model, train_df, val_df, test_df,
                                   ext._proteins, ext._sites, ext._hp(seed), device)

    def fold_job(fold_i, inner_pos, holdout_pos):
        inner_df = train_df.iloc[inner_pos]
        holdout_df = train_df.iloc[holdout_pos]
        fit_pos, innerval_pos = train_test_split(
            np.arange(len(inner_df)), test_size=ext.inner_val_fraction,
            random_state=seed + ext.seed_offset + fold_i, stratify=inner_df["label"].to_numpy())
        _, e_holdout, _ = _train_and_predict(ext._build_model, inner_df.iloc[fit_pos], inner_df.iloc[innerval_pos],
                                              holdout_df, ext._proteins, ext._sites,
                                              ext._hp(seed + ext.seed_offset + fold_i), device)
        return holdout_pos, e_holdout

    skf = StratifiedKFold(n_splits=ext.n_folds, shuffle=True, random_state=seed + ext.seed_offset)
    fold_splits = list(skf.split(train_df, y_train))
    p_train = np.full((len(train_df), ext.dim_out), np.nan, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=ext.n_folds + 1) as pool:
        whole_future = pool.submit(whole_train_job)
        fold_futures = [pool.submit(fold_job, i, inner_pos, holdout_pos)
                        for i, (inner_pos, holdout_pos) in enumerate(fold_splits)]
        p_val, p_test, whole_model = whole_future.result()
        for f in fold_futures:
            holdout_pos, e_holdout = f.result()
            p_train[holdout_pos] = e_holdout

    if checkpoint_dir is not None:
        checkpoint_path = pathlib.Path(checkpoint_dir) / f"attn_{ext.name}.pt"
        torch.save(whole_model.state_dict(), checkpoint_path)

    assert not np.isnan(p_train).any(), "OOF pass left some train rows unfilled"
    return p_train.astype(np.float32), p_val.astype(np.float32), p_test.astype(np.float32)


# --------------------------------------------------------------------------- MilNoisyOrExtractor

class _NoisyOrSiteModel(nn.Module):
    """Per-site logits -> noisy-OR combine ("at least one site fires").

    `forward` (scalar logit, used only for training/early-stopping) and
    `embed` (pooled 2*dim hidden vector, the actual feature handed to the
    boosting stage) share the same per-site hidden representation -- `embed`
    just pools it with noisy-OR-style weights instead of collapsing it to
    one probability."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.protein = nn.Sequential(nn.Linear(1280, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.site = nn.Sequential(nn.Linear(300, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.hidden = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(2 * dim, 1)

    def _site_hidden_and_logits(self, protein, sites, mask):
        p = self.protein(protein)
        s = self.site(sites)
        p_rep = p.unsqueeze(1).expand_as(s)
        h = self.hidden(torch.cat([s, p_rep, s * p_rep], dim=-1))
        logits = self.out(h).squeeze(-1).masked_fill(mask, -torch.inf)
        return h, logits

    def forward(self, protein, sites, mask):
        _, logits = self._site_hidden_and_logits(protein, sites, mask)
        probs = torch.sigmoid(logits).masked_fill(mask, 0.0)
        log_no_hit = torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(1)
        p_hit = 1.0 - torch.exp(log_no_hit)
        return torch.logit(p_hit.clamp(1e-6, 1.0 - 1e-6))

    def embed(self, protein, sites, mask):
        h, logits = self._site_hidden_and_logits(protein, sites, mask)
        probs = torch.sigmoid(logits).masked_fill(mask, 0.0)
        w = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (h * w.unsqueeze(-1)).sum(dim=1)


@dataclass
class MilNoisyOrExtractor:
    name: str
    protein_path: str = "data/embeddings/proteins/esm1b_650m_mean.npz"
    molecule_sites_path: str = "data/embeddings/molecules/gin_supervised_contextpred_all_m2or_per_atom.npz"
    dim: int = 64
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    min_delta: float = 1e-4
    pos_fraction: float | None = 0.5
    epochs: int = 80
    patience: int = 12
    batch_size: int = 256
    n_folds: int = 5
    inner_val_fraction: float = 0.1
    seed_offset: int = 5000
    pooling: str = "noisy_or"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="mil_noisy_or")

    def __post_init__(self):
        self.dim_out = 2 * self.dim
        self._proteins = D.load_npz_dict(self.protein_path)
        self._sites = D.load_npz_dict(self.molecule_sites_path)
        self.path = f"{self.protein_path} + {self.molecule_sites_path}"

    def _build_model(self):
        return _NoisyOrSiteModel(self.dim, self.dropout)

    def _hp(self, seed):
        return dict(lr=self.lr, weight_decay=self.weight_decay, clip_grad=self.clip_grad,
                    min_delta=self.min_delta, pos_fraction=self.pos_fraction, epochs=self.epochs,
                    patience=self.patience, batch_size=self.batch_size, seed=seed)

    def covered(self, pairs, idx):
        prot = pairs["receptor"].to_numpy()[idx]
        mol = pairs["inchikey"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and (m in self._sites) for p, m in zip(prot, mol)),
                            dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_oof(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)


# --------------------------------------------------------------------------- MilLseExtractor

class _LseSiteModel(nn.Module):
    """Per-site logits -> temperature-scaled log-sum-exp combine (smooth max).

    `forward` (scalar logit, used only for training/early-stopping) and
    `embed` (pooled 2*dim hidden vector, the actual feature handed to the
    boosting stage) share the same per-site hidden representation -- `embed`
    pools it with the same softmax-over-temperature weights LSE implies,
    instead of collapsing it to one probability."""

    def __init__(self, dim: int, dropout: float, temperature: float):
        super().__init__()
        self.protein = nn.Sequential(nn.Linear(1280, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.site = nn.Sequential(nn.Linear(300, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.hidden = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(2 * dim, 1)
        self.temperature = temperature

    def _site_hidden_and_logits(self, protein, sites, mask):
        p = self.protein(protein)
        s = self.site(sites)
        p_rep = p.unsqueeze(1).expand_as(s)
        h = self.hidden(torch.cat([s, p_rep, s * p_rep], dim=-1))
        logits = self.out(h).squeeze(-1).masked_fill(mask, -torch.inf)
        return h, logits

    def forward(self, protein, sites, mask):
        _, logits = self._site_hidden_and_logits(protein, sites, mask)
        n = (~mask).sum(1).clamp_min(1).to(logits.dtype)
        return self.temperature * (torch.logsumexp(logits / self.temperature, dim=1) - n.log())

    def embed(self, protein, sites, mask):
        h, logits = self._site_hidden_and_logits(protein, sites, mask)
        w = torch.softmax(logits / self.temperature, dim=1)
        return (h * w.unsqueeze(-1)).sum(dim=1)


@dataclass
class MilLseExtractor:
    name: str
    protein_path: str = "data/embeddings/proteins/esm1b_650m_mean.npz"
    molecule_sites_path: str = "data/embeddings/molecules/gin_supervised_contextpred_all_m2or_per_atom.npz"
    dim: int = 64
    dropout: float = 0.1
    temperature: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    min_delta: float = 1e-4
    pos_fraction: float | None = 0.5
    epochs: int = 80
    patience: int = 12
    batch_size: int = 256
    n_folds: int = 5
    inner_val_fraction: float = 0.1
    seed_offset: int = 5000
    pooling: str = "lse"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="mil_lse")

    def __post_init__(self):
        self.dim_out = 2 * self.dim
        self._proteins = D.load_npz_dict(self.protein_path)
        self._sites = D.load_npz_dict(self.molecule_sites_path)
        self.path = f"{self.protein_path} + {self.molecule_sites_path}"

    def _build_model(self):
        return _LseSiteModel(self.dim, self.dropout, self.temperature)

    def _hp(self, seed):
        return dict(lr=self.lr, weight_decay=self.weight_decay, clip_grad=self.clip_grad,
                    min_delta=self.min_delta, pos_fraction=self.pos_fraction, epochs=self.epochs,
                    patience=self.patience, batch_size=self.batch_size, seed=seed)

    def covered(self, pairs, idx):
        prot = pairs["receptor"].to_numpy()[idx]
        mol = pairs["inchikey"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and (m in self._sites) for p, m in zip(prot, mol)),
                            dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_oof(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)
