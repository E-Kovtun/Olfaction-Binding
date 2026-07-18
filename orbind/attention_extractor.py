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
model's own prediction.

`n_models` picks between two ways of producing that embedding (see
`_run_models`):
  n_models=1  -- ProSmith's own scheme: one model is fit on the whole of
                 train_idx and embeds train/val/test through itself. This
                 accepts mild leakage for train rows (the model that
                 produced a train row's embedding did see that row's label
                 during training, bounded by early stopping on real val)
                 in exchange for every row's embedding coming from the
                 exact same basis.
  n_models>1  -- a bagging ensemble: N independently-seeded whole-train
                 models (same accept-train-leakage scheme as above, just
                 run N times), each embeds every row through itself, and
                 their N embeddings are concatenated (N*2*dim columns).
                 Every row gets a block from every model -- no missing
                 blocks, no held-out subsets -- so there's no cross-model
                 basis mismatch to reconcile: each column range always
                 means "model k's embedding", consistently across
                 train/val/test.

An earlier version used true out-of-fold (OOF) fold-holdout retraining
instead (honest, leak-free train features, but each train row's embedding
came from whichever one of 5 *different* fold-models held it out, while
val/test came from a 6th, separately-trained whole-train model) -- that
turned out to break badly in practice: a boosting head trained on OOF
train features, whose per-row embedding basis silently varies fold to
fold, does not transfer to the single, differently-trained whole-train
model's basis used for val/test. Unlike a calibrated scalar probability
(a canonical 0-1 number, meaningful regardless of which model instance
produced it), an arbitrary hidden-layer embedding is not identifiable
across independently retrained networks. `n_models=1`/`>1` above both
sidestep this by construction: every embedding-producing model instance
always embeds every split, so there's never a "some rows only have model
A's basis, other rows only have model B's" split to misalign.
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


def _run_models(ext, pairs: pd.DataFrame, train_idx, val_idx, test_idx, seed: int, checkpoint_dir=None):
    """Train `ext.n_models` independent whole-train model instances -- same
    train_idx for every one of them, no fold holdouts -- each embedding
    train/val/test through itself (accepting mild train-row leakage,
    bounded by early stopping on real val; see the module docstring for why
    this replaced a true OOF fold-holdout scheme). Concatenate the N
    models' embeddings into one feature: with n_models=1 this is exactly
    ProSmith's own single-model scheme; with n_models>1 it's a bagging
    ensemble of it. `ext` supplies `_build_model()`, `_hp(seed)`,
    `_proteins`/`_sites`, `n_models`, `seed_offset`.

    All N models are independent trainings (each builds its own fresh
    model, no shared state) and are always run concurrently via a thread
    pool -- not an option, since every future pair-level extractor needs
    this same shape.

    If `checkpoint_dir` is given, every model's state_dict is saved as
    `attn_{ext.name}_model{k}.pt`."""
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            missing = int((~mask).sum())
            raise KeyError(
                f"{ext.name}: {missing}/{len(idx)} {split_name} rows missing a protein or "
                f"per-atom molecule embedding (protein={ext.protein_path}, sites={ext.molecule_sites_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = pairs.iloc[train_idx], pairs.iloc[val_idx], pairs.iloc[test_idx]

    def model_job(m):
        hp = ext._hp(seed + ext.seed_offset * m)
        e_val, e_test, model = _train_and_predict(ext._build_model, train_df, val_df, test_df,
                                                    ext._proteins, ext._sites, hp, device)
        train_loader = _make_loader(train_df, ext._proteins, ext._sites, hp["batch_size"], None, hp["seed"], train=False)
        e_train = _embed(model, train_loader, device)
        return m, e_train, e_val, e_test, model

    with ThreadPoolExecutor(max_workers=ext.n_models) as pool:
        futures = [pool.submit(model_job, m) for m in range(ext.n_models)]
        results = sorted((f.result() for f in futures), key=lambda r: r[0])

    if checkpoint_dir is not None:
        for m, _, _, _, model in results:
            checkpoint_path = pathlib.Path(checkpoint_dir) / f"attn_{ext.name}_model{m}.pt"
            torch.save(model.state_dict(), checkpoint_path)

    p_train = np.concatenate([r[1] for r in results], axis=1)
    p_val = np.concatenate([r[2] for r in results], axis=1)
    p_test = np.concatenate([r[3] for r in results], axis=1)
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
    dim: int = 32
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    min_delta: float = 1e-4
    pos_fraction: float | None = 0.5
    epochs: int = 80
    patience: int = 12
    batch_size: int = 256
    n_models: int = 5
    seed_offset: int = 5000
    pooling: str = "noisy_or"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="mil_noisy_or")

    def __post_init__(self):
        self.dim_out = self.n_models * 2 * self.dim
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
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)


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
    dim: int = 32
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
    n_models: int = 5
    seed_offset: int = 5000
    pooling: str = "lse"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="mil_lse")

    def __post_init__(self):
        self.dim_out = self.n_models * 2 * self.dim
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
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)
