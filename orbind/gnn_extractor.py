"""Standalone pair-level "graph" cls extractor for orbind.ensemble.run_ensemble:
GnnSignedExtractor -- a heterogeneous, signed-edge GraphSAGE reimplementation
of this project's own "signed" GNN architecture (see orbind/hetero.py and the
graph-pipeline training scripts), rewritten fresh here with no import from
those scripts, mirroring attention_extractor.py's independence rule.

Node features start as *mean-pooled* GIN (molecule) / mean-pooled ESM
(protein) embeddings -- entity-level, same convention as EsmExtractor/
GinExtractor -- then get refined by two layers of signed message passing.

"Signed": message-passing edges are the train split's own positive/negative
pairs, but positive and negative edges flow through *separate* SAGEConv
stacks whose outputs are subtracted at each layer, rather than a single
sign-blind edge type (see `_SignedSage.encode`). An optional per-molecule
quantile filter (`q`, default 0.99 -- "q99") keeps only the most-measured
molecules as message-passing participants (measured by train-edge count);
supervision (the decoder's BCE loss and the features handed back to the
boosting stage) still covers every row regardless of this filter.

Leakage handling: uses the same n_models bagging pattern as
attention_extractor.py, not this project's own GNN pipeline's
--disjoint-probe-train split. Every one of `n_models` independently-seeded
whole-graph encoders sees ALL of train_idx's pairs as message-passing edges
(accepting mild train-row leakage, bounded by early stopping on real val --
the encoder "knows" a training edge exists, the same tradeoff ProSmith's own
cls model and our attention cls sources already make), then embeds every
node in the graph through itself; the N models' [molecule_embedding ||
protein_embedding] vectors are concatenated. This sidesteps the
basis-alignment problem a true fold-holdout OOF scheme would have (see
attention_extractor.py's module docstring for the full argument): every
model embeds every split, so there's no missing/misaligned block to
reconcile.
"""
from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from . import dataset as D

MOL, PROT = "mol", "prot"
ETYPE = (MOL, "binds", PROT)
RTYPE = (PROT, "rev_binds", MOL)
ETYPE_NEG = (MOL, "no_binds", PROT)
RTYPE_NEG = (PROT, "rev_no_binds", MOL)


# --------------------------------------------------------------------------- graph plumbing (no model logic)

def _build_universe(pairs: pd.DataFrame, all_idx, proteins: dict, molecules: dict):
    """Local node-id maps + initial (mean-pooled) feature tensors for every
    molecule/protein referenced anywhere in train/val/test for this call."""
    sub = pairs.iloc[all_idx]
    mols = pd.unique(sub["inchikey"])
    prots = pd.unique(sub["receptor"])
    mol_to_i = {m: i for i, m in enumerate(mols)}
    prot_to_i = {p: i for i, p in enumerate(prots)}
    x_mol = torch.tensor(np.stack([molecules[m] for m in mols]), dtype=torch.float32)
    x_prot = torch.tensor(np.stack([proteins[p] for p in prots]), dtype=torch.float32)
    return mol_to_i, prot_to_i, x_mol, x_prot


def _mp_edges(pairs: pd.DataFrame, train_idx, mol_to_i, prot_to_i, q: float):
    """Train split's own pairs -> (pos, neg) local (mol_id, prot_id) arrays,
    optionally dropping molecules below the q-th quantile of train-edge count
    (the "q99" style quality filter) from message passing only -- every train
    row still gets decoded/supervised regardless."""
    sub = pairs.iloc[train_idx]
    mol_ids = sub["inchikey"].map(mol_to_i).to_numpy()
    prot_ids = sub["receptor"].map(prot_to_i).to_numpy()
    y = sub["label"].to_numpy()
    if q and q > 0:
        counts = np.bincount(mol_ids, minlength=len(mol_to_i))
        threshold = np.quantile(counts, q)
        keep = counts >= threshold
        mask = keep[mol_ids]
        mol_ids, prot_ids, y = mol_ids[mask], prot_ids[mask], y[mask]
    pos_mask = y == 1
    return (mol_ids[pos_mask], prot_ids[pos_mask]), (mol_ids[~pos_mask], prot_ids[~pos_mask])


def _edge_index_dict(pos, neg):
    pm, pp = pos
    nm, npt = neg
    def _idx(a, b):
        if len(a) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(np.stack([a, b]), dtype=torch.long)
    pos_eidx = {ETYPE: _idx(pm, pp), RTYPE: _idx(pp, pm)}
    neg_eidx = {ETYPE_NEG: _idx(nm, npt), RTYPE_NEG: _idx(npt, nm)}
    return pos_eidx, neg_eidx


# --------------------------------------------------------------------------- model

class _SignedSage(nn.Module):
    """Two-layer heterogeneous GraphSAGE. Positive and negative edges are
    message-passed through separate SAGEConv stacks per layer, then
    subtracted (ReLU(pos - neg) after layer 1, pos - neg after layer 2) --
    sign is encoded structurally by which stack an edge flows through, not
    by a learned sign scalar. A 3-layer MLP decodes a (molecule, protein)
    node-embedding pair to one binding logit."""

    def __init__(self, mol_dim: int, prot_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj_mol = nn.Linear(mol_dim, hidden)
        self.proj_prot = nn.Linear(prot_dim, hidden)
        self.conv1 = HeteroConv({ETYPE: SAGEConv((-1, -1), hidden), RTYPE: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.conv1_neg = HeteroConv({ETYPE_NEG: SAGEConv((-1, -1), hidden), RTYPE_NEG: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.conv2 = HeteroConv({ETYPE: SAGEConv((-1, -1), hidden), RTYPE: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.conv2_neg = HeteroConv({ETYPE_NEG: SAGEConv((-1, -1), hidden), RTYPE_NEG: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.dec = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def encode(self, x_mol, x_prot, pos_eidx, neg_eidx):
        x = {MOL: F.relu(self.proj_mol(x_mol)), PROT: F.relu(self.proj_prot(x_prot))}
        x_p = self.conv1(x, pos_eidx)
        x_n = self.conv1_neg(x, neg_eidx)
        x = {k: F.relu(x_p[k] - x_n.get(k, torch.zeros_like(x_p[k]))) for k in x_p}
        x_p = self.conv2(x, pos_eidx)
        x_n = self.conv2_neg(x, neg_eidx)
        return {k: x_p[k] - x_n.get(k, torch.zeros_like(x_p[k])) for k in x_p}

    def decode(self, z, mol_idx, prot_idx):
        return self.dec(torch.cat([z[MOL][mol_idx], z[PROT][prot_idx]], dim=-1)).squeeze(-1)


def _train_one(build_model, x_mol, x_prot, pos_eidx, neg_eidx,
                mol_idx_train, prot_idx_train, y_train,
                mol_idx_val, prot_idx_val, y_val, hp, device):
    """Full-batch training loop (the whole graph is small enough to fit in
    one forward/backward per epoch): BCE loss on train-row decodes, early
    stopping on real-val AUPRC, ReduceLROnPlateau (this architecture is
    known to collapse under a flat high LR -- see orbind/hetero.py history
    and notes/ -- lr=1e-3 + grad-clip + plateau scheduling is the fix)."""
    model = build_model().to(device)
    x_mol_d, x_prot_d = x_mol.to(device), x_prot.to(device)
    pos_eidx_d = {k: v.to(device) for k, v in pos_eidx.items()}
    neg_eidx_d = {k: v.to(device) for k, v in neg_eidx.items()}
    mi_tr = torch.as_tensor(mol_idx_train, dtype=torch.long, device=device)
    pi_tr = torch.as_tensor(prot_idx_train, dtype=torch.long, device=device)
    y_tr = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    mi_va = torch.as_tensor(mol_idx_val, dtype=torch.long, device=device)
    pi_va = torch.as_tensor(prot_idx_val, dtype=torch.long, device=device)

    pos = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - pos) / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=hp["sched_patience"])

    best_ap, best_state, stale = -np.inf, None, 0
    for _ in range(hp["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model.encode(x_mol_d, x_prot_d, pos_eidx_d, neg_eidx_d)
        loss = loss_fn(model.decode(z, mi_tr, pi_tr), y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), hp["clip_grad"])
        opt.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(x_mol_d, x_prot_d, pos_eidx_d, neg_eidx_d)
            p_val = torch.sigmoid(model.decode(z, mi_va, pi_va)).cpu().numpy()
        ap = D.metrics(y_val, p_val)["AUPRC"]
        sched.step(ap)
        if ap > best_ap + hp["min_delta"]:
            best_ap, stale = ap, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= hp["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        z = model.encode(x_mol_d, x_prot_d, pos_eidx_d, neg_eidx_d)
        z_mol = z[MOL].cpu().numpy()
        z_prot = z[PROT].cpu().numpy()
    return z_mol, z_prot, model


def _run_models(ext, pairs: pd.DataFrame, train_idx, val_idx, test_idx, seed: int, checkpoint_dir=None):
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            missing = int((~mask).sum())
            raise KeyError(
                f"{ext.name}: {missing}/{len(idx)} {split_name} rows missing a protein or "
                f"molecule embedding (protein={ext.protein_path}, molecule={ext.molecule_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    mol_to_i, prot_to_i, x_mol, x_prot = _build_universe(pairs, all_idx, ext._proteins, ext._molecules)

    train_df = pairs.iloc[train_idx]
    mol_idx_train = train_df["inchikey"].map(mol_to_i).to_numpy().copy()
    prot_idx_train = train_df["receptor"].map(prot_to_i).to_numpy().copy()
    y_train = train_df["label"].to_numpy(dtype=np.float32)

    val_df = pairs.iloc[val_idx]
    mol_idx_val = val_df["inchikey"].map(mol_to_i).to_numpy().copy()
    prot_idx_val = val_df["receptor"].map(prot_to_i).to_numpy().copy()
    y_val = val_df["label"].to_numpy(dtype=np.float32)

    pos, neg = _mp_edges(pairs, train_idx, mol_to_i, prot_to_i, ext.q)
    pos_eidx, neg_eidx = _edge_index_dict(pos, neg)

    def model_job(m):
        hp = ext._hp(seed + ext.seed_offset * m)
        z_mol, z_prot, model = _train_one(ext._build_model, x_mol, x_prot, pos_eidx, neg_eidx,
                                           mol_idx_train, prot_idx_train, y_train,
                                           mol_idx_val, prot_idx_val, y_val, hp, device)
        return m, z_mol, z_prot, model

    with ThreadPoolExecutor(max_workers=ext.n_models) as pool:
        futures = [pool.submit(model_job, m) for m in range(ext.n_models)]
        results = sorted((f.result() for f in futures), key=lambda r: r[0])

    if checkpoint_dir is not None:
        for m, _, _, model in results:
            path = pathlib.Path(checkpoint_dir) / f"gnn_{ext.name}_model{m}.pt"
            torch.save(model.state_dict(), path)

    def features_for(idx):
        sub = pairs.iloc[idx]
        mi = sub["inchikey"].map(mol_to_i).to_numpy()
        pi = sub["receptor"].map(prot_to_i).to_numpy()
        blocks = [np.concatenate([z_mol[mi], z_prot[pi]], axis=1) for _, z_mol, z_prot, _ in results]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    return features_for(train_idx), features_for(val_idx), features_for(test_idx)


@dataclass
class GnnSignedExtractor:
    name: str
    protein_path: str = "data/embeddings/proteins/esm1b_650m_mean.npz"
    molecule_path: str = "data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz"
    hidden: int = 256
    dropout: float = 0.3
    q: float = 0.99
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    min_delta: float = 1e-4
    epochs: int = 300
    patience: int = 30
    sched_patience: int = 8
    n_models: int = 5
    seed_offset: int = 5000
    pooling: str = "signed_sage"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="gnn_signed")

    def __post_init__(self):
        self._proteins = D.load_npz_dict(self.protein_path)
        self._molecules = D.load_npz_dict(self.molecule_path)
        self.dim_out = self.n_models * 2 * self.hidden
        self.path = f"{self.protein_path} + {self.molecule_path}"

    def _build_model(self):
        mol_dim = next(iter(self._molecules.values())).shape[-1]
        prot_dim = next(iter(self._proteins.values())).shape[-1]
        return _SignedSage(mol_dim, prot_dim, self.hidden, self.dropout)

    def _hp(self, seed):
        return dict(lr=self.lr, weight_decay=self.weight_decay, clip_grad=self.clip_grad,
                    min_delta=self.min_delta, epochs=self.epochs, patience=self.patience,
                    sched_patience=self.sched_patience, seed=seed)

    def covered(self, pairs, idx):
        prot = pairs["receptor"].to_numpy()[idx]
        mol = pairs["inchikey"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and (m in self._molecules) for p, m in zip(prot, mol)),
                            dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)
