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
(accepting mild train-row leakage -- the encoder "knows" a training edge
exists, the same tradeoff ProSmith's own cls model and our attention cls
sources already make), then embeds every node in the graph through itself;
the N models' per-row vectors are concatenated. This sidesteps the
basis-alignment problem a true fold-holdout OOF scheme would have (see
attention_extractor.py's module docstring for the full argument): every
model embeds every split, so there's no missing/misaligned block to
reconcile.

`emit` picks *which* node embeddings leave the extractor (training is
identical either way -- the decoder always sees both sides):
  "prot" (default) -- protein only (`n_models * hidden` columns). This is
                      what the v5 graph screen's own "unentangled boost"
                      probe did: it fed XGBoost
                      [raw ChemBERTa molecule || graph-enriched ESM protein]
                      (see train_graph_full_full.py's `probe_with`), never
                      the graph's molecule vector -- consistent with this
                      project's finding that the graph helps cold-molecule
                      generalization through the *protein* side, while
                      graph-enriched molecule features hurt (see the
                      inductive-enrichment study in notes/). Pair this
                      source with a raw molecule source in the same combo to
                      reproduce v5's exact feature set.
  "both"           -- [molecule_embedding || protein_embedding],
                      `n_models * 2 * hidden` columns.

Training protocol matches the actual v5 graph-architecture-screen runs
(scripts/modeling/train/run_graph_full_full_v5.ps1: lr=3e-3, epochs=900,
`--probe-checkpoint last`, no `--lr-scheduler`) rather than this project's
earlier anti-collapse fix (lr=1e-3 + grad-clip + ReduceLROnPlateau +
best-val checkpoint selection, see notes/ and orbind/hetero.py history) --
fixed epoch count, no early stopping, no LR scheduler, last epoch's weights
are always what gets kept. Grad-clip (1.0) is kept from the anti-collapse
fix since v5's own command also passes `--grad-clip 1.0`.
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
from . import mol_selection
from .tasks import check_task

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


def _mp_edges(pairs: pd.DataFrame, train_idx, mol_to_i, prot_to_i, q: float,
              criterion: str = "coverage", edge_threshold: float = 0.0,
              k_mode: str = "coverage_quantile"):
    """Train split's own pairs -> (pos, neg) local (mol_id, prot_id) arrays,
    optionally dropping molecules below the q-th quantile from message passing
    only -- every train row still gets decoded/supervised regardless.

    The quantile sets how many molecules to keep (coverage-quantile count); the
    `criterion` (see orbind/mol_selection.CRITERIA) picks WHICH ones.

    This function's own fallback stays "coverage" because that value reproduces
    the historical "keep counts >= quantile(counts, q)" filter bit-for-bit
    (zero-coverage molecules carry no edges, so their mask value is
    irrelevant). `GnnSignedExtractor` -- the only real caller -- always passes
    its own criterion explicitly, and **its** default is now
    "greedy_pair_cover"; see the note on that field."""
    sub = pairs.iloc[train_idx]
    mol_ids = sub["inchikey"].map(mol_to_i).to_numpy()
    prot_ids = sub["receptor"].map(prot_to_i).to_numpy()
    y = sub["label"].to_numpy()
    if q and q > 0:
        keep = mol_selection.select_keep_mask(
            criterion, mol_ids, prot_ids, y, len(mol_to_i), len(prot_to_i), q, k_mode)
        mask = keep[mol_ids]
        mol_ids, prot_ids, y = mol_ids[mask], prot_ids[mask], y[mask]
    # Which edges are "positive" for the signed message passing. `y > 0` is
    # IDENTICAL to the historical `y == 1` on binary labels, so every M2OR run
    # reproduces bit-for-bit; on a z-scored continuous response it reads as
    # "responds above the pool average", which is what a z-score's zero means.
    # Raise `edge_threshold` to make the positive graph stricter.
    pos_mask = y > edge_threshold
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
    subtracted (LeakyReLU(pos - neg) after layer 1, pos - neg after layer 2)
    -- sign is encoded structurally by which stack an edge flows through,
    not by a learned sign scalar. LeakyReLU (not plain ReLU) everywhere in
    this module: a unit whose pre-activation goes negative under plain ReLU
    gets zero gradient and can never recover, which is what plain ReLU
    contributed to in the collapse this architecture is known for (see
    notes/ and orbind/hetero.py history, [[gnn-collapse-fix]]) -- a small
    negative-side slope keeps every unit's gradient alive. A 3-layer MLP
    decodes a (molecule, protein) node-embedding pair to one binding logit."""

    LEAK = 0.1

    def __init__(self, mol_dim: int, prot_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj_mol = nn.Linear(mol_dim, hidden)
        self.proj_prot = nn.Linear(prot_dim, hidden)
        self.conv1 = HeteroConv({ETYPE: SAGEConv((-1, -1), hidden), RTYPE: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.conv1_neg = HeteroConv({ETYPE_NEG: SAGEConv((-1, -1), hidden), RTYPE_NEG: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.conv2 = HeteroConv({ETYPE: SAGEConv((-1, -1), hidden), RTYPE: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.conv2_neg = HeteroConv({ETYPE_NEG: SAGEConv((-1, -1), hidden), RTYPE_NEG: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.dec = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LeakyReLU(self.LEAK), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(self.LEAK), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def encode(self, x_mol, x_prot, pos_eidx, neg_eidx):
        x = {MOL: F.leaky_relu(self.proj_mol(x_mol), self.LEAK),
             PROT: F.leaky_relu(self.proj_prot(x_prot), self.LEAK)}
        x_p = self.conv1(x, pos_eidx)
        x_n = self.conv1_neg(x, neg_eidx)
        x = {k: F.leaky_relu(x_p[k] - x_n.get(k, torch.zeros_like(x_p[k])), self.LEAK) for k in x_p}
        x_p = self.conv2(x, pos_eidx)
        x_n = self.conv2_neg(x, neg_eidx)
        return {k: x_p[k] - x_n.get(k, torch.zeros_like(x_p[k])) for k in x_p}

    def decode(self, z, mol_idx, prot_idx):
        return self.dec(torch.cat([z[MOL][mol_idx], z[PROT][prot_idx]], dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------- DGI (Deep Graph Infomax)

def _dgi_pool(z, scope):
    """Pool the encoder's node embeddings into the DGI node set, L2-normalized
    (the signed encoder's final layer has no activation, so raw embeddings are
    unbounded and would saturate the sigmoid readout). scope="shared" pools
    molecules+proteins into one set; scope="prot" uses proteins only."""
    if scope == "prot":
        return F.normalize(z[PROT], dim=-1)
    return torch.cat([F.normalize(z[MOL], dim=-1), F.normalize(z[PROT], dim=-1)], dim=0)


def _dgi_loss(weight_mat, z_pos, z_neg):
    """Deep Graph Infomax loss: a bilinear discriminator tells real node
    embeddings (which should agree with the global summary) from embeddings
    produced on a corrupted graph. Mirrors torch_geometric's DeepGraphInfomax
    math without the wrapper (our encoder is heterogeneous, returning a dict)."""
    summary = torch.sigmoid(z_pos.mean(dim=0))
    pos = z_pos @ torch.matmul(weight_mat, summary)
    neg = z_neg @ torch.matmul(weight_mat, summary)
    return (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))


def _train_one(build_model, x_mol, x_prot, pos_eidx, neg_eidx,
                mol_idx_train, prot_idx_train, y_train, hp, device, checkpoint_path=None,
                dgi_weight=0.0, dgi_scope="shared", hidden=None):
    """Full-batch training loop (the whole graph is small enough to fit in
    one forward/backward per epoch): BCE loss on train-row decodes, fixed
    epoch count, no early stopping, no LR scheduler -- matches the actual v5
    graph-screen protocol (scripts/modeling/train/run_graph_full_full_v5.ps1:
    lr=3e-3, epochs=900, `--probe-checkpoint last`, no `--lr-scheduler` flag),
    not the earlier anti-collapse fix (lr=1e-3 + ReduceLROnPlateau + best-val
    checkpoint) this module used before -- the last epoch's weights are
    always what gets kept, val is not consulted during training at all.

    If `checkpoint_path` already exists on disk, training is skipped
    entirely: the state_dict is loaded and only the final encode pass runs
    -- lets a resumed run reuse a previously-trained model instance instead
    of retraining it from scratch.

    `dgi_weight`>0 adds a DeepGraphInfomax auxiliary term (see `_dgi_loss`):
    each epoch a corrupted graph (row-shuffled input features, same edges) is
    re-encoded and contrasted against the real graph's summary via a bilinear
    discriminator whose weights (`hidden`x`hidden`) join the optimizer. The
    discriminator is training-only -- it is not saved and not needed to emit
    embeddings, so a reloaded checkpoint ignores DGI entirely. `dgi_scope`
    picks the node set (see `_dgi_pool`)."""
    model = build_model().to(device)
    x_mol_d, x_prot_d = x_mol.to(device), x_prot.to(device)
    pos_eidx_d = {k: v.to(device) for k, v in pos_eidx.items()}
    neg_eidx_d = {k: v.to(device) for k, v in neg_eidx.items()}

    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        with torch.no_grad():
            z = model.encode(x_mol_d, x_prot_d, pos_eidx_d, neg_eidx_d)
            return z[MOL].cpu().numpy(), z[PROT].cpu().numpy(), model

    mi_tr = torch.as_tensor(mol_idx_train, dtype=torch.long, device=device)
    pi_tr = torch.as_tensor(prot_idx_train, dtype=torch.long, device=device)
    y_tr = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    # Deliberately NOT routed through orbind.tasks.batch_loss_fn: this decoder
    # uses a single `pos_weight` scalar, not per-row weights, and rewriting it
    # would change every existing M2OR number. The classification branch stays
    # byte-identical; regression just swaps in plain squared error.
    if hp.get("task", "classification") == "regression":
        loss_fn = nn.MSELoss()
    else:
        pos = float(y_train.sum())
        pos_weight = torch.tensor([(len(y_train) - pos) / max(pos, 1.0)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dgi_w = None
    params = list(model.parameters())
    if dgi_weight > 0:
        dgi_w = nn.init.xavier_uniform_(torch.empty(hidden, hidden, device=device)).requires_grad_(True)
        params = params + [dgi_w]
    opt = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["weight_decay"])

    for _ in range(hp["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model.encode(x_mol_d, x_prot_d, pos_eidx_d, neg_eidx_d)
        loss = loss_fn(model.decode(z, mi_tr, pi_tr), y_tr)
        if dgi_w is not None:
            x_mol_c = x_mol_d[torch.randperm(x_mol_d.shape[0], device=device)]
            x_prot_c = x_prot_d[torch.randperm(x_prot_d.shape[0], device=device)]
            z_c = model.encode(x_mol_c, x_prot_c, pos_eidx_d, neg_eidx_d)
            loss = loss + dgi_weight * _dgi_loss(dgi_w, _dgi_pool(z, dgi_scope), _dgi_pool(z_c, dgi_scope))
        loss.backward()
        nn.utils.clip_grad_norm_(params, hp["clip_grad"])
        opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x_mol_d, x_prot_d, pos_eidx_d, neg_eidx_d)
        z_mol = z[MOL].cpu().numpy()
        z_prot = z[PROT].cpu().numpy()
    return z_mol, z_prot, model


# Criteria computable without labels, hence usable under task="regression".
# `coverage` is a bincount over train edges; `greedy_pair_cover` orders molecules
# by newly covered protein PAIRS, built from `prof`/`active` only. The other five
# are functions of npos/nneg (`y == 1` / `y == 0`) and collapse to zeros on a
# continuous target. See orbind/mol_selection.compute_mol_scores.
_LABEL_AGNOSTIC_CRITERIA = {"coverage", "greedy_pair_cover"}


def _run_models(ext, pairs: pd.DataFrame, train_idx, val_idx, test_idx, seed: int, checkpoint_dir=None):
    check_task(ext.task)
    if ext.task == "regression" and ext.criterion not in _LABEL_AGNOSTIC_CRITERIA:
        # Every other criterion is built from npos/nneg, i.e. `y == 1` / `y == 0`
        # counts, which are empty on a continuous target -- the ranking would
        # silently collapse to zeros rather than fail.
        raise ValueError(
            f"{ext.name}: criterion {ext.criterion!r} scores molecules by their "
            f"positive/negative mix and is undefined for task='regression'; "
            f"use one of {sorted(_LABEL_AGNOSTIC_CRITERIA)}")
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

    pos, neg = _mp_edges(pairs, train_idx, mol_to_i, prot_to_i, ext.q,
                         getattr(ext, "criterion", "greedy_pair_cover"),
                         getattr(ext, "edge_threshold", 0.0),
                         getattr(ext, "k_mode", "coverage_quantile"))
    n_pos, n_neg = len(pos[0]), len(neg[0])
    print(f"  {ext.name}: MP graph {n_pos} positive / {n_neg} negative edges "
          f"(q={ext.q}, criterion={getattr(ext, 'criterion', 'greedy_pair_cover')}, "
          f"k_mode={getattr(ext, 'k_mode', 'coverage_quantile')}, "
          f"edge_threshold={getattr(ext, 'edge_threshold', 0.0)})", flush=True)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"{ext.name}: signed message passing needs both edge signs, got "
            f"{n_pos} positive / {n_neg} negative. Check `edge_threshold` "
            f"(={getattr(ext, 'edge_threshold', 0.0)}) against the label scale.")
    pos_eidx, neg_eidx = _edge_index_dict(pos, neg)

    def model_job(m):
        hp = ext._hp(seed + ext.seed_offset * m)
        checkpoint_path = (pathlib.Path(checkpoint_dir) / f"gnn_{ext.name}_model{m}.pt"
                            if checkpoint_dir is not None else None)
        z_mol, z_prot, model = _train_one(ext._build_model, x_mol, x_prot, pos_eidx, neg_eidx,
                                           mol_idx_train, prot_idx_train, y_train, hp, device,
                                           checkpoint_path=checkpoint_path,
                                           dgi_weight=getattr(ext, "dgi_weight", 0.0),
                                           dgi_scope=getattr(ext, "dgi_scope", "shared"),
                                           hidden=ext.hidden)
        return m, z_mol, z_prot, model

    with ThreadPoolExecutor(max_workers=ext.n_models) as pool:
        futures = [pool.submit(model_job, m) for m in range(ext.n_models)]
        results = sorted((f.result() for f in futures), key=lambda r: r[0])

    if checkpoint_dir is not None:
        for m, _, _, model in results:
            path = pathlib.Path(checkpoint_dir) / f"gnn_{ext.name}_model{m}.pt"
            if not path.exists():
                torch.save(model.state_dict(), path)

    def features_for(idx):
        sub = pairs.iloc[idx]
        mi = sub["inchikey"].map(mol_to_i).to_numpy()
        pi = sub["receptor"].map(prot_to_i).to_numpy()
        if ext.emit == "prot":
            blocks = [z_prot[pi] for _, _, z_prot, _ in results]
        else:
            blocks = [np.concatenate([z_mol[mi], z_prot[pi]], axis=1)
                       for _, z_mol, z_prot, _ in results]
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
    # MP molecule-keep ranking (mol_selection.CRITERIA).
    #
    # DEFAULT CHANGED Aug 2026: "coverage" -> "greedy_pair_cover". The quantile
    # decides HOW MANY molecules survive, the criterion decides WHICH. At q=0.99
    # that is 6 molecules out of 596, and the two criteria disagree almost
    # completely (1 molecule in common): coverage takes the most-measured ones
    # and yields a near-all-negative graph (3751 edges, 31 positive), while
    # greedy maximises newly covered protein PAIRS and yields 2799 edges with
    # 137 positive.
    #
    # Consequence to remember: every result produced before this change used
    # "coverage" and recorded no explicit criterion in its config.json, so
    # re-running an old command now reproduces a DIFFERENT model. Name the
    # criterion explicitly in commands whose numbers you intend to keep.
    criterion: str = "greedy_pair_cover"
    # How `q` turns into K (mol_selection.resolve_K). "coverage_quantile" is the
    # M2OR-era reading and stays the default so every existing number
    # reproduces. On Carey/Hallem it is a NO-OP -- those matrices are complete,
    # so coverage is constant across train molecules and `cov >= quantile(cov,q)`
    # keeps all of them (CC/our_inductive: q=0.99 and q=0 both give K=70 of 70).
    # Use "fraction" there: keep the top (1-q) share outright.
    k_mode: str = "coverage_quantile"
    # The task axis (orbind/tasks.py). `run_ensemble` overwrites this to match
    # the run, so the decoder's criterion and the boosting head downstream agree.
    task: str = "classification"
    # Label above which an edge joins the POSITIVE message-passing graph.
    # 0.0 reproduces the historical `y == 1` exactly on binary labels.
    edge_threshold: float = 0.0
    lr: float = 3e-3
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    epochs: int = 900
    # DEFAULT CHANGED Aug 2026: 5 -> 1, together with the criterion switch above.
    # The two form one "tactic": greedy selection with a single model, no
    # init-bagging. Consequence for feature width: emit="prot" now yields
    # `hidden` = 256 columns, not 5 x 256 = 1280.
    n_models: int = 1
    seed_offset: int = 5000
    emit: str = "prot"
    pooling: str = "signed_sage"
    dgi_weight: float = 0.0            # >0 enables the DeepGraphInfomax auxiliary loss
    dgi_scope: str = "shared"          # "shared" (mol+prot) or "prot"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="gnn_signed")

    def __post_init__(self):
        if self.emit not in ("prot", "both"):
            raise ValueError(f"emit must be 'prot' or 'both', got {self.emit!r}")
        if self.criterion not in mol_selection.CRITERIA:
            raise ValueError(f"criterion must be one of {mol_selection.CRITERIA}, "
                             f"got {self.criterion!r}")
        if self.k_mode not in mol_selection.K_MODES:
            raise ValueError(f"k_mode must be one of {mol_selection.K_MODES}, "
                             f"got {self.k_mode!r}")
        if self.dgi_scope not in ("shared", "prot"):
            raise ValueError(f"dgi_scope must be 'shared' or 'prot', got {self.dgi_scope!r}")
        self._proteins = D.load_npz_dict(self.protein_path)
        self._molecules = D.load_npz_dict(self.molecule_path)
        per_model = self.hidden if self.emit == "prot" else 2 * self.hidden
        self.dim_out = self.n_models * per_model
        self.path = f"{self.protein_path} + {self.molecule_path}"

    def _build_model(self):
        mol_dim = next(iter(self._molecules.values())).shape[-1]
        prot_dim = next(iter(self._proteins.values())).shape[-1]
        return _SignedSage(mol_dim, prot_dim, self.hidden, self.dropout)

    def _hp(self, seed):
        return dict(lr=self.lr, weight_decay=self.weight_decay, clip_grad=self.clip_grad,
                    epochs=self.epochs, task=self.task, seed=seed)

    def covered(self, pairs, idx):
        prot = pairs["receptor"].to_numpy()[idx]
        mol = pairs["inchikey"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and (m in self._molecules) for p, m in zip(prot, mol)),
                            dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)


@dataclass
class GnnSignedDgiExtractor(GnnSignedExtractor):
    """Signed GraphSAGE cls source (identical to GnnSignedExtractor -- q99,
    signed MP, emit=prot, v5-parity training, n_models bagging) PLUS a
    DeepGraphInfomax auxiliary loss mixed into training: shared scope
    (molecules+proteins pooled into one summary) at lambda=0.5. Everything the
    boost eventually sees is still the same graph-enriched protein vector; DGI
    only reshapes the encoder during training and is not part of what leaves
    the extractor. Override `dgi_weight`/`dgi_scope` to sweep."""
    dgi_weight: float = 0.5
    dgi_scope: str = "shared"
    model_name: str = field(init=False, default="gnn_signed_dgi")
