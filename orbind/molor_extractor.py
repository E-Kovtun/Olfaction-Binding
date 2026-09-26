"""Pair-level "cls"-style extractor for orbind.ensemble.run_ensemble:
MolorExtractor -- a reimplementation of MolOR (the cross-attention odorant-
receptor model from the `olfaction` repo, `gcn_or_predictor.py::MolORPredictor`),
in the same spirit as prosmith_extractor.py and lorax_extractor.py: the model
definition is inline and directly editable, and only this project's own generic
plumbing (Hladis/M2OR sample weights, best-val + checkpoint-resume protocol,
n_models bagging, ensemble wiring) is reused.

What upstream MolOR does (olfaction/gcn_or_predictor.py::MolORPredictor)
-----------------------------------------------------------------------
The molecule is a 2D graph passed through a GCN (dgllife) to per-atom features;
the protein is a per-residue ESM matrix. A cross-attention block
(`OdorantReceptorCrossAttention`, mol2prot=False) collapses the protein to the
GNN width and produces one pooled protein vector and one pooled molecule vector,
concatenated into a joint representation that a small MLP maps to a logit. Its
canonical M2OR config (data/configures/M2OR_Pairs/MolOR_canonical.json): GCN with
gnn_hidden_feats=256, num_gnn_layers=2, residual, no batchnorm, dropout 0.05,
predictor_hidden_feats=128, lr 2e-2, batch 128.

This project's variant (one deliberate scope choice)
----------------------------------------------------
**Protein = frozen precomputed ESM-1b, per-residue.** Exactly like our ProSmith
and LORAX baselines, the protein encoder is not trained: we feed the same
per-residue ESM-1b matrices (`esm1b_650m_per_residue_full_full.npz`) straight in
as the cross-attention protein input. That is what makes MolOR directly
comparable to ProSmith and LORAX here -- all three sit on ESM-1b, so the delta
measures the architecture (GCN molecule encoder + this cross-attention), not a
protein-encoder swap. Upstream MolOR's own checkpoints use ESM-2 650M; pass a
different `protein_path` if you want that instead. The molecule GCN is trained
live end-to-end (that IS MolOR's identity -- a graph encoder, not ChemBERTa).

Feature handed to the boosting stage = the joint cross-attention representation
(`graph_feats` after the final LayerNorm, before the MLP head): the concatenation
of the pooled protein vector (prot_dim) and pooled molecule vector (gnn width),
the MolOR analogue of ProSmith's <cls> and LORAX's `cat_rep`.

dgl + dgllife (and rdkit, for featurization) are imported lazily -- only when a
model/graph is actually built -- so this module imports fine on a box that lacks
them.
"""
from __future__ import annotations

import pathlib
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import dataset as D
from .prosmith_extractor import _M2ORWeights   # identical Hladis/M2OR weighting
from .tasks import batch_loss_fn, check_task

_INIT_LOCK = threading.Lock()   # see _train_one (global manual_seed under threads)


# --------------------------------------------------------------------------- featurization

_FEAT_LOCK = threading.Lock()   # dgllife/rdkit graph construction is not thread-safe


def _smiles_to_graph_builder(add_self_loop: bool = True):
    """A `str -> DGLGraph | None` callable using dgllife's CanonicalAtomFeaturizer
    (74-d atom features under ndata key 'h'), matching MolOR's canonical setup.
    Lazy import so this module loads without dgl/dgllife/rdkit."""
    from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer
    featurizer = CanonicalAtomFeaturizer()
    to_graph = SMILESToBigraph(add_self_loop=add_self_loop, node_featurizer=featurizer)
    return to_graph, featurizer.feat_size()


# --------------------------------------------------------------------------- data plumbing

class _MolorDataset(Dataset):
    """(DGLGraph for the molecule, per-residue protein matrix, label, weight).

    Molecule graphs are prebuilt and cached by SMILES in the extractor, so
    __getitem__ is a dict lookup, not a re-featurization."""

    def __init__(self, pairs: pd.DataFrame, graphs: dict, proteins: dict, weights):
        self.pairs = pairs.reset_index(drop=True)
        self.graphs = graphs
        self.proteins = proteins
        self.weights = weights

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        return (self.graphs[str(row.smiles)],
                np.asarray(self.proteins[row.receptor], dtype=np.float32),
                np.float32(row.label),
                np.float32(1.0 if self.weights is None else self.weights[idx]))


def _make_collate(max_node_len: int):
    """Batch the molecule graphs (dgl.batch), pad per-residue protein matrices to
    the batch max (seq_mask: 1=real residue), and record the per-graph node count
    (node_mask, padded to the fixed global max_node_len: 1=real atom)."""
    import dgl

    def collate(batch):
        graphs, prots, labels, weights = zip(*batch)
        bg = dgl.batch(graphs)

        lengths = [len(p) for p in prots]
        width = max(lengths)
        prot_dim = prots[0].shape[-1]
        prot = torch.zeros(len(prots), width, prot_dim, dtype=torch.float32)
        seq_mask = torch.zeros(len(prots), width, dtype=torch.float32)
        for i, p in enumerate(prots):
            n = len(p)
            prot[i, :n] = torch.from_numpy(p)
            seq_mask[i, :n] = 1.0

        node_mask = torch.zeros(len(graphs), max_node_len, dtype=torch.float32)
        for i, g in enumerate(graphs):
            node_mask[i, :g.num_nodes()] = 1.0

        return (bg, prot, seq_mask, node_mask,
                torch.tensor(labels, dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32))
    return collate


def _make_loader(df, graphs, proteins, weights, max_node_len, batch_size, train):
    ds = _MolorDataset(df, graphs, proteins, weights)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      collate_fn=_make_collate(max_node_len))


@torch.inference_mode()
def _embed(model, loader, device):
    """joint graph_feats after the last LayerNorm -- MolOR's boosting feature."""
    model.eval()
    out = []
    for bg, prot, seq_mask, node_mask, _, _ in loader:
        _, rep = model(bg.to(device), prot.to(device), seq_mask.to(device),
                       node_mask.to(device), get_repr=True)
        out.append(rep.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- model

class _CrossAttention(nn.Module):
    """MolOR's OdorantReceptorCrossAttention (mol2prot=False): protein (D1) and
    molecule-node (D2) tensors are linearly mapped to D2, cross-attended with a
    ReLU-scored scaled dot product, aggregated to per-token weights, masked, and
    used to pool each modality to a single vector. Output = [protein_vec(D1) ||
    mol_vec(D2)]. Copied inline from gcn_or_predictor.py so it stays editable."""

    def __init__(self, D1: int, D2: int):
        super().__init__()
        self.q1 = nn.Linear(D1, D2); self.k1 = nn.Linear(D1, D2); self.v1 = nn.Linear(D1, D2)
        self.q2 = nn.Linear(D2, D2); self.k2 = nn.Linear(D2, D2); self.v2 = nn.Linear(D2, D2)
        self.lin1 = nn.Linear(D2, 1)
        self.lin2 = nn.Linear(D2, 1)

    @staticmethod
    def _attend(query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(float(d_k)))
        return torch.matmul(torch.relu(scores), value)

    def forward(self, prot, mol, seq_mask, node_mask):
        # prot:[B,R,D1] mol:[B,A,D2]  seq_mask:[B,R] node_mask:[B,A] (1=real)
        prot = prot * seq_mask[:, :, None]
        q1, k1, v1 = self.q1(prot), self.k1(prot), self.v1(prot)
        q2, k2, v2 = self.q2(mol), self.k2(mol), self.v2(mol)

        attn_prot = self._attend(q1, k2, v2)                 # [B,R,D2]
        attn_mol = self._attend(q2, k1, v1)                  # [B,A,D2]

        w_prot = self.lin1(attn_prot).squeeze(-1)            # [B,R]
        w_mol = self.lin2(attn_mol).squeeze(-1)              # [B,A]
        w_prot = w_prot.masked_fill(seq_mask == 0, 0.0)
        w_mol = w_mol.masked_fill(node_mask == 0, 0.0)

        protein_vec = torch.einsum("bdr,br->bd", prot.transpose(1, 2), w_prot)      # [B,D1]
        mol_vec = torch.einsum("bda,ba->bd", mol.transpose(1, 2), w_mol)            # [B,D2]
        return torch.cat([protein_vec, mol_vec], dim=1)      # [B, D1+D2]


class _MolorMM(nn.Module):
    """MolOR: GCN molecule encoder (dgllife) + frozen per-residue protein matrix,
    joined by cross-attention, LayerNorm'd, mapped to a logit by an MLP.

    `get_repr=True` also returns the joint representation (the boosting feature)."""

    def __init__(self, in_feats: int, gnn_hidden: int, prot_dim: int,
                 num_gnn_layers: int, dropout: float, residual: bool, batchnorm: bool,
                 max_node_len: int, predictor_hidden: int, predictor_dropout: float):
        super().__init__()
        from dgllife.model.gnn.gcn import GCN
        from dgllife.model.model_zoo.mlp_predictor import MLPPredictor

        self.max_node_len = max_node_len
        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=[gnn_hidden] * num_gnn_layers,
                       gnn_norm=["none"] * num_gnn_layers,
                       activation=[torch.relu] * num_gnn_layers,
                       residual=[residual] * num_gnn_layers,
                       batchnorm=[batchnorm] * num_gnn_layers,
                       dropout=[dropout] * num_gnn_layers)
        gnn_out = self.gnn.hidden_feats[-1]

        self.cross_attn = _CrossAttention(prot_dim, gnn_out)
        self.prot_norm = nn.LayerNorm(prot_dim)
        self.mol_norm = nn.LayerNorm(gnn_out)
        self.feat_norm = nn.LayerNorm(prot_dim + gnn_out)
        self.predict = MLPPredictor(prot_dim + gnn_out, predictor_hidden, 1, predictor_dropout)
        self.repr_dim = prot_dim + gnn_out

    def forward(self, bg, prot, seq_mask, node_mask, get_repr: bool = False):
        import dgl
        node_feats = self.gnn(bg, bg.ndata["h"])
        graphs = dgl.unbatch(bg)
        batch_nodes = torch.zeros(len(graphs), self.max_node_len, node_feats.shape[1],
                                  device=node_feats.device)
        counter = 0
        for i, g in enumerate(graphs):
            n = g.num_nodes()
            batch_nodes[i, :n] = node_feats[counter:counter + n]
            counter += n

        prot = self.prot_norm(prot)
        batch_nodes = self.mol_norm(batch_nodes)
        graph_feats = self.cross_attn(prot, batch_nodes, seq_mask, node_mask)
        graph_feats = self.feat_norm(graph_feats)
        logit = self.predict(graph_feats).squeeze(-1)
        return (logit, graph_feats) if get_repr else logit


# --------------------------------------------------------------------------- training

def _train_one(build_model, train_df, val_df, test_df, graphs, proteins,
               w_train, w_val, w_test, hp, device, checkpoint_path=None):
    """Same protocol as prosmith/lorax _train_one: Adam over the model params,
    fixed epochs, no scheduler/early-stop, keep the best-val-loss weights.
    Weighted BCE on logits (Hladis/M2OR weights when present). If
    `checkpoint_path` exists, skip training and only embed."""
    val_loader = _make_loader(val_df, graphs, proteins, w_val, hp["max_node_len"],
                              hp["batch_size"], train=False)
    test_loader = _make_loader(test_df, graphs, proteins, w_test, hp["max_node_len"],
                               hp["batch_size"], train=False)

    with _INIT_LOCK:
        torch.manual_seed(hp["seed"])
        model = build_model().to(device)

    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return _embed(model, val_loader, device), _embed(model, test_loader, device), model

    train_loader = _make_loader(train_df, graphs, proteins, w_train, hp["max_node_len"],
                                hp["batch_size"], train=True)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    batch_loss = batch_loss_fn(hp["task"])   # BCE, or squared error under regression

    best_val, best_state = np.inf, None
    for ep in range(hp["epochs"]):
        model.train()
        run_loss, seen = 0.0, 0
        for bg, prot, seq_mask, node_mask, y, w in train_loader:
            if len(y) == 1:      # a size-1 batch breaks nothing here, but skip for parity
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(bg.to(device), prot.to(device), seq_mask.to(device),
                           node_mask.to(device))
            loss = batch_loss(logits, y.to(device), w.to(device))
            loss.backward()
            opt.step()
            run_loss += float(loss) * len(y); seen += len(y)

        model.eval()
        total, n = 0.0, 0
        with torch.inference_mode():
            for bg, prot, seq_mask, node_mask, y, w in val_loader:
                logits = model(bg.to(device), prot.to(device), seq_mask.to(device),
                               node_mask.to(device))
                total += float(batch_loss(logits, y.to(device), w.to(device))) * len(y)
                n += len(y)
        val_loss = total / max(n, 1)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    molor epoch {ep + 1}/{hp['epochs']}  "
              f"train_loss={run_loss / max(seen, 1):.4f}  val_loss={val_loss:.4f}"
              f"{'  *best' if is_best else ''}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return _embed(model, val_loader, device), _embed(model, test_loader, device), model


def _run_models(ext, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
    check_task(ext.task)
    ext._ensure_loaded()
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            raise KeyError(f"{ext.name}: {int((~mask).sum())}/{len(idx)} {split_name} rows "
                           f"missing a protein embedding or a valid molecule graph "
                           f"(protein={ext.protein_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = pairs.iloc[train_idx], pairs.iloc[val_idx], pairs.iloc[test_idx]

    w = _M2ORWeights.maybe_build(pairs)
    w_train = None if w is None else w.per_row[train_idx]
    w_val = None if w is None else w.per_row[val_idx]
    w_test = None if w is None else w.per_row[test_idx]

    def model_job(m):
        hp = ext._hp(seed + ext.seed_offset * m)
        checkpoint_path = (pathlib.Path(checkpoint_dir) / f"molor_{ext.name}_model{m}.pt"
                           if checkpoint_dir is not None else None)
        e_val, e_test, model = _train_one(
            ext._build_model, train_df, val_df, test_df, ext._graphs, ext._proteins,
            w_train, w_val, w_test, hp, device, checkpoint_path=checkpoint_path)
        train_loader = _make_loader(train_df, ext._graphs, ext._proteins, w_train,
                                    hp["max_node_len"], hp["batch_size"], train=False)
        return m, _embed(model, train_loader, device), e_val, e_test, model

    # The GCN is light, but a live molecule encoder per model is still non-trivial;
    # like LORAX we run the n_models jobs sequentially (n_models defaults to 1).
    results = sorted((model_job(m) for m in range(ext.n_models)), key=lambda r: r[0])

    if checkpoint_dir is not None:
        for m, _, _, _, model in results:
            path = pathlib.Path(checkpoint_dir) / f"molor_{ext.name}_model{m}.pt"
            if not path.exists():
                torch.save(model.state_dict(), path)

    return (np.concatenate([r[1] for r in results], axis=1).astype(np.float32),
            np.concatenate([r[2] for r in results], axis=1).astype(np.float32),
            np.concatenate([r[3] for r in results], axis=1).astype(np.float32))


@dataclass
class MolorExtractor:
    """MolOR cls-style source: a GCN molecule encoder (trained live) cross-attended
    with a frozen per-residue ESM-1b protein matrix. Defaults follow MolOR's M2OR
    canonical config (data/configures/M2OR_Pairs/MolOR_canonical.json): GCN with
    num_gnn_layers=2, gnn_hidden=256, residual, no batchnorm, dropout 0.05,
    predictor_hidden 128, lr 2e-2, weight_decay 0. Protein is frozen ESM-1b, for
    apples-to-apples comparability with the ProSmith and LORAX baselines (all on
    ESM-1b); MolOR's own checkpoints use ESM-2 -- pass that `protein_path` to match
    upstream. The boosting feature is the joint cross-attention representation
    (prot_dim + gnn_hidden = 1280 + 256 = 1536 per model).

    `protein_path` must be a *per-residue* npz keyed by sequence
    ({sequence: float32[L, prot_dim]}); default is ESM-1b.

    NOTE: needs `dgl`, `dgllife` and `rdkit` in the run environment -- imported
    lazily, so this class imports without them but building a model/graph does not.
    A GCN over canonical atom features expects 74-d node features; the actual width
    is read from the featurizer at load time."""

    name: str
    # ESM3 since 26.09.2026 (was esm1b_650m_per_residue_full_full.npz); M2OR only --
    # the insect datasets pass their own file in the source spec.
    protein_path: str = "data/embeddings/proteins/esm3_per_residue_m2or.npz"
    n_models: int = 1
    gnn_hidden: int = 256
    num_gnn_layers: int = 2
    dropout: float = 0.05
    residual: bool = True
    batchnorm: bool = False
    predictor_hidden: int = 128
    predictor_dropout: float = 0.0
    prot_dim: int = 1280           # ESM-1b per-residue width
    lr: float = 2e-2               # MolOR_canonical lr
    weight_decay: float = 0.0
    epochs: int = 30
    batch_size: int = 128          # MolOR_canonical batch_size
    add_self_loop: bool = True
    seed_offset: int = 7000
    # The task axis (orbind/tasks.py). `run_ensemble` overwrites this to match
    # the run, so this head's criterion and the boosting head downstream agree.
    task: str = "classification"
    pooling: str = "cross_attn_cat"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="molor_gcn")

    def __post_init__(self):
        self.dim_out = self.n_models * (self.prot_dim + self.gnn_hidden)
        self.path = self.protein_path
        self._proteins = None
        self._graphs = None          # {smiles: DGLGraph} for covered molecules
        self._in_feats = None
        self._max_node_len = None

    def _ensure_loaded(self):
        # per-residue npz is ~1.9 GB and --max-parallel pickles the extractor to
        # each worker, so it's loaded lazily (see ProSmith's/LORAX's same note).
        if self._proteins is None:
            self._proteins = D.load_npz_dict(self.protein_path)

    def _ensure_graphs(self, pairs):
        """Build (once) a SMILES->DGLGraph cache over every molecule in `pairs`,
        and the global max node count needed to pad the cross-attention batch."""
        if self._graphs is not None:
            return
        with _FEAT_LOCK:
            to_graph, in_feats = _smiles_to_graph_builder(self.add_self_loop)
            graphs = {}
            for s in pairs["smiles"].astype(str).unique():
                g = to_graph(s)
                if g is not None and g.num_nodes() > 0:
                    graphs[s] = g
        self._graphs = graphs
        self._in_feats = in_feats
        self._max_node_len = max((g.num_nodes() for g in graphs.values()), default=1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_proteins"] = None
        state["_graphs"] = None       # rebuilt per worker; DGL graphs don't pickle cheaply
        return state

    def _build_model(self):
        prot_dim = next(iter(self._proteins.values())).shape[-1]
        if prot_dim != self.prot_dim:
            self.prot_dim = prot_dim
            self.dim_out = self.n_models * (self.prot_dim + self.gnn_hidden)
        return _MolorMM(self._in_feats, self.gnn_hidden, prot_dim, self.num_gnn_layers,
                        self.dropout, self.residual, self.batchnorm, self._max_node_len,
                        self.predictor_hidden, self.predictor_dropout)

    def _hp(self, seed):
        return dict(lr=self.lr, weight_decay=self.weight_decay, epochs=self.epochs,
                    batch_size=self.batch_size, max_node_len=self._max_node_len,
                    task=self.task, seed=seed)

    def covered(self, pairs, idx):
        self._ensure_loaded()
        self._ensure_graphs(pairs)
        prot = pairs["receptor"].to_numpy()[idx]
        smi = pairs["smiles"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and (str(s) in self._graphs)
                            for p, s in zip(prot, smi)), dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        self._ensure_graphs(pairs)
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed,
                           checkpoint_dir=checkpoint_dir)
