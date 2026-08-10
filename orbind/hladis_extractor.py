"""Standalone pair-level "cls" extractor for orbind.ensemble.run_ensemble:
HladisExtractor -- a PyTorch reimplementation of Hladis et al. (ICLR 2023),
"Matching receptor to odorant with protein language and graph neural
networks" (MatejHl/Receptor2Odorant, model `normal_QK_model`).

Written fresh here in the same spirit as prosmith_extractor.py,
lorax_extractor.py and molor_extractor.py: the model is defined inline and
directly editable, and only this project's own generic plumbing is reused
(Hladis `_M2ORWeights`, best-val + checkpoint resume, n_models bagging, the
ensemble contract).

A full rewrite, not a port: upstream is JAX/Flax/jraph (43 jax, 22 flax, 13
jraph imports), so no weights transfer and nothing can be imported. What
follows is a re-derivation from Receptor2Odorant/main/model/normal_QK.py and
its layers.

Architecture (upstream `normal_QK_model`)
-----------------------------------------
1. Protein vector -> Dense(256) -> ReLU -> Dense(node_d_model) -> LayerNorm.
2. Atom features: AtomicNum (Embed 36, index = Z-1), ChiralTag (Embed 4) and
   Hybridization (Embed 8) get learned embeddings; the remaining five
   (FormalCharge, NumImplicitHs, ExplicitValence, Mass, IsAromatic) are
   concatenated **with the protein vector broadcast onto every atom** and
   projected. The embeddings and that projection are summed.

   That broadcast is the whole hybrid coupling. Unlike ProSmith, LORAX and
   MolOR, this model has no cross-attention block at all -- the receptor
   enters as a per-atom bias and everything after it is molecule-graph
   machinery conditioned on the receptor.
3. Bond features: BondType (Embed 22) and Stereo (Embed 6) embedded,
   IsAromatic projected, summed.
4. Five `GraphProcessingEncoderLayer`s. Each runs three message-passing
   networks to produce Q, K and V node states, applies ordinary multi-head
   dot-product attention over the atoms *within* each molecule, then
   residual + LayerNorm, a widening-factor-8 feed-forward block, and
   residual + LayerNorm again. Note upstream passes the **same module
   instance** for K and V (`_mpnn_V = self.mha_mpnn_K`), so they share
   weights but are invoked separately -- see `_MessagePassing` on why that
   still makes them differ during training.
5. An edge-conditioned convolution (ECC), then attention-sum pooling over
   atoms, giving one vector per molecule.
6. Dropout(0.5) -> Dense(1) -> logit.

The feature handed to the boosting stage is the pooled graph vector from
step 5, before dropout and the output layer -- the analogue of ProSmith's
`<cls>`, LORAX's `cat_rep` and MolOR's joint representation. Its width is
`node_d_model`, **72 by default**: an order of magnitude narrower than the
other three cls sources (768 / 1664 / 1536). That is upstream's own size,
not a truncation, but it is worth remembering when reading the comparison --
raise `node_d_model` to widen it, at the cost of no longer being Hladis.

Protein encoder: a deliberate substitution
------------------------------------------
Upstream feeds a precomputed **ProtBERT CLS** vector
(`config_train.yml: BERT_H5FILE .../PrecomputeProtBERT_CLS/ProtBERT_CLS.h5`),
one vector per receptor. The default here is instead frozen **mean-pooled
ESM-1b** (`esm1b_650m_mean.npz`), for the same reason the LORAX extractor
fixes its protein side to ESM-1b: it puts Hladis, ProSmith, LORAX and MolOR
on one protein encoder so the comparison isolates the architecture rather
than confounding it with an encoder swap. Structurally it is the right
substitute -- both are a single frozen vector per receptor, not per-residue.
Point `protein_path` at a ProtBERT npz to run upstream's own input.

Relationship to LORAX
---------------------
LORAX does **not** implement or run this model. Its repo touches Hladis in
exactly two places: `data_utils/convert_m2or_dat.py`, which consumes Hladis's
own rand-split 5-fold M2OR data, and `loss_functions.py`, which reuses his
data-quality weighting (the same weights `_M2ORWeights` reproduces). The
Hladis row in that paper is quoted from the original, not reproduced -- so
there is no reference implementation to validate against, and this module is
the only Hladis in the project.

The upside is that the *split* is shared: LoRaX's `rand_split_1..5` are
Hladis's own folds, which is exactly the `full_full` pool this project's
transductive regime uses. Published Hladis numbers are therefore on the same
partition as our transductive runs. (`inductive_molecule_v5` is ours, and has
no published Hladis counterpart.)
"""
from __future__ import annotations

import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import dataset as D
from .prosmith_extractor import _M2ORWeights

# Upstream's config_train.yml ATOM_FEATURES / BOND_FEATURES, in order.
ATOM_FEATURES = ("AtomicNum", "ChiralTag", "Hybridization", "FormalCharge",
                 "NumImplicitHs", "ExplicitValence", "Mass", "IsAromatic")
BOND_FEATURES = ("BondType", "Stereo", "IsAromatic")
# Embedded (categorical) atom/bond features -> (position in the tuple, vocabulary).
# Vocabulary sizes are upstream's, from the rdkit enums; AtomicNum is shifted by
# one because atomic numbers start at 1.
# The third element is the index shift upstream applies before the lookup:
# atomic numbers start at 1, so AtomicNumEmbedding does `X - 1`.
ATOM_EMBEDS = {"AtomicNum": (0, 36, 1), "ChiralTag": (1, 4, 0), "Hybridization": (2, 8, 0)}
BOND_EMBEDS = {"BondType": (0, 22, 0), "Stereo": (1, 6, 0)}
ATOM_SCALAR_POS = [3, 4, 5, 6, 7]     # FormalCharge, NumImplicitHs, ExplicitValence, Mass, IsAromatic
BOND_SCALAR_POS = [2]                 # IsAromatic

_INIT_LOCK = threading.Lock()


# --------------------------------------------------------------------------- molecule graphs

def build_graph(smiles: str):
    """One molecule -> (atom features [N, 8], edge index [2, E], edge features
    [E, 3]), matching Receptor2Odorant/mol2graph/jraph/convert.py: rdkit's own
    integer enum values (not one-hot), and every bond emitted in both
    directions with its features duplicated."""
    from rdkit import Chem                      # lazy: only a Hladis run needs rdkit

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"rdkit could not parse SMILES {smiles!r}")

    atoms = [[a.GetAtomicNum(), int(a.GetChiralTag()), int(a.GetHybridization()),
              a.GetFormalCharge(), a.GetNumImplicitHs(), a.GetExplicitValence(),
              a.GetMass(), int(a.GetIsAromatic())] for a in mol.GetAtoms()]

    begin, end, feats = [], [], []
    for b in mol.GetBonds():
        begin.append(b.GetBeginAtomIdx())
        end.append(b.GetEndAtomIdx())
        feats.append([int(b.GetBondType()), int(b.GetStereo()), int(b.GetIsAromatic())])
    if not begin:
        # Upstream raises NoBondsError here; a single-atom molecule has nothing
        # to message-pass over.
        raise ValueError(f"molecule with no bonds: {smiles!r}")

    edge_index = np.array([begin + end, end + begin], dtype=np.int64)
    return (np.array(atoms, dtype=np.float32),
            edge_index,
            np.array(feats + feats, dtype=np.float32))


class _PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, proteins: dict, graphs: dict, weights):
        self.pairs = pairs.reset_index(drop=True)
        self.proteins = proteins
        self.graphs = graphs
        self.weights = weights

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        x, ei, ea = self.graphs[row.inchikey]
        return (x, ei, ea,
                np.asarray(self.proteins[row.receptor], dtype=np.float32),
                np.float32(row.label),
                np.float32(1.0 if self.weights is None else self.weights[idx]))


def _collate(batch):
    """Pads atoms to the batch maximum and offsets every edge into the flat
    (B*N) node layout, so message passing can run as flat scatter-adds while
    attention still sees a dense (B, N, d) tensor -- the same two views
    upstream moves between via `jnp.reshape`."""
    xs, eis, eas, prots, labels, weights = zip(*batch)
    b = len(xs)
    n_max = max(len(x) for x in xs)
    x = torch.zeros(b, n_max, xs[0].shape[1])
    node_mask = torch.zeros(b, n_max, dtype=torch.bool)     # True = real atom
    src, dst, ea_all = [], [], []
    for i, (xi, ei, ea) in enumerate(zip(xs, eis, eas)):
        n = len(xi)
        x[i, :n] = torch.from_numpy(xi)
        node_mask[i, :n] = True
        src.append(torch.from_numpy(ei[0]) + i * n_max)
        dst.append(torch.from_numpy(ei[1]) + i * n_max)
        ea_all.append(torch.from_numpy(ea))
    return (x, node_mask, torch.cat(src), torch.cat(dst), torch.cat(ea_all),
            torch.from_numpy(np.stack(prots)),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32))


def _make_loader(df, proteins, graphs, weights, batch_size, train):
    return DataLoader(_PairDataset(df, proteins, graphs, weights),
                      batch_size=batch_size, shuffle=train, collate_fn=_collate)


# --------------------------------------------------------------------------- model

class _MessagePassing(nn.Module):
    """Upstream's `BasicTruncatedNormalDynamicMessagePassing`: one set of
    shared weights applied a *random number of times*.

    Each step updates edges then nodes, and the updated edges feed the next
    step:
        edge  <- ReLU(Linear([node_src, node_dst, edge]))
        node  <- GRUCell(carry=node, input=sum of edges arriving at that node)

    The step count is drawn per call from a truncated normal (mean 6, std 1,
    clipped to [3, 9]) during training and fixed at 6 in eval. Because it is
    drawn *per call*, the K and V invocations differ during training even
    though upstream hands them the same module instance -- that stochastic
    difference is the only thing separating K from V, and dropping it would
    make V an exact copy of K.

    Upstream always unrolls the maximum 9 steps and then indexes the state it
    wants, purely so the whole thing stays jit-able; running the sampled count
    directly is equivalent and cheaper.
    """

    def __init__(self, node_dim: int, edge_dim: int, lo: int = 3, hi: int = 9,
                 mean: float = 6.0, std: float = 1.0):
        super().__init__()
        self.edge_mlp = nn.Linear(2 * node_dim + edge_dim, edge_dim)
        self.gru = nn.GRUCell(edge_dim, node_dim)
        self.lo, self.hi, self.mean, self.std = lo, hi, mean, std

    def n_steps(self) -> int:
        if not self.training:
            return int(round(self.mean))
        lo, hi = (self.lo - self.mean) / self.std, (self.hi - self.mean) / self.std
        while True:                                  # rejection sampling, as jax does
            z = torch.randn(()).item()
            if lo <= z <= hi:
                return int(round(self.std * z + self.mean))

    def forward(self, nodes, edges, src, dst):
        for _ in range(self.n_steps()):
            edges = torch.relu(self.edge_mlp(torch.cat([nodes[src], nodes[dst], edges], dim=-1)))
            agg = torch.zeros(nodes.shape[0], edges.shape[-1],
                              device=nodes.device, dtype=nodes.dtype)
            agg = agg.index_add(0, dst, edges)       # messages arriving at each node
            nodes = self.gru(agg, nodes)
        return nodes, edges


class _EncoderLayer(nn.Module):
    """Upstream's `GraphProcessingEncoderLayer`: message-passing-derived Q/K/V,
    multi-head attention over the atoms of one molecule, then the usual
    transformer residual/LayerNorm/FFN sandwich."""

    def __init__(self, node_dim: int, edge_dim: int, n_heads: int,
                 dropout: float, widening: int):
        super().__init__()
        self.mp_q = _MessagePassing(node_dim, edge_dim)
        self.mp_kv = _MessagePassing(node_dim, edge_dim)      # upstream shares K and V
        self.attn = nn.MultiheadAttention(node_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(node_dim)
        self.ffn = nn.Sequential(nn.Linear(node_dim, widening * node_dim), nn.ReLU(),
                                 nn.Linear(widening * node_dim, node_dim))
        self.norm_ffn = nn.LayerNorm(node_dim)

    def forward(self, nodes, edges, src, dst, node_mask):
        b, n = node_mask.shape
        d = nodes.shape[-1]
        q, _ = self.mp_q(nodes, edges, src, dst)
        k, _ = self.mp_kv(nodes, edges, src, dst)
        v, _ = self.mp_kv(nodes, edges, src, dst)             # separate call: see _MessagePassing

        pad = ~node_mask.reshape(b, n)
        h, _ = self.attn(q.view(b, n, d), k.view(b, n, d), v.view(b, n, d),
                         key_padding_mask=pad, need_weights=False)
        h = (h * node_mask.unsqueeze(-1).float()).reshape(-1, d)

        h = self.norm_attn(h + nodes)
        return self.norm_ffn(h + self.ffn(h))


class _HladisNet(nn.Module):
    """`normal_QK_model`. Returns (logit, pooled) where `pooled` is the
    attention-sum-pooled graph vector -- the cls analogue."""

    def __init__(self, prot_dim: int, node_dim: int = 72, edge_dim: int = 36,
                 n_layers: int = 5, n_heads: int = 6, dropout: float = 0.1,
                 widening: int = 8, out_dropout: float = 0.5):
        super().__init__()
        self.node_dim = node_dim
        self.or_mlp = nn.Sequential(nn.Linear(prot_dim, 256), nn.ReLU(), nn.Linear(256, node_dim))
        self.or_norm = nn.LayerNorm(node_dim)

        self.atom_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, node_dim) for name, (_, vocab, _s) in ATOM_EMBEDS.items()})
        self.bond_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, edge_dim) for name, (_, vocab, _s) in BOND_EMBEDS.items()})
        # The protein rides along with the non-embedded atom features.
        self.atom_proj = nn.Linear(len(ATOM_SCALAR_POS) + node_dim, node_dim)
        self.bond_proj = nn.Linear(len(BOND_SCALAR_POS), edge_dim)

        self.layers = nn.ModuleList([
            _EncoderLayer(node_dim, edge_dim, n_heads, dropout, widening) for _ in range(n_layers)])

        # ECC: the edge MLP emits a (node_dim x node_dim) matrix per edge.
        self.ecc_mlp = nn.Linear(edge_dim, node_dim * node_dim, bias=False)
        self.ecc_root = nn.Linear(node_dim, node_dim, bias=True)
        self.pool_logits = nn.Linear(node_dim, 1, bias=False)
        self.dropout = nn.Dropout(out_dropout)
        self.out = nn.Linear(node_dim, 1)

    def forward(self, x, node_mask, src, dst, edge_attr, prot):
        b, n, _ = x.shape
        flat_mask = node_mask.reshape(-1, 1).float()

        s = self.or_norm(self.or_mlp(prot))                       # (B, node_dim)
        s_nodes = s.unsqueeze(1).expand(b, n, self.node_dim).reshape(-1, self.node_dim)

        xf = x.reshape(-1, x.shape[-1])
        nodes = torch.zeros(xf.shape[0], self.node_dim, device=x.device)
        for name, (pos, vocab, shift) in ATOM_EMBEDS.items():
            idx = (xf[:, pos].long() - shift).clamp_(0, vocab - 1)
            nodes = nodes + self.atom_embeds[name](idx)
        nodes = nodes + self.atom_proj(torch.cat([xf[:, ATOM_SCALAR_POS], s_nodes], dim=-1))
        nodes = nodes * flat_mask                                  # zero the padding atoms back out

        edges = torch.zeros(edge_attr.shape[0], self.bond_proj.out_features, device=x.device)
        for name, (pos, vocab, shift) in BOND_EMBEDS.items():
            idx = (edge_attr[:, pos].long() - shift).clamp_(0, vocab - 1)
            edges = edges + self.bond_embeds[name](idx)
        edges = edges + self.bond_proj(edge_attr[:, BOND_SCALAR_POS])

        for layer in self.layers:
            nodes = layer(nodes, edges, src, dst, node_mask)

        # ECC. Upstream aggregates by SENDER here (its update_node_fn takes
        # sent_attributes), unlike the message-passing steps above which
        # aggregate by receiver.
        w = self.ecc_mlp(edges).view(-1, self.node_dim, self.node_dim)
        msg = torch.einsum("ij,ijk->ik", nodes[src], w)
        agg = torch.zeros_like(nodes).index_add(0, src, msg)
        nodes = agg + self.ecc_root(nodes)

        # Attention-sum pooling over the real atoms of each molecule.
        logits = self.pool_logits(nodes).view(b, n)
        logits = logits.masked_fill(~node_mask, -torch.inf)
        a = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (a * nodes.view(b, n, self.node_dim)).sum(dim=1)

        return self.out(self.dropout(pooled)).squeeze(-1), pooled


# --------------------------------------------------------------------------- training

def _transformer_lr(step: int, init: float, warmup: int) -> float:
    """Upstream's `transformer_schedule` (schedulers.py): the Vaswani warmup,
    `init * min(step^-0.5, step * warmup^-1.5)`. Note this makes `init` a
    scale, not a peak -- at warmup=6000 the actual peak is init/sqrt(6000),
    so the nominal 1e-3 tops out near 1.3e-5."""
    step = max(step, 1)
    return init * min(step ** -0.5, step * warmup ** -1.5)


@torch.inference_mode()
def _embed(model, loader, device):
    model.eval()
    out = []
    for x, mask, src, dst, ea, prot, _, _ in loader:
        _, pooled = model(x.to(device), mask.to(device), src.to(device), dst.to(device),
                          ea.to(device), prot.to(device))
        out.append(pooled.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _train_one(build_model, train_df, val_df, test_df, proteins, graphs,
               w_train, w_val, w_test, hp, device, checkpoint_path=None):
    """Upstream's protocol: Adam on the transformer warmup schedule, a fixed
    epoch count, no early stopping, keeping the best-validation-loss weights.
    The criterion is the Hladis-weighted BCE (his own weighting, which this
    project already reproduces for ProSmith and LORAX) -- upstream's
    `LOSS_OPTION: cross_entropy` with `WEIGHT_COL: sample_weight`.

    If `checkpoint_path` exists, training is skipped and only the embed passes
    run, so a resumed run reuses the trained model."""
    val_loader = _make_loader(val_df, proteins, graphs, w_val, hp["batch_size"], train=False)
    test_loader = _make_loader(test_df, proteins, graphs, w_test, hp["batch_size"], train=False)

    with _INIT_LOCK:
        torch.manual_seed(hp["seed"])
        model = build_model().to(device)

    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return _embed(model, val_loader, device), _embed(model, test_loader, device), model

    train_loader = _make_loader(train_df, proteins, graphs, w_train, hp["batch_size"], train=True)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    def loss_of(logits, y, w):
        return nn.functional.binary_cross_entropy_with_logits(logits, y, weight=w, reduction="mean")

    best_val, best_state, step = np.inf, None, 0
    for epoch in range(hp["epochs"]):
        model.train()
        for x, mask, src, dst, ea, prot, y, w in train_loader:
            step += 1
            for g in opt.param_groups:
                g["lr"] = _transformer_lr(step, hp["lr"], hp["warmup_steps"])
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x.to(device), mask.to(device), src.to(device), dst.to(device),
                              ea.to(device), prot.to(device))
            loss_of(logits, y.to(device), w.to(device)).backward()
            opt.step()

        model.eval()
        total, n = 0.0, 0
        with torch.inference_mode():
            for x, mask, src, dst, ea, prot, y, w in val_loader:
                logits, _ = model(x.to(device), mask.to(device), src.to(device), dst.to(device),
                                  ea.to(device), prot.to(device))
                total += float(loss_of(logits, y.to(device), w.to(device))) * len(y)
                n += len(y)
        val_loss = total / max(n, 1)
        print(f"    hladis epoch {epoch + 1}/{hp['epochs']}: val loss {val_loss:.4f}"
              + ("  *" if val_loss < best_val else ""), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return _embed(model, val_loader, device), _embed(model, test_loader, device), model


def _run_models(ext, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
    ext._ensure_loaded(pairs)
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            raise KeyError(
                f"{ext.name}: {int((~mask).sum())}/{len(idx)} {split_name} rows missing a protein "
                f"embedding or a buildable molecule graph (protein={ext.protein_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = pairs.iloc[train_idx], pairs.iloc[val_idx], pairs.iloc[test_idx]

    w = _M2ORWeights.maybe_build(pairs)
    w_train, w_val, w_test = (None, None, None) if w is None else \
        (w.per_row[train_idx], w.per_row[val_idx], w.per_row[test_idx])

    def model_job(m):
        hp = ext._hp(seed + ext.seed_offset * m)
        ckpt = (pathlib.Path(checkpoint_dir) / f"hladis_{ext.name}_model{m}.pt"
                if checkpoint_dir is not None else None)
        e_val, e_test, model = _train_one(ext._build_model, train_df, val_df, test_df,
                                          ext._proteins, ext._graphs,
                                          w_train, w_val, w_test, hp, device, checkpoint_path=ckpt)
        train_loader = _make_loader(train_df, ext._proteins, ext._graphs, w_train,
                                    hp["batch_size"], train=False)
        return m, _embed(model, train_loader, device), e_val, e_test, model

    with ThreadPoolExecutor(max_workers=ext.n_models) as pool:
        results = sorted((f.result() for f in [pool.submit(model_job, m) for m in range(ext.n_models)]),
                         key=lambda r: r[0])

    if checkpoint_dir is not None:
        for m, _, _, _, model in results:
            path = pathlib.Path(checkpoint_dir) / f"hladis_{ext.name}_model{m}.pt"
            if not path.exists():
                torch.save(model.state_dict(), path)

    return (np.concatenate([r[1] for r in results], axis=1).astype(np.float32),
            np.concatenate([r[2] for r in results], axis=1).astype(np.float32),
            np.concatenate([r[3] for r in results], axis=1).astype(np.float32))


@dataclass
class HladisExtractor:
    """Hladis et al. (ICLR 2023) cls source. Defaults follow upstream's
    `configs/config_train.yml` and `normal_QK_model`: node width 72, edge
    width 36, 5 encoder layers, 6 heads, widening 8, dropout 0.1 (0.5 before
    the head), Adam at 1e-3 on the transformer schedule with 6000 warmup
    steps, 10 epochs, batch 100.

    `protein_path` is a *mean-pooled* (one vector per receptor) npz keyed by
    sequence -- ESM-1b by default, which is a deliberate substitution for
    upstream's ProtBERT CLS; see the module docstring.

    Needs rdkit to build molecule graphs (imported lazily, only when a Hladis
    source is actually used). Molecule graphs are built once per extractor and
    cached, so the cost is 596 rdkit parses per run.

    The emitted feature is `node_d_model` wide -- 72 by default, much narrower
    than the other cls baselines.

    **Watch the learning rate.** Upstream's schedule is the Vaswani warmup with
    `warmup_steps=6000`, which makes `lr` a scale rather than a peak: the
    maximum it ever reaches is `lr / sqrt(6000)` ~ 1.3e-5, and only at step
    6000. On this pool 10 epochs at batch 100 is only ~3500 steps, so training
    ends *inside* the warmup ramp with an effective rate around 7e-6 -- the
    smoke test shows the loss barely moving over two epochs. That is faithfully
    upstream's config (`config_train.yml` + `make_create_optimizer.py`; the
    `TRANSITION_EPOCHS: 500` in that file belongs to the other optimizer branch
    and is dead for `adam_transformer`), and it is left as the default rather
    than quietly "fixed". If a run produces a near-untrained model, the honest
    levers are `warmup_steps` (lower it so the peak is reached) or `epochs`,
    and either one should be reported as a deviation."""

    name: str
    protein_path: str = "data/embeddings/proteins/esm1b_650m_mean.npz"
    node_d_model: int = 72
    edge_d_model: int = 36
    n_layers: int = 5
    n_heads: int = 6
    dropout: float = 0.1
    out_dropout: float = 0.5
    widening_factor: int = 8
    lr: float = 1e-3
    warmup_steps: int = 6000
    epochs: int = 10
    batch_size: int = 100
    n_models: int = 1
    seed_offset: int = 5000
    pooling: str = "attn_sum_pool"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="hladis_normal_qk")

    def __post_init__(self):
        self.dim_out = self.n_models * self.node_d_model
        self.path = f"{self.protein_path} + rdkit graphs"
        self._proteins = None
        self._graphs = None

    def _ensure_loaded(self, pairs: pd.DataFrame | None = None):
        """Loaded lazily and dropped from the pickle (see `__getstate__`): the
        graph cache is rebuilt per worker rather than shipped through the
        spawn pipe."""
        if self._proteins is None:
            self._proteins = D.load_npz_dict(self.protein_path)
        if self._graphs is None:
            if pairs is None:
                raise RuntimeError(f"{self.name}: molecule graphs need `pairs` to build from")
            if "smiles" not in pairs.columns:
                raise KeyError(f"{self.name}: pairs has no 'smiles' column; the Hladis source "
                               f"builds molecule graphs from SMILES with rdkit")
            uniq = pairs[["inchikey", "smiles"]].drop_duplicates("inchikey")
            self._graphs = {k: build_graph(s) for k, s in zip(uniq["inchikey"], uniq["smiles"])}
            print(f"  {self.name}: built {len(self._graphs)} molecule graphs", flush=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_proteins"] = None
        state["_graphs"] = None
        return state

    def _build_model(self):
        prot_dim = next(iter(self._proteins.values())).shape[-1]
        return _HladisNet(prot_dim, self.node_d_model, self.edge_d_model, self.n_layers,
                          self.n_heads, self.dropout, self.widening_factor, self.out_dropout)

    def _hp(self, seed):
        return dict(lr=self.lr, warmup_steps=self.warmup_steps, epochs=self.epochs,
                    batch_size=self.batch_size, seed=seed)

    def covered(self, pairs, idx):
        self._ensure_loaded(pairs)
        prot = pairs["receptor"].to_numpy()[idx]
        mol = pairs["inchikey"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and (m in self._graphs) for p, m in zip(prot, mol)),
                           dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed,
                           checkpoint_dir=checkpoint_dir)
