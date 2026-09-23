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

v8, the alpha gate (opt-in, `alpha=None` by default and then nothing below
applies). The receptor readout becomes a FIXED convex mix of a frozen
structural branch and the trained graph:

    z_prot = (1 - alpha) * frozen_SVD(ESM, train basis) + alpha * graph(...)

Why it was added. Without it the only path from ESM to the receptor embedding
runs through trainable weights, and at the titular 900 epochs the binding loss
empties it: the refined receptor cloud was measured sitting AT its permutation
null against ESM on CC (z = +0.4) while scoring z = +11 against the response
profile -- in every cell of a two-axis ablation over ESM rank and kept MP
edges. Both of those axes could only REMOVE information; neither could pull
back toward structure, because no term in the objective ever pulled that way,
and the structural end was reachable only by not training at all. A frozen
branch is a path the optimizer cannot drain, which turns "structure vs
function" from two ablations into one dial with two known ends: alpha=0 is
ESM's own geometry (exactly, on the train span -- see `_structural_anchor` on
why the projection is uncentered) and alpha=1 is the historical model up to a
global scale. Set once before training and used unchanged at inference.

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
                      (see legacy/scripts/modeling/train/train_graph_full_full.py's `probe_with`), never
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
(legacy/scripts/modeling/train/run_graph_full_full_v5.ps1: lr=3e-3, epochs=900,
`--probe-checkpoint last`, no `--lr-scheduler`) rather than this project's
earlier anti-collapse fix (lr=1e-3 + grad-clip + ReduceLROnPlateau +
best-val checkpoint selection, see notes/ and orbind/hetero.py history) --
fixed epoch count, no early stopping, no LR scheduler, last epoch's weights
are always what gets kept. Grad-clip (1.0) is kept from the anti-collapse
fix since v5's own command also passes `--grad-clip 1.0`.
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
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, MessagePassing, SAGEConv

from . import dataset as D
from . import mol_selection
from .tasks import check_task

# torch.manual_seed is GLOBAL while the n_models bags train as concurrent
# threads, so seeding must not interleave with another bag's build. Same guard
# lorax/molor use, and for the same reason.
_INIT_LOCK = threading.Lock()

MOL, PROT = "mol", "prot"
ETYPE = (MOL, "binds", PROT)
RTYPE = (PROT, "rev_binds", MOL)
ETYPE_NEG = (MOL, "no_binds", PROT)
RTYPE_NEG = (PROT, "rev_no_binds", MOL)


# --------------------------------------------------------------------------- graph plumbing (no model logic)


# --------------------------------------------------------------------------- v9 dial
# The v9 dial moves the receptor NODE FEATURES between the two graphs we already know,
# instead of gating the graph's OUTPUT the way v8 does:
#
#     x_prot(rho) = mu + rho * centred(ESM) + (1 - rho) * centred(identity vectors)
#
#     rho = 1   the node features ARE the embedding file -- the legacy graph, exactly
#     rho = 0   one fixed random vector per receptor, near-orthogonal to every other:
#               identity and nothing else, which is a one-hot in all but coordinates
#
# Both ends were already reachable (`onehot_nodes` and the plain extractor); what was
# missing was the road between them, because "half a one-hot" has no meaning as a
# discrete object. Random near-orthogonal vectors give it one: they live in the same
# space as ESM, so the two can be mixed continuously, and in 1280 dimensions n random
# directions are mutually near-perpendicular (expected |cos| ~ 1/sqrt(d) = 0.028), which
# is the property that makes them carry identity and no similarity structure.


def _identity_vectors(receptors, dim, seed=0):
    """One fixed unit vector per receptor, drawn once and near-orthogonal to the rest.

    Keyed by the receptor NAME, not by its position: the same receptor must get the same
    vector in every fold, every worker process and every dataset ordering, or "identity"
    silently means something different from one cell of a sweep to the next. Python's
    own `hash` is salted per process and cannot be used for this.

    `seed` shifts the whole draw. It is deliberately NOT the model seed -- the identity
    of a receptor is a property of the experiment, not of a training run -- but sweeping
    it is the honest robustness check that no conclusion rests on one lucky draw.
    """
    import hashlib
    X = np.empty((len(receptors), int(dim)), dtype=np.float64)
    for i, r in enumerate(receptors):
        h = hashlib.blake2b(str(r).encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(h, "big") ^ (int(seed) & 0xFFFFFFFF))
        X[i] = rng.standard_normal(int(dim))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _centred_rms(A):
    """Root-mean-square of a receptor cloud about its own centre, per receptor."""
    return float(np.sqrt((A ** 2).sum(1).mean())) if len(A) else 0.0


def mix_protein_features(proteins, train_receptors, rho, seed=0, renorm=True):
    """The v9 node features at dial position `rho`. Returns a new {receptor: vector}.

    THREE decisions, and the first is the one that decides whether the dial is a dial or
    a step function:

    1. MATCH THE CENTRED SPREAD, NOT THE TOTAL ENERGY. Mean-pooled ESM is about 94% a
       vector every receptor shares -- the between-receptor share of its energy is 5.6%
       on CC and 7.2% on HC. A random vector has no common part at all: every bit of it
       separates receptors. Scaling the two clouds to equal NORM therefore hands the
       random end an order of magnitude more usable signal, and the mixture stops being
       ESM almost immediately. v8 measured exactly that: with total-energy matching the
       crossover sat at alpha ~ 0.05 and the knob was a step. So both sides are centred
       on the TRAIN receptors and the random side is rescaled to the train ESM's centred
       RMS -- after which rho really is the fraction of receptor-separating signal that
       comes from structure.

    2. RENORMALISE THE MIXTURE. Two independent clouds of equal spread mix to spread
       sqrt(rho^2 + (1-rho)^2), which dips to 0.71 at rho = 0.5. Without a correction the
       middle of the dial is quieter than both ends by construction, and any dip in the
       curve there would be an artefact of the parameterisation rather than a finding.
       `renorm=False` leaves it uncorrected, which is how you check that claim.

    3. THE TRAIN MEAN IS THE OFFSET. It carries no between-receptor information -- a
       constant added to every node is a bias the first linear layer absorbs -- but
       keeping it is what makes rho = 1 the embedding file itself rather than something
       one rescaling away from it. Fitting it on train receptors matches what the v8
       structural anchor does; on these two regimes every receptor is warm anyway, so it
       is a discipline rather than a fix. Note the pooled mean over ALL receptors does
       drift slightly with rho, because a test receptor's deviation from the train
       centre is scaled like everyone else's -- which is the point of applying one
       fitted transform to all of them.

    rho = 1 short-circuits to the input dict, so the legacy end of the dial is the legacy
    input and not a float-error away from it.
    """
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"prot_mix must be in [0, 1], got {rho!r}")
    if rho == 1.0:
        return dict(proteins)
    recs = sorted(proteins)
    E = np.stack([np.asarray(proteins[r], dtype=np.float64) for r in recs])
    want = set(map(str, train_receptors or []))
    tr = [i for i, r in enumerate(recs) if str(r) in want] or list(range(len(recs)))

    mu = E[tr].mean(0)
    Ec = E - mu
    s_esm = _centred_rms(Ec[tr])

    R = _identity_vectors(recs, E.shape[1], seed)
    Rc = R - R[tr].mean(0)
    s_rand = _centred_rms(Rc[tr])
    if s_rand > 1e-12:
        Rc = Rc * (s_esm / s_rand)

    Z = rho * Ec + (1.0 - rho) * Rc
    if renorm:
        s_mix = _centred_rms(Z[tr])
        if s_mix > 1e-12:
            Z = Z * (s_esm / s_mix)
    out = mu + Z
    return {r: out[i].astype(np.float32) for i, r in enumerate(recs)}



def _identity_overlap(proteins, rho):
    """How one-hot the rho=0 end really is, as a printed number.

    Random directions are only NEARLY orthogonal, and how nearly depends on how many
    receptors are packed into the dimension: 24 vectors in 1280-d are essentially a
    basis, 1237 are not quite. The largest off-diagonal cosine is the honest statement
    of that, and it belongs in the log rather than in a footnote nobody reads."""
    recs = sorted(proteins)
    d = int(np.asarray(next(iter(proteins.values()))).shape[-1])
    R = _identity_vectors(recs, d)
    C = R @ R.T
    np.fill_diagonal(C, 0.0)
    return (f"identity vectors: {len(recs)} in {d}-d, |cos| mean "
            f"{np.abs(C).mean():.3f} max {np.abs(C).max():.3f}")


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
              k_mode: str = "coverage_quantile", task: str = "classification",
              edge_center: str = "global", edge_weight_mode: str = "none",
              select_seed: int = 0):
    """Train split's own pairs -> (pos, neg) local (mol_id, prot_id[, weight])
    arrays, optionally dropping molecules below the q-th quantile from message
    passing only -- every train row still gets decoded/supervised regardless.

    The quantile sets how many molecules to keep (coverage-quantile count); the
    `criterion` (see orbind/mol_selection.CRITERIA) picks WHICH ones.

    This function's own fallback stays "coverage" because that value reproduces
    the historical "keep counts >= quantile(counts, q)" filter bit-for-bit
    (zero-coverage molecules carry no edges, so their mask value is
    irrelevant). `GnnSignedExtractor` -- the only real caller -- always passes
    its own criterion explicitly, and **its** default is now
    "greedy_pair_cover"; see the note on that field.

    Two EXPERIMENTAL continuous-label knobs, both no-ops at their defaults so the
    historical binary path is byte-identical:
      `edge_center="per_receptor"` compares each edge's y to that RECEPTOR's own
        train-mean (+ edge_threshold) instead of the global edge_threshold (#4);
      `edge_weight_mode="magnitude"` returns a per-edge weight |y - center| per
        edge, normalised to unit mean within each sign, else the weight is None
        (#1). On binary labels |y - 0| = 1 everywhere, so weighting is the
        identity and this too reduces to the current graph.
    """
    sub = pairs.iloc[train_idx]
    mol_ids = sub["inchikey"].map(mol_to_i).to_numpy()
    prot_ids = sub["receptor"].map(prot_to_i).to_numpy()
    y = sub["label"].to_numpy()
    if q and q > 0:
        # Under classification pos_threshold stays None so `y == 1` -- and every
        # M2OR number -- reproduces exactly; under regression the ranking splits
        # positives at the very same threshold the edge signs use below.
        keep = mol_selection.select_keep_mask(
            criterion, mol_ids, prot_ids, y, len(mol_to_i), len(prot_to_i), q, k_mode,
            pos_threshold=(edge_threshold if task == "regression" else None),
            select_seed=select_seed)
        mask = keep[mol_ids]
        mol_ids, prot_ids, y = mol_ids[mask], prot_ids[mask], y[mask]
    # Deviation that decides an edge's sign. "global": y itself (so `y > 0` is
    # IDENTICAL to the historical `y == 1` on binary labels -- every M2OR run
    # reproduces bit-for-bit, and on a z-score it reads as "above the pool
    # average"). "per_receptor": y minus that receptor's own train-mean, which
    # removes cross-receptor baseline/dynamic-range heterogeneity (#4).
    y = y.astype(np.float64)
    if edge_center == "per_receptor":
        sums = np.bincount(prot_ids, weights=y, minlength=len(prot_to_i))
        cnts = np.bincount(prot_ids, minlength=len(prot_to_i)).astype(np.float64)
        center = np.divide(sums, np.maximum(cnts, 1.0))
        dev = y - center[prot_ids]
    else:
        dev = y
    pos_mask = dev > edge_threshold
    pw = nw = None
    if edge_weight_mode == "magnitude":
        w = np.abs(dev)
        pw, nw = w[pos_mask], w[~pos_mask]
        # normalise to unit mean per sign so the weighted graph keeps the same
        # overall message scale as the unweighted (all-ones) one
        if len(pw):
            pw = (pw / (pw.mean() + 1e-8)).astype(np.float32)
        if len(nw):
            nw = (nw / (nw.mean() + 1e-8)).astype(np.float32)
    return ((mol_ids[pos_mask], prot_ids[pos_mask], pw),
            (mol_ids[~pos_mask], prot_ids[~pos_mask], nw))


def _edge_index_dict(pos, neg):
    pm, pp = pos[0], pos[1]
    nm, npt = neg[0], neg[1]
    def _idx(a, b):
        if len(a) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(np.stack([a, b]), dtype=torch.long)
    pos_eidx = {ETYPE: _idx(pm, pp), RTYPE: _idx(pp, pm)}
    neg_eidx = {ETYPE_NEG: _idx(nm, npt), RTYPE_NEG: _idx(npt, nm)}
    return pos_eidx, neg_eidx


def _edge_weights(pos, neg):
    """Per-edge weight dicts aligned with `_edge_index_dict`, or (None, None)
    when `_mp_edges` produced no weights (the default unweighted graph)."""
    pw = pos[2] if len(pos) > 2 else None
    nw = neg[2] if len(neg) > 2 else None
    if pw is None and nw is None:
        return None, None
    def _t(a):
        return None if a is None else torch.as_tensor(a, dtype=torch.float32)
    pos_ew = None if pw is None else {ETYPE: _t(pw), RTYPE: _t(pw)}
    neg_ew = None if nw is None else {ETYPE_NEG: _t(nw), RTYPE_NEG: _t(nw)}
    return pos_ew, neg_ew


# --------------------------------------------------------------------------- model

def _fit_pca(X: np.ndarray, k: int):
    """PCA basis (components [k, dim], mean [dim]) fit on train node features,
    for the `dummy_compression` variant: the learned input projection to `hidden`
    is replaced by this fixed linear map. Fit on TRAIN entities only (no leak)."""
    from sklearn.decomposition import PCA
    n, d = X.shape
    if k > min(n, d):
        raise ValueError(f"dummy_compression: PCA n_components={k} needs "
                         f"min(n_train={n}, dim={d}) >= {k}")
    p = PCA(n_components=k, random_state=0).fit(X.astype(np.float64))
    return p.components_.astype(np.float32), p.mean_.astype(np.float32)


def _freeze_pca(linear: nn.Linear, comp: np.ndarray, mean: np.ndarray):
    """Overwrite a Linear with a fixed PCA projection and freeze it, so
    `linear(x) = comp @ (x - mean)` and it carries no trainable parameters."""
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(comp, dtype=linear.weight.dtype))
        linear.bias.copy_(torch.as_tensor(-(comp @ mean), dtype=linear.bias.dtype))
    linear.weight.requires_grad_(False)
    linear.bias.requires_grad_(False)


def _rms(x: "torch.Tensor") -> "torch.Tensor":
    """Scale a node block by the RMS of its BETWEEN-NODE variation.

    One scalar for the whole block, so the arrangement of the rows -- all any
    geometry readout looks at -- is untouched and only the overall scale is fixed.
    The block itself is NOT centred: `_structural_anchor` needs its raw offset to
    keep ESM's cosines (see there), and the decoder uses magnitude.

    The divisor is the centred RMS rather than the plain one, and that distinction
    turned out to be the whole ballgame. Mean-pooled ESM is ~94% common mean: on CC
    the between-receptor share of its energy is 5.6%, on HC 7.2%, against ~99% for
    the graph's (essentially mean-free) output. Equalising TOTAL energy therefore
    handed the structural branch only ~6% of the geometry-carrying signal at the
    same nominal weight, and the crossover landed at alpha ~ 0.05 instead of 0.5 --
    measured: alpha = 0.25 through 1.0 were geometrically indistinguishable, and the
    dial was a step at zero. Dividing by the spread makes alpha an exchange rate
    between the two clouds' actual variation, which is what it was supposed to be.
    Both endpoints survive unchanged: at 0 and at 1 this is a positive scalar on a
    single branch, and no geometry here sees a scalar."""
    spread = (x - x.mean(0, keepdim=True)).pow(2).mean().sqrt()
    return x / spread.clamp_min(1e-8)


def _structural_anchor(x_prot, train_rows, hidden: int):
    """The v8 gate's FROZEN structural branch: raw ESM rotated into a train-fit basis.

    Fit on TRAIN receptors only and applied to every receptor in the universe, so a
    cold-receptor split projects test rows through a train basis and nothing leaks.

    UNCENTERED (a truncated SVD of X_train, not sklearn's mean-subtracting PCA), and
    that is not a detail. Every geometry readout we report -- RSA above all -- scores
    the COSINES between receptors, and cosines are not invariant to a shift of the
    origin: centring preserves distances while changing every angle, so a centred
    anchor would sit at RSA ~0.9 against the very cloud it is a copy of. Projecting
    onto the train span instead makes the branch an exact isometry there. With
    k >= n_train every train receptor keeps its norm and every pair its inner
    product, so `alpha=0` reproduces ESM's geometry exactly rather than nearly.

    Rank is min(hidden, n_train, dim), zero-padded out to `hidden`. On these panels
    that binds at n_train (CC 50, HC 24, both far below hidden=256), which is
    precisely the regime where the map is lossless.

    Returns (tensor [n_prot, hidden], rank actually used)."""
    X = np.asarray(x_prot.detach().cpu().numpy(), dtype=np.float64)
    tr = np.unique(np.asarray(list(train_rows), dtype=int))
    k = int(min(hidden, len(tr), X.shape[1]))
    if k < 2:
        raise ValueError(f"structural anchor needs >= 2 train receptors, got {len(tr)}")
    _, _, Vt = np.linalg.svd(X[tr], full_matrices=False)
    S = np.zeros((X.shape[0], hidden), dtype=np.float32)
    S[:, :k] = (X @ Vt[:k].T).astype(np.float32)
    return torch.as_tensor(S), k


class _WSAGE(MessagePassing):
    """Weighted bipartite conv used ONLY in the experimental magnitude-weighted
    edge mode (`edge_weight_mode="magnitude"`). Same shape as the SAGEConv it
    replaces -- root transform plus mean-aggregated neighbour transform, hidden
    -> hidden on both endpoints (inputs are already projected to `hidden`) --
    but each neighbour message is scaled by its scalar edge weight. With unit
    weights it matches SAGEConv's mean aggregation; the default path never builds
    this class, so no existing number moves."""

    def __init__(self, hidden: int):
        super().__init__(aggr="mean")
        self.lin_r = nn.Linear(hidden, hidden)   # neighbour
        self.lin_l = nn.Linear(hidden, hidden)   # root (target)

    def forward(self, x, edge_index, edge_weight=None):
        x_src, x_dst = x
        out = self.lin_r(self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weight))
        if x_dst is not None:
            out = out + self.lin_l(x_dst)
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)


#: The message-passing operators this encoder can be built from. Everything else --
#: the signed two-stack structure, the subtraction, the decoder, the training loop --
#: is held fixed, so a row of the architecture table differs from ours in the operator
#: and in nothing else.
#:
#: Why these four and not the textbook list:
#:   sage        ours. Mean-aggregation GraphSAGE, the operator every number in this
#:               paper was produced with. It is stock `SAGEConv` at its defaults
#:               (mean aggregation, a root weight for the node's own state, no
#:               post-normalisation), but see WHAT "GRAPHSAGE" MEANS HERE below --
#:               the operator is Hamilton et al.'s, the algorithm around it is not.
#:   gat         attention. Ported from the v1--v5 GAT epoch (orbind/legacy/hetero_gat)
#:               including the two details that make it work on a bipartite graph:
#:               `add_self_loops=False` (a molecule has no molecule neighbours, so PyG's
#:               default self-loop is a type error waiting to happen) and multi-head
#:               concat on layer 1, single head on layer 2.
#:   graphconv   the GCN-shaped one. Plain `GCNConv` is NOT here and cannot be: its
#:               symmetric normalisation and mandatory self-loops assume one node set,
#:               and our graph has two. `GraphConv` is the same idea (sum over
#:               neighbours + a self term) that is defined on a bipartite graph.
#:   gin         the molecular-domain standard (Xu et al. 2019) -- the operator behind
#:               the GIN molecule embeddings this repo already uses as a molecule
#:               source. Sum aggregation and an MLP, which is the most expressive of
#:               the four in the WL sense.
CONVS = ("sage", "gat", "graphconv", "gin")

#: Layer 1 concatenates `heads` of `hidden // heads`; layer 2 is single-head, so the
#: width the decoder sees is `hidden` for every operator. Comparing operators at
#: different widths would compare widths.
GAT_HEADS = 4

# WHAT "GRAPHSAGE" MEANS HERE, AND WHERE IT DEPARTS FROM THE PAPER
# ---------------------------------------------------------------
# Hamilton, Ying & Leskovec (NeurIPS 2017) is two things: a convolution and a training
# regime. We take the first and not the second, and a reviewer who asks "is this
# GraphSAGE?" deserves the list rather than the label.
#
# THE SAME
#   * the operator: h_v' = W_root x_v + W_neigh MEAN_{u in N(v)} x_u, which is PyG's
#     SAGEConv at its defaults and the paper's mean aggregator in its CONCAT form;
#   * depth 2, which is what the paper uses and recommends.
#
# DIFFERENT, AND THE FIRST ONE IS THE BIG ONE
#   * NO NEIGHBOURHOOD SAMPLING. The paper's defining mechanism is fixed-size sampled
#     neighbourhoods per layer (25 then 10) in minibatches; we run FULL BATCH over the
#     whole graph every epoch, because this graph fits. Consequence worth stating out
#     loud: every training pair is simultaneously a message-passing edge and a
#     supervision target, so the encoder sees the label it is asked to predict through
#     the graph. That is the coupling the edge-sampling line of work is about, and it
#     is a known open direction, not an oversight.
#   * NO PER-LAYER L2 NORMALISATION. The paper normalises h_v to unit norm after each
#     layer; we do not, and the final layer has no activation either -- which is why
#     `_dgi_pool` has to normalise by hand before it touches a sigmoid.
#   * THE SIGN. Positive and negative edges flow through separate stacks and are
#     SUBTRACTED. There is no such thing in GraphSAGE; it is this project's own
#     construction and the reason the encoder is called signed.
#   * HETEROGENEOUS AND BIPARTITE. Two node types, separate weights per direction
#     (molecule->receptor, receptor->molecule), combined by `HeteroConv(aggr="sum")`.
#     The paper is homogeneous; this is the standard PyG extension of it.
#   * LeakyReLU(0.1) rather than ReLU, from the collapse fix: a unit driven negative
#     under plain ReLU gets zero gradient and never comes back.
#   * SUPERVISION AND READOUT. The paper's headline loss is unsupervised with negative
#     sampling; ours is BCE (or regression) on decoded pairs through a 3-layer MLP
#     over the concatenated (molecule, receptor) embeddings, at a fixed epoch budget
#     with no early stopping and no scheduler.
#
# So the architecture row labelled "ours" compares OPERATORS inside one fixed
# algorithm. It is not a claim that we reproduced GraphSAGE the paper.


def _make_conv(kind: str, hidden: int, layer: int, heads: int, dropout: float,
               weighted: bool):
    """One message-passing operator for one edge type.

    `layer` is 1 or 2 and only GAT reads it (multi-head then single-head). Lazy
    `(-1, -1)` input dims everywhere they are supported, because layer 1 sees the two
    projected node types and layer 2 sees the subtraction's output.
    """
    if kind == "sage":
        return _WSAGE(hidden) if weighted else SAGEConv((-1, -1), hidden)
    if weighted:
        # `edge_weight_mode="magnitude"` is implemented by _WSAGE alone; silently
        # ignoring the weights under another operator would report a weighted run that
        # was not one.
        raise ValueError(f"edge_weight_mode='magnitude' is implemented for conv='sage' "
                         f"only, got conv={kind!r}")
    if kind == "gat":
        from torch_geometric.nn import GATConv
        if layer == 1:
            return GATConv((-1, -1), hidden // heads, heads=heads, dropout=dropout,
                           add_self_loops=False)
        return GATConv((-1, -1), hidden, heads=1, dropout=dropout,
                       add_self_loops=False)
    if kind == "graphconv":
        from torch_geometric.nn import GraphConv
        return GraphConv((-1, -1), hidden, aggr="mean")
    if kind == "gin":
        from torch_geometric.nn import GINConv
        # GIN has no lazy input, which is fine: both node types are `hidden`-wide from
        # the input projections onward, and the subtraction keeps them there.
        return GINConv(nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.LeakyReLU(_SignedSage.LEAK),
                                     nn.Linear(hidden, hidden)), train_eps=True)
    raise ValueError(f"conv must be one of {CONVS}, got {kind!r}")


def sample_neighbours(eidx, fanout, generator=None):
    """Keep at most `fanout` incoming edges per DESTINATION node, drawn uniformly.

    This is GraphSAGE's mechanism, brought into a full-batch loop: the paper samples a
    fixed-size neighbourhood per layer (25 then 10) inside a minibatch, and what the
    sampling actually does is bound how much of the graph any one node sees per layer
    and resample it every step. Resampling each EPOCH over the whole graph gives the
    same two effects -- a bounded receptive field and stochastic message passing --
    without the minibatch machinery.

    Sampled WITHOUT replacement, so a thin neighbourhood is kept whole rather than
    padded with duplicates: duplicating an edge would reweight that neighbour in a
    mean aggregation, which is the opposite of what a fan-out is for.

    Each edge TYPE is sampled independently, and in the signed encoder that matters:
    positive and negative edges live in different dicts, so each sign keeps its own
    fan-out and the pos/neg balance a receptor sees is not rewritten by the draw.

    `edge_index[0]` is the source and `edge_index[1]` the destination, which is the
    direction messages travel -- sampling by source would bound how much each node
    SENDS, and a hub would still flood its neighbours.
    """
    if not fanout or fanout <= 0:
        return eidx
    out = {}
    for k, e in eidx.items():
        n_edges = e.shape[1]
        if n_edges == 0:
            out[k] = e
            continue
        perm = torch.randperm(n_edges, generator=generator, device=e.device)
        dst = e[1, perm]
        order = torch.argsort(dst, stable=True)
        dst_sorted = dst[order]
        counts = torch.bincount(dst_sorted)
        # rank of each edge inside its destination's group, after the shuffle: the
        # shuffle is what makes "the first `fanout` of the group" a uniform draw
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(n_edges, device=e.device) - starts[dst_sorted]
        out[k] = e[:, perm[order[rank < fanout]]]
    return out


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

    def __init__(self, mol_dim: int, prot_dim: int, hidden: int, dropout: float,
                 weighted: bool = False, pca_mol=None, pca_prot=None,
                 alpha=None, s_prot=None, conv: str = "sage", heads: int = GAT_HEADS,
                 normalize: bool = False):
        super().__init__()
        self.weighted = weighted
        self.conv_kind = conv
        # GraphSAGE normalises h_v to unit norm after every layer; we historically did
        # not, and every reported number is un-normalised. Opt-in, so the default path
        # is byte-identical.
        self.normalize = bool(normalize)
        # v8 hard gate. `alpha=None` is the historical model, untouched: no branch,
        # no normalisation, every pre-v8 number reproduces byte-for-byte.
        self.alpha = None if alpha is None else float(alpha)
        self.register_buffer("s_prot", None if s_prot is None else _rms(s_prot),
                             persistent=False)   # frozen; never in a state_dict
        # weighted mode swaps SAGEConv for the edge-weight-aware _WSAGE; the
        # default (conv="sage", weighted=False) keeps PyG's SAGEConv byte-for-byte,
        # and the parameters are still created in the same order, so a seeded init
        # reproduces every number made before this switch existed.
        if conv == "gat" and hidden % heads:
            raise ValueError(f"hidden={hidden} must divide by heads={heads}: layer 1 "
                             f"concatenates the heads back to `hidden`")

        def mk(layer=1):
            return _make_conv(conv, hidden, layer, heads, dropout, weighted)
        self.proj_mol = nn.Linear(mol_dim, hidden)
        self.proj_prot = nn.Linear(prot_dim, hidden)
        # dummy_compression: freeze the two input projections to a fixed PCA basis
        # fit on train features, so ~426k learned weights become non-trainable.
        if pca_mol is not None:
            _freeze_pca(self.proj_mol, *pca_mol)
        if pca_prot is not None:
            _freeze_pca(self.proj_prot, *pca_prot)
        self.conv1 = HeteroConv({ETYPE: mk(1), RTYPE: mk(1)}, aggr="sum")
        self.conv1_neg = HeteroConv({ETYPE_NEG: mk(1), RTYPE_NEG: mk(1)}, aggr="sum")
        self.conv2 = HeteroConv({ETYPE: mk(2), RTYPE: mk(2)}, aggr="sum")
        self.conv2_neg = HeteroConv({ETYPE_NEG: mk(2), RTYPE_NEG: mk(2)}, aggr="sum")
        self.dec = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LeakyReLU(self.LEAK), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(self.LEAK), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def encode(self, x_mol, x_prot, pos_eidx, neg_eidx, pos_ew=None, neg_ew=None):
        """`pos_eidx`/`neg_eidx` are one edge dict, or a PAIR of them -- layer 1's and
        layer 2's. The pair is what neighbour sampling needs: the paper draws a
        different fan-out per layer (25 then 10), and one dict for both would be a
        single draw applied twice."""
        def per_layer(e):
            return e if isinstance(e, (list, tuple)) else (e, e)

        pos1, pos2 = per_layer(pos_eidx)
        neg1, neg2 = per_layer(neg_eidx)
        x = {MOL: F.leaky_relu(self.proj_mol(x_mol), self.LEAK),
             PROT: F.leaky_relu(self.proj_prot(x_prot), self.LEAK)}
        kp = {"edge_weight_dict": pos_ew} if (self.weighted and pos_ew is not None) else {}
        kn = {"edge_weight_dict": neg_ew} if (self.weighted and neg_ew is not None) else {}
        x_p = self.conv1(x, pos1, **kp)
        x_n = self.conv1_neg(x, neg1, **kn)
        x = {k: F.leaky_relu(x_p[k] - x_n.get(k, torch.zeros_like(x_p[k])), self.LEAK) for k in x_p}
        if self.normalize:
            x = {k: F.normalize(v, dim=-1) for k, v in x.items()}
        x_p = self.conv2(x, pos2, **kp)
        x_n = self.conv2_neg(x, neg2, **kn)
        z = {k: x_p[k] - x_n.get(k, torch.zeros_like(x_p[k])) for k in x_p}
        if self.normalize:
            # before the alpha gate, which does its own RMS scaling: normalising after
            # it would undo the mixture it was set to produce
            z = {k: F.normalize(v, dim=-1) for k, v in z.items()}
        if self.alpha is not None and self.s_prot is not None and PROT in z:
            # The receptor readout is a FIXED convex mix of a frozen structural
            # branch and the trained graph. alpha is set before training and used
            # unchanged at inference -- it is a property of the model, not a
            # post-hoc dial. Gradient cannot reach s_prot, which is the whole
            # point: 900 epochs of binding loss can no longer erase ESM.
            z[PROT] = (1.0 - self.alpha) * self.s_prot + self.alpha * _rms(z[PROT])
        return z

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
                dgi_weight=0.0, dgi_scope="shared", hidden=None, pos_ew=None, neg_ew=None,
                deterministic_init=False, fanout=None):
    """Full-batch training loop (the whole graph is small enough to fit in
    one forward/backward per epoch): BCE loss on train-row decodes, fixed
    epoch count, no early stopping, no LR scheduler -- matches the actual v5
    graph-screen protocol (legacy/scripts/modeling/train/run_graph_full_full_v5.ps1:
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
    if deterministic_init:
        # global RNG under a lock: the n_models bags run as concurrent THREADS in one
        # process, so two of them seeding and building at once would interleave their
        # draws and neither would be reproducible. `hp["seed"]` already differs per bag
        # (seed + seed_offset*m), so the lock costs nothing but the build itself.
        with _INIT_LOCK:
            torch.manual_seed(hp["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(hp["seed"])
            model = build_model().to(device)
    else:
        model = build_model().to(device)
    x_mol_d, x_prot_d = x_mol.to(device), x_prot.to(device)
    pos_eidx_d = {k: v.to(device) for k, v in pos_eidx.items()}
    neg_eidx_d = {k: v.to(device) for k, v in neg_eidx.items()}
    pos_ew_d = None if pos_ew is None else {k: v.to(device) for k, v in pos_ew.items()}
    neg_ew_d = None if neg_ew is None else {k: v.to(device) for k, v in neg_ew.items()}

    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        with torch.no_grad():
            z = model.encode(x_mol_d, x_prot_d, pos_eidx_d, neg_eidx_d, pos_ew_d, neg_ew_d)
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

    # Neighbour sampling: a fresh draw every epoch, one fan-out per layer. Its own
    # generator, seeded from the model seed, so the draw is reproducible and does not
    # consume the global RNG the rest of the run depends on. Edge weights are NOT
    # sampled with the edges, which is why `--edge-weight-mode magnitude` and fanout
    # are refused together upstream.
    gen = None
    if fanout:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(hp["seed"]))

    def sampled():
        if not fanout:
            return pos_eidx_d, neg_eidx_d
        p = [sample_neighbours(pos_eidx_d, f, gen) for f in fanout]
        n = [sample_neighbours(neg_eidx_d, f, gen) for f in fanout]
        return p, n

    for _ in range(hp["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        pe, ne = sampled()
        z = model.encode(x_mol_d, x_prot_d, pe, ne, pos_ew_d, neg_ew_d)
        loss = loss_fn(model.decode(z, mi_tr, pi_tr), y_tr)
        if dgi_w is not None:
            x_mol_c = x_mol_d[torch.randperm(x_mol_d.shape[0], device=device)]
            x_prot_c = x_prot_d[torch.randperm(x_prot_d.shape[0], device=device)]
            z_c = model.encode(x_mol_c, x_prot_c, pe, ne, pos_ew_d, neg_ew_d)
            loss = loss + dgi_weight * _dgi_loss(dgi_w, _dgi_pool(z, dgi_scope), _dgi_pool(z_c, dgi_scope))
        loss.backward()
        nn.utils.clip_grad_norm_(params, hp["clip_grad"])
        opt.step()

    # INFERENCE IS FULL-NEIGHBOURHOOD, sampling or not -- the paper does the same.
    # Sampling is a training-time device; emitting embeddings from one random draw
    # would make the row we report depend on which draw happened last.
    model.eval()
    with torch.no_grad():
        z = model.encode(x_mol_d, x_prot_d, pos_eidx_d, neg_eidx_d, pos_ew_d, neg_ew_d)
        z_mol = z[MOL].cpu().numpy()
        z_prot = z[PROT].cpu().numpy()
    return z_mol, z_prot, model


# Criteria computable without labels, hence usable under task="regression".
# `coverage` is a bincount over train edges; `greedy_pair_cover` orders molecules
# by newly covered protein PAIRS, built from `prof`/`active` only. The other five
# are functions of npos/nneg (`y == 1` / `y == 0`) and collapse to zeros on a
# continuous target. See orbind/mol_selection.compute_mol_scores.
_LABEL_AGNOSTIC_CRITERIA = {"coverage", "greedy_pair_cover", "random"}


def _run_models(ext, pairs: pd.DataFrame, train_idx, val_idx, test_idx, seed: int, checkpoint_dir=None):
    check_task(ext.task)
    # Label-based criteria on a continuous target are fine as long as "positive"
    # is DEFINED -- `_mp_edges` passes `edge_threshold` down as the binarisation
    # point, the same one the signed graph uses for edge signs. What is not fine
    # is leaving it undefined: `y == 1` matches nothing on a z-score and the
    # ranking would silently collapse to all-zeros instead of failing.
    if ext.task == "regression" and ext.criterion not in _LABEL_AGNOSTIC_CRITERIA \
            and getattr(ext, "edge_threshold", None) is None:
        raise ValueError(
            f"{ext.name}: criterion {ext.criterion!r} scores molecules by their "
            f"positive/negative mix, which needs a threshold on a continuous "
            f"target; set `edge_threshold` or use one of "
            f"{sorted(_LABEL_AGNOSTIC_CRITERIA)}")
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            missing = int((~mask).sum())
            raise KeyError(
                f"{ext.name}: {missing}/{len(idx)} {split_name} rows missing a protein or "
                f"molecule embedding (protein={ext.protein_path}, molecule={ext.molecule_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if getattr(ext, "prot_mix", None) is not None and ext._anchor_proteins is None:
        # v9: the node features move along the dial. The originals are kept aside for
        # the same reason the one-hot swap keeps them -- the v8 gate's frozen branch is
        # built from ESM, and an SVD of the mixture would quietly make that axis mean
        # something else. The two dials are independent and may be combined; neither
        # reads the other's value.
        ext._anchor_proteins = ext._proteins
        tr_recs = pd.unique(pairs.iloc[train_idx]["receptor"])
        ext._proteins = mix_protein_features(
            ext._anchor_proteins, tr_recs, ext.prot_mix,
            seed=getattr(ext, "mix_seed", 0),
            renorm=getattr(ext, "mix_renorm", True))
        diag = _identity_overlap(ext._anchor_proteins, ext.prot_mix)
        print(f"  {ext.name}: v9 node dial rho={ext.prot_mix:g} over "
              f"{len(ext._proteins)} receptors (rho=1 -> the embedding file itself, "
              f"rho=0 -> one fixed near-orthogonal vector each); {diag}", flush=True)
    if getattr(ext, "onehot_nodes", False) and ext._anchor_proteins is None:
        # Swap the receptor NODE features for an identity over exactly the receptors
        # the embedding file covers -- same universe, same edges, only the structural
        # prior removed (the construction `mechanism_holdout.train_refined` uses for
        # its `feat="onehot"` arm). The real vectors are kept aside because the alpha
        # gate's frozen branch is built from THEM, not from the node features: with
        # both swapped the branch would be an SVD of an identity matrix, which carries
        # no structure at all and would silently turn the axis into nonsense.
        ext._anchor_proteins = ext._proteins
        recs = sorted(ext._anchor_proteins)
        eye = np.eye(len(recs), dtype=np.float32)
        ext._proteins = {r: eye[i] for i, r in enumerate(recs)}
        print(f"  {ext.name}: node features = one-hot over {len(recs)} receptors; "
              f"ESM enters only through the gate's frozen branch", flush=True)
    mol_to_i, prot_to_i, x_mol, x_prot = _build_universe(pairs, all_idx, ext._proteins, ext._molecules)

    if getattr(ext, "dummy_compression", False):
        tr_mols = pd.unique(pairs.iloc[train_idx]["inchikey"])
        tr_prots = pd.unique(pairs.iloc[train_idx]["receptor"])
        ext._pca_mol = _fit_pca(np.stack([ext._molecules[m] for m in tr_mols]), ext.hidden)
        ext._pca_prot = _fit_pca(np.stack([ext._proteins[p] for p in tr_prots]), ext.hidden)
        print(f"  {ext.name}: dummy_compression PCA fit on {len(tr_mols)} train mols / "
              f"{len(tr_prots)} train prots -> hidden {ext.hidden} "
              f"(frozen input projections)", flush=True)

    ext._s_prot = None
    if getattr(ext, "alpha", None) is not None:
        src = ext._anchor_proteins or ext._proteins      # ESM, even when nodes are one-hot
        order = sorted(prot_to_i, key=prot_to_i.get)
        x_anchor = torch.tensor(np.stack([src[r] for r in order]), dtype=torch.float32)
        tr_prots = pd.unique(pairs.iloc[train_idx]["receptor"])
        ext._s_prot, k_pca = _structural_anchor(
            x_anchor, [prot_to_i[r] for r in tr_prots if r in prot_to_i], ext.hidden)
        ext._k_pca = k_pca
        print(f"  {ext.name}: alpha gate {ext.alpha:g} -- structural branch = SVD{k_pca} of "
              f"{'ESM (nodes are one-hot)' if ext._anchor_proteins else 'ESM'} fit on "
              f"{len(tr_prots)} train receptors, frozen "
              f"(alpha=0 -> pure ESM geometry, alpha=1 -> the graph alone)", flush=True)

    train_df = pairs.iloc[train_idx]
    mol_idx_train = train_df["inchikey"].map(mol_to_i).to_numpy().copy()
    prot_idx_train = train_df["receptor"].map(prot_to_i).to_numpy().copy()
    y_train = train_df["label"].to_numpy(dtype=np.float32)

    pos, neg = _mp_edges(pairs, train_idx, mol_to_i, prot_to_i, ext.q,
                         getattr(ext, "criterion", "greedy_pair_cover"),
                         getattr(ext, "edge_threshold", 0.0),
                         getattr(ext, "k_mode", "coverage_quantile"), ext.task,
                         getattr(ext, "edge_center", "global"),
                         getattr(ext, "edge_weight_mode", "none"),
                         getattr(ext, "select_seed", 0))
    n_pos, n_neg = len(pos[0]), len(neg[0])
    print(f"  {ext.name}: MP graph {n_pos} positive / {n_neg} negative edges "
          f"(q={ext.q}, criterion={getattr(ext, 'criterion', 'greedy_pair_cover')}, "
          f"k_mode={getattr(ext, 'k_mode', 'coverage_quantile')}, "
          f"edge_threshold={getattr(ext, 'edge_threshold', 0.0)}, "
          f"edge_center={getattr(ext, 'edge_center', 'global')}, "
          f"edge_weight_mode={getattr(ext, 'edge_weight_mode', 'none')})", flush=True)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"{ext.name}: signed message passing needs both edge signs, got "
            f"{n_pos} positive / {n_neg} negative. Check `edge_threshold` "
            f"(={getattr(ext, 'edge_threshold', 0.0)}) against the label scale.")
    pos_eidx, neg_eidx = _edge_index_dict(pos, neg)
    pos_ew, neg_ew = _edge_weights(pos, neg)

    def model_job(m):
        hp = ext._hp(seed + ext.seed_offset * m)
        checkpoint_path = (pathlib.Path(checkpoint_dir) / f"gnn_{ext.name}_model{m}.pt"
                            if checkpoint_dir is not None else None)
        z_mol, z_prot, model = _train_one(ext._build_model, x_mol, x_prot, pos_eidx, neg_eidx,
                                           mol_idx_train, prot_idx_train, y_train, hp, device,
                                           checkpoint_path=checkpoint_path,
                                           dgi_weight=getattr(ext, "dgi_weight", 0.0),
                                           dgi_scope=getattr(ext, "dgi_scope", "shared"),
                                           hidden=ext.hidden, pos_ew=pos_ew, neg_ew=neg_ew,
                                           deterministic_init=getattr(
                                               ext, "deterministic_init", False),
                                           fanout=getattr(ext, "fanout", ()))
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
        elif ext.emit == "mol":
            # mirror of emit="prot": the graph-refined MOLECULE vector, for the
            # cold-RECEPTOR regime where molecules are the seen (transductive)
            # side, so pairing [raw receptor || refined molecule] is the honest
            # reversal of our cold-molecule "refined protein" paradigm.
            blocks = [z_mol[mi] for _, z_mol, _, _ in results]
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
    # Which draw the `random` CONTROL criterion makes; ignored by every other
    # criterion, which are deterministic given the train edges. Set it per graph seed
    # when sweeping, or the control is one lucky draw reported as a baseline.
    select_seed: int = 0
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
    # EXPERIMENTAL continuous-label edge modes (off by default -> the graph is
    # byte-identical to the historical signed one; every M2OR number reproduces).
    # Meant for one-off Carey/Hallem probing, NOT default or mass runs.
    #   edge_center      "global" (compare y to edge_threshold, historical) |
    #                    "per_receptor" (compare y to that receptor's own
    #                    train-mean + edge_threshold -- #4)
    #   edge_weight_mode "none" (unweighted messages, historical) |
    #                    "magnitude" (scale each edge's message by |y - center|,
    #                    unit-mean-normalised per sign -- #1; on binary labels
    #                    |y|=1 so it is the identity)
    edge_center: str = "global"
    edge_weight_mode: str = "none"
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
    # dummy_compression: replace the learned input projections (mol_dim->hidden,
    # prot_dim->hidden) with a fixed PCA fit on train node features. Drops ~426k
    # learned weights; message passing + decoder still train on top.
    dummy_compression: bool = False
    # v8 HARD GATE (default None = the historical graph, bit-identical).
    #   z_prot = (1 - alpha) * frozen_PCA(ESM) + alpha * graph_output
    # Both branches RMS-normalised so alpha is a real exchange rate rather than a
    # contest between two arbitrary scales. Fixed at construction, used unchanged
    # at inference. Rationale: with alpha=None the ONLY path from ESM to the
    # receptor embedding is trainable, and 900 epochs of binding loss empty it --
    # measured, the refined receptor cloud sits at the permutation null against
    # ESM. The frozen branch is a path the optimizer cannot drain, which is what
    # turns "structure vs function" from two ablations into one dial.
    alpha: float | None = None
    # Receptor NODE features: the embedding file (default) or a one-hot identity.
    # With the gate on, one-hot nodes are what make alpha an honest fraction of
    # structure: otherwise ESM reaches the receptor vector by TWO routes -- the
    # frozen branch at weight (1 - alpha) AND the node features the graph is trained
    # on -- so the alpha=1 end is "a graph that has already seen ESM", not "no ESM".
    # With one-hot nodes the only route is the branch, and alpha=1 contains no
    # structural information whatsoever.
    onehot_nodes: bool = False
    # v9 NODE DIAL (default None = the node features are the embedding file, untouched).
    #   x_prot = mu + rho*centred(ESM) + (1 - rho)*centred(one fixed random unit vector
    #                                                      per receptor)
    # rho = 1 is the legacy graph exactly; rho = 0 is a graph over receptor IDENTITY and
    # nothing else -- a one-hot in all but coordinates, and near-orthogonal because n
    # random directions in 1280-d are. This is a different dial from `alpha`: that one
    # gates the graph's OUTPUT against a frozen ESM branch and runs structure -> function
    # as it rises, this one moves the graph's INPUT and runs function -> structure. They
    # are independent and may be set together; neither reads the other.
    prot_mix: float | None = None
    # Which draw of the identity vectors. NOT the model seed -- who a receptor is should
    # not change with the training run -- but sweeping it checks no result rests on one
    # lucky set of directions.
    mix_seed: int = 0
    # Rescale the mixture to constant centred spread. Off, the middle of the dial is
    # quieter than both ends by sqrt(rho^2 + (1-rho)^2), and a dip there would be an
    # artefact of the parameterisation. See `mix_protein_features`.
    mix_renorm: bool = True
    # Seed the GRAPH's own initialisation from `hp["seed"]`, the way every other
    # extractor in this repo already does (hladis/lorax/molor/prosmith all call
    # torch.manual_seed there). Default False ONLY because turning it on changes every
    # number already in results/graph/: with it off the weights come from torch's global
    # RNG, which is drawn from OS entropy at first use and is then advanced by whatever
    # else that worker process happened to train first. That is the whole reason the
    # alpha=1 and graph_legacy arms -- provably the same computation on the same input
    # -- do not land on the same number.
    #
    # It does not buy bit-determinism on a GPU: the message passing's scatter-add is
    # atomic and reorders between runs. It removes the init lottery, which is the large
    # half.
    deterministic_init: bool = False
    # WHICH message-passing operator the signed stacks are built from
    # (orbind.gnn_extractor.CONVS). "sage" is ours and is what every reported number
    # uses; the others exist for the architecture ablation and change nothing else
    # about the model -- same signed structure, same decoder, same training loop.
    conv: str = "sage"
    # GAT only: heads on layer 1, concatenated back to `hidden`. Ignored otherwise.
    heads: int = GAT_HEADS
    # The two things our encoder historically did NOT take from GraphSAGE, both
    # opt-in so every reported number is untouched (see WHAT "GRAPHSAGE" MEANS HERE):
    #   fanout=(25, 10)   sample at most this many incoming edges per node per layer,
    #                     redrawn each epoch; inference stays full-neighbourhood
    #   normalize_layers  L2-normalise the node embeddings after every layer
    fanout: tuple = ()
    normalize_layers: bool = False
    pooling: str = "signed_sage"
    dgi_weight: float = 0.0            # >0 enables the DeepGraphInfomax auxiliary loss
    dgi_scope: str = "shared"          # "shared" (mol+prot) or "prot"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="gnn_signed")

    def __post_init__(self):
        if self.emit not in ("prot", "mol", "both"):
            raise ValueError(f"emit must be 'prot', 'mol' or 'both', got {self.emit!r}")
        if self.criterion not in mol_selection.CRITERIA:
            raise ValueError(f"criterion must be one of {mol_selection.CRITERIA}, "
                             f"got {self.criterion!r}")
        if self.fanout:
            self.fanout = tuple(int(f) for f in self.fanout)
            if any(f <= 0 for f in self.fanout):
                raise ValueError(f"fanout entries must be positive, got {self.fanout}")
            if len(self.fanout) != 2:
                raise ValueError(f"the encoder has two layers, so fanout needs two "
                                 f"entries, got {self.fanout}")
            if self.edge_weight_mode != "none":
                # the sampler drops edges but not their weights, so the two would
                # silently disagree about which edge a weight belongs to
                raise ValueError("fanout and edge_weight_mode='magnitude' cannot be "
                                 "combined: the sampler does not carry edge weights")
        if self.conv not in CONVS:
            raise ValueError(f"conv must be one of {CONVS}, got {self.conv!r}")
        if self.conv == "gat" and self.hidden % self.heads:
            raise ValueError(f"hidden={self.hidden} must divide by heads={self.heads}")
        if self.k_mode not in mol_selection.K_MODES:
            raise ValueError(f"k_mode must be one of {mol_selection.K_MODES}, "
                             f"got {self.k_mode!r}")
        if self.alpha is not None and not (0.0 <= float(self.alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0, 1] or None, got {self.alpha!r}")
        if self.prot_mix is not None and not (0.0 <= float(self.prot_mix) <= 1.0):
            raise ValueError(f"prot_mix must be in [0, 1] or None, got {self.prot_mix!r}")
        if self.prot_mix is not None and self.onehot_nodes:
            # Both replace the receptor node features, so one would silently win. They
            # also mean nearly the same thing at one end -- prot_mix=0 IS the one-hot
            # arm, up to a rotation -- which is exactly why picking by accident is bad.
            raise ValueError("prot_mix and onehot_nodes both replace the receptor node "
                             "features; prot_mix=0 is the one-hot end of that dial, so "
                             "set one or the other, never both")
        if self.dgi_scope not in ("shared", "prot"):
            raise ValueError(f"dgi_scope must be 'shared' or 'prot', got {self.dgi_scope!r}")
        if self.edge_center not in ("global", "per_receptor"):
            raise ValueError(f"edge_center must be 'global' or 'per_receptor', got {self.edge_center!r}")
        if self.edge_weight_mode not in ("none", "magnitude"):
            raise ValueError(f"edge_weight_mode must be 'none' or 'magnitude', got {self.edge_weight_mode!r}")
        self._proteins = D.load_npz_dict(self.protein_path)
        self._molecules = D.load_npz_dict(self.molecule_path)
        self._pca_mol = None
        self._pca_prot = None
        self._s_prot = None          # built per split in _run_models (train-fit)
        self._k_pca = 0
        self._anchor_proteins = None  # the real vectors, when nodes were swapped
        per_model = 2 * self.hidden if self.emit == "both" else self.hidden
        self.dim_out = self.n_models * per_model
        self.path = f"{self.protein_path} + {self.molecule_path}"

    def _build_model(self):
        mol_dim = next(iter(self._molecules.values())).shape[-1]
        prot_dim = next(iter(self._proteins.values())).shape[-1]
        return _SignedSage(mol_dim, prot_dim, self.hidden, self.dropout,
                           weighted=(self.edge_weight_mode != "none"),
                           pca_mol=self._pca_mol, pca_prot=self._pca_prot,
                           alpha=self.alpha, s_prot=self._s_prot,
                           conv=self.conv, heads=self.heads,
                           normalize=self.normalize_layers)

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
