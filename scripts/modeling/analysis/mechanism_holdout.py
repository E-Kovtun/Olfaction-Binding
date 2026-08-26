"""Ligand-class holdout: does a binding-refined receptor carry a transferable MECHANISM?

Hold out every odorant of a chemical class C, train the signed graph on what remains, and
ask what the resulting receptor geometry still knows about C. The graph never saw a single
class-C example, so anything it gets right transferred.

Five receptor representations, grouped by what each is allowed to know:

    raw ESM             structure only  -- sequence, sees no binding at all
    GNN+ESM             both            -- our graph, ESM node features refined by binding
    GNN one-hot         function only   -- the same graph fed one-hot receptor identity, so
                                           the geometry comes from binding alone
    GNN + PCA128(ESM)   both, stapled   -- the graph's embedding concatenated with the
                                           principal scores of raw ESM, each block scaled to
                                           equal weight. Answers what no single-source row
                                           can: did refinement DROP something ESM had?
    retained profile    function, raw   -- the receptor's own measured responses to the
                                           odorants that stayed. No model in it at all, so
                                           it is the floor: whatever the graph scores above
                                           it is what refinement added, and whatever it does
                                           not was already lying in the response table.

The last two are DERIVED -- computed from the stored embeddings after the fact, with no
training (`--derive`, automatic after a fresh run). The first three cost a graph each.

Readouts, deliberately different in kind:

    kNN     the original leave-one-out neighbour readout. DEPRECATED -- k, the coarse label
            and the smoothing are all moving parts. Computed and stored so the deprecated
            panel still renders; not a number to quote.
    GEOMETRY -- three second-order measures on the same (embedding, class-profile) pair, of
            increasing strictness. None has a head or predicts anything, which is why they
            are the metrics of record; `tab:t6` reports RSA.
              rsa         Mantel/Spearman over similarity off-diagonals -- neighbour ORDER;
                          invariant to any monotone map of the similarities.
              cca         mean canonical correlation -- is the profile a linear function of
                          the embedding at all; invariant to any invertible linear map.
              procrustes  1 - disparity after optimal rotation/scale -- same SHAPE; the
                          strictest, invariant only to rigid motion.
            A claim surviving all three does not depend on which notion of "aligned" one
            prefers. CCA and Procrustes reduce both sides to a common small rank; see
            GEOMETRY_K_CAP for why "small" is calibrated, not guessed.
    OOD     the pipeline's own boosting head (`orbind.baselines.train_boost`, the same
            400-tree XGBoost the paper's tables use) fitted on `[receptor || molecule]` for
            every pair whose odorant is OUTSIDE the class and scored on the class pairs.
            Prediction, in units another method can be scored in -- a CONTROLLED OOD, where
            the pipeline's cold-molecule split hides a random 20% of odorants and this hides
            a chemistry. Optionally the same masks for a competitor (`--hladis`).

ISOLATION, and why classes are weighted. A holdout only tests mechanism transfer if it
really removed the class. Two ways it does not: `struct_leak` -- the mean best Tanimoto from
a class member to a retained odorant, i.e. a structural twin the graph did see -- and
`func_redund` -- the correlation across receptors between the class target and general
responsiveness, i.e. the class was never functionally distinct. `trust = (1 - struct_leak) *
(1 - func_redund)` weights the notebook's final per-representation number; multiplicative
because either leak alone voids the test. Both are properties of the DATASET, not of any
model, so weighting by them cannot favour a representation.

Agreement between RSA and OOD is the point: RSA has no moving parts and OOD has both a head
and hyperparameters, so a conclusion surviving in both does not live in either one's.

A run made before those columns existed does not have to be repeated: `--backfill` adds them
to an existing directory from its own `embeddings.npz`, with no training.

This script does all the computing and writes small artifacts;
`notebooks/graph/mechanism_holdout/mechanism_holdout.ipynb` only reads and draws them.

    .venv/bin/python scripts/modeling/analysis/mechanism_holdout.py --dataset all

Artifacts land in `results/mechanism_holdout/<dataset>/`:

    readouts.csv   one row per (class, model, seed): knn, rsa, cca, procrustes
    model_nulls.csv one row per (class, model): each geometry's null and its spread.
                   Per model because the spread depends on the representation and the
                   notebook's headline number divides by it
    nulls.csv      one row per class: the null / baseline line for each readout, each
                   geometry's null SPREAD (`*_null_sd`, what makes a score
                   dimensionless), and the two isolation controls `struct_leak` /
                   `func_redund` with the class weight `trust` they combine into
    ood.csv        one row per (class, model, seed): the metric family for the task,
                   including the `naive (train mean)` and `receptor tuning` references
    panels.npz     the flagship class's actual-vs-three-models reconstruction
    overview.npz   the response matrix with the class pulled into a block, for the heat map
    molecules.csv  inchikey, smiles and class membership, for the RDKit grids
    embeddings.npz the receptor embeddings themselves, per (class, model, seed), plus each
                   class's target matrix and receptor order -- float16. This is what makes a
                   NEW second-order metric free later: no retraining, just read and score.
                   Skip with --no-embeddings.
    meta.json      config, shapes, timing, what `knn` means for this dataset

One process, one GPU, and the three datasets share nothing -- separate data, separate output
directories. So the way onto several GPUs is one dataset per process
(`--dataset cc`, pinned with CUDA_VISIBLE_DEVICES); there is nothing to merge afterwards.

A note on the seeds: they vary ONLY the graph's weight initialisation (torch.manual_seed
before _train_one). The class split is fixed by SMARTS and the receptor set is fixed by the
data, so the per-seed CI is training-run variance, NOT sampling error -- raw ESM has no seed
dependence at all and its geometric CI is zero by construction. The dominant uncertainty here
is the receptor sample (24 for HC, 50 for CC), which seeds do not touch; embeddings.npz is
there so a receptor bootstrap can be added post-hoc.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

ROOT = _root
DATA = ROOT / "data"

# Mechanistically-meaningful classes. M2OR gets thiol (Cu-coordination) because it has
# enough of them; the insect panels are complete matrices where thiols are absent.
SMARTS_ALL = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "thiol":           "[#16X2H]",
    "alcohol":         "[#6][OX2H]",
    "aldehyde":        "[CX3H1](=O)[#6]",
    "ketone":          "[#6][CX3](=O)[#6]",
    "ester":           "[#6][CX3](=O)[OX2H0][#6]",
    "aromatic":        "c1ccccc1",
}

# `kind` is what actually differs between the datasets:
#   sparse_binary       M2OR -- assayed pairs only, 0/1 label, receptor-level class label
#   complete_continuous CC/HC -- every receptor x odorant measured, z-scored response, so the
#                       target can be residualised against each receptor's own tuning
DATASETS = {
    "m2or": dict(
        kind="sparse_binary", task="classification",
        prot="embeddings/proteins/esm1b_650m_mean.npz",
        mol="embeddings/molecules/gin_supervised_contextpred_all_m2or.npz",
        classes=["carboxylic_acid", "thiol", "aldehyde", "ester"],
        k_nn=10, min_members=5, hladis=(10000, 6000, 500),
    ),
    "cc": dict(
        kind="complete_continuous", task="regression",
        prot="embeddings/proteins/esm1b_650m_mean_cc.npz",
        mol="embeddings/molecules/gin_supervised_contextpred_cc.npz",
        classes=["carboxylic_acid", "alcohol", "aldehyde", "ketone", "ester", "aromatic"],
        k_nn=5, min_members=6, hladis=(2000, 1200, 100),
    ),
    "hc": dict(
        kind="complete_continuous", task="regression",
        prot="embeddings/proteins/esm1b_650m_mean_hc.npz",
        mol="embeddings/molecules/gin_supervised_contextpred_hc.npz",
        classes=["carboxylic_acid", "alcohol", "aldehyde", "ketone", "ester", "aromatic"],
        k_nn=5, min_members=6, hladis=(2000, 1200, 100),
    ),
}

# The graph is the full-coverage signed graph. q99 was dropped for this analysis: with a
# class removed, a quantile cut on top of that removal is a second moving part nobody wants
# to defend. The pipeline elsewhere stays titular q99.
VARIANT = dict(q=0.0, criterion="coverage", k_mode="coverage_quantile")
MODEL_ESM = "raw ESM"
MODEL_GNN = "GNN+ESM full (q=0)"
MODEL_ONEHOT = "GNN one-hot full (q=0)"
MODEL_CONCAT = "GNN + PCA128(ESM)"
MODEL_PROFILE = "retained profile"
DERIVED = (MODEL_CONCAT, MODEL_PROFILE)
PCA_DIM = 128


# --------------------------------------------------------------------------- data

def load_dataset(ds: str):
    """-> pairs frame, receptor list, odorant list, response matrix, {inchikey: smiles}.

    The matrix is receptors x odorants with NaN where nothing was measured. For the insects
    it is dense by construction; for M2OR it is mostly NaN, which is exactly the confound
    the insect stands were fetched to remove.
    """
    if ds == "m2or":
        from orbind.regimes import full_full_pairs
        p = full_full_pairs()
        agg = "max"          # a receptor binds a molecule if any assay says so
    else:
        from orbind.regimes_ofm import ofm_pairs
        p = ofm_pairs(ds)
        agg = "mean"         # replicate responses average
    recs = sorted(p["receptor"].unique())
    ods = sorted(p["inchikey"].unique())
    r_i = {r: i for i, r in enumerate(recs)}
    o_i = {o: j for j, o in enumerate(ods)}
    g = p.groupby(["receptor", "inchikey"], as_index=False)["label"].agg(agg)
    R = np.full((len(recs), len(ods)), np.nan, np.float32)
    R[g["receptor"].map(r_i), g["inchikey"].map(o_i)] = g["label"].to_numpy()
    smi = p.drop_duplicates("inchikey").set_index("inchikey")["smiles"].to_dict()
    return p, recs, ods, R, smi


def class_members(ods, smi, names, min_members):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mols = {o: Chem.MolFromSmiles(smi[o]) for o in ods}
    out = {}
    for cname in names:
        q = Chem.MolFromSmarts(SMARTS_ALL[cname])
        members = {o for o, m in mols.items() if m is not None and m.HasSubstructMatch(q)}
        if len(members) >= min_members:
            out[cname] = members
        else:
            print(f"  {cname:16} skipped: only {len(members)} molecules", flush=True)
    return out


# --------------------------------------------------------------------------- the graph

def train_refined(spec, p, seed, drop_iks, feat, epochs, device):
    """Signed graph over every pair whose ODORANT is not in `drop_iks`.

    Returns ({receptor: refined vector}, {receptor: raw node-feature vector}). `feat="onehot"`
    swaps the ESM node features for a one-hot identity over exactly the receptors ESM covers,
    so the universe and the edges are identical and only the structural prior is removed.
    """
    import torch
    from orbind.gnn_extractor import (GnnSignedExtractor, _build_universe, _mp_edges,
                                       _edge_index_dict, _train_one)
    ext = GnnSignedExtractor(
        name="cls", protein_path=str(DATA / spec["prot"]), molecule_path=str(DATA / spec["mol"]),
        emit="prot", n_models=1, hidden=256, edge_threshold=0.0, task=spec["task"],
        epochs=epochs, **VARIANT)
    if feat == "onehot":
        recs = sorted(ext._proteins)
        eye = np.eye(len(recs), dtype=np.float32)
        ext._proteins = {r: eye[i] for i, r in enumerate(recs)}
    idx = np.where((~p["inchikey"].isin(drop_iks)).to_numpy())[0]
    idx = idx[ext.covered(p, idx)]
    m2i, p2i, xm, xp = _build_universe(p, idx, ext._proteins, ext._molecules)
    pos, neg = _mp_edges(p, idx, m2i, p2i, ext.q, ext.criterion,
                         ext.edge_threshold, ext.k_mode, ext.task)
    pe, ne = _edge_index_dict(pos, neg)
    sub = p.iloc[idx]
    torch.manual_seed(seed)
    _, zp, _ = _train_one(ext._build_model, xm, xp, pe, ne,
                          sub["inchikey"].map(m2i).to_numpy(),
                          sub["receptor"].map(p2i).to_numpy(),
                          sub["label"].to_numpy(np.float32), ext._hp(seed), device, None)
    return {r: zp[i] for r, i in p2i.items()}, {r: xp[i].numpy() for r, i in p2i.items()}


# --------------------------------------------------------------------------- readouts

def emb_sim(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return Xn @ Xn.T


def knn_neighbours(X, k):
    S = emb_sim(X).copy()
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, min(k, len(X) - 1), axis=1)[:, :k]


def knn_pred(X, target, k):
    nn = knn_neighbours(X, k)
    return np.stack([np.nanmean(target[nn[i]], axis=0) for i in range(len(target))])


def col_corr(actual, pred):
    """Mean per-odorant correlation between actual and neighbour-predicted responses."""
    cs = []
    for o in range(actual.shape[1]):
        a, q = actual[:, o], pred[:, o]
        if a.std() > 1e-8 and q.std() > 1e-8:
            cs.append(np.corrcoef(a, q)[0, 1])
    return float(np.mean(cs)) if cs else np.nan


def knn_auroc(X, y, k):
    from sklearn.metrics import roc_auc_score
    nn = knn_neighbours(X, k)
    score = np.array([y[row].mean() for row in nn])
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else np.nan


def rsa(X, M):
    """Mantel/RSA: Spearman between the off-diagonals of embedding similarity and of
    mean-centred class-profile similarity. Parameter-free -- no k, no smoothing."""
    from scipy.stats import spearmanr
    Mc = M - M.mean(1, keepdims=True)
    Mn = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9)
    B = Mn @ Mn.T
    iu = np.triu_indices(len(X), 1)
    return float(spearmanr(B[iu], emb_sim(X)[iu]).correlation)


def _pca(A, k):
    """Mean-centre and project onto the top-k principal directions."""
    A = np.asarray(A, np.float64)
    A = A - A.mean(0)
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] * S[:k]


# Rank both sides are reduced to before CCA / Procrustes. Small on purpose: CCA's
# permutation null climbs with k and swallows the signal. Measured on CC/carboxylic_acid
# (50 receptors, 60 permutations, GNN vs raw ESM vs null):
#
#     k  |  CCA gnn / esm / null   |  Proc gnn / esm / null
#     2  |  0.510 / 0.166 / 0.165  |  0.209 / 0.050 / 0.024
#     3  |  0.399 / 0.224 / 0.203  |  0.207 / 0.051 / 0.027
#     5  |  0.399 / 0.363 / 0.266  |  0.208 / 0.095 / 0.034
#     10 |  0.395 / 0.425 / 0.391  |  0.209 / 0.097 / 0.045
#
# At k=10 CCA is pure noise -- the null is 0.391 and the graph scores BELOW it. Procrustes
# is insensitive to k (0.207-0.209 throughout) and keeps a flat null, so one small cap
# serves both. HC has 24 receptors, where the inflation is worse still.
GEOMETRY_K_CAP = 3


def geometry_k(X, M, cap=None):
    """Common rank for CCA / Procrustes.

    Both compare two configurations of the SAME receptors, so both need the two sides
    reduced to a shared, non-degenerate number of columns: ESM is 1280-d over 24-50
    receptors. Bounded by the cap above and by n-1 / the class size."""
    return int(max(2, min(cap or GEOMETRY_K_CAP, len(X) - 1, M.shape[1], X.shape[1])))


def cca(X, M, k=None):
    """Mean of the canonical correlations between the embedding and the class profile.

    Computed exactly, via QR of each side and the SVD of their cross-product -- the
    singular values ARE the canonical correlations. Unlike RSA this is invariant to any
    invertible linear map of either side, so it asks a strictly weaker question: is the
    class profile a linear function of the embedding at all, ignoring how distances are
    arranged. A high CCA with a flat RSA means the information is present but not laid
    out geometrically."""
    k = k or geometry_k(X, M)
    A, B = _pca(X, k), _pca(M, k)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))


def procrustes(X, M, k=None):
    """1 - Procrustes disparity after optimal rotation/scaling of the two configurations.

    The strictest of the three: it allows only a rigid map (rotation, reflection,
    uniform scale), so it asks whether the two clouds have the same SHAPE, not merely
    the same neighbour ordering (RSA) or a shared linear subspace (CCA). Reported as
    1 - disparity so that, like the other two, larger is better and 0 is no alignment."""
    k = k or geometry_k(X, M)
    return _procrustes_pair(_pca(X, k), _pca(M, k))


def _procrustes_pair(A, B):
    from scipy.spatial import procrustes as _proc
    if min(A.shape) < 2 or np.allclose(A.std(), 0) or np.allclose(B.std(), 0):
        return np.nan
    try:
        _, _, disparity = _proc(A, B)
    except ValueError:
        return np.nan
    return float(1.0 - disparity)


# The three geometry readouts, all second-order, all scored on the same (X, M) and all
# nulled the same way. They are deliberately of increasing strictness: RSA (neighbour
# ordering) -> CCA (shared linear subspace) -> Procrustes (same shape). A claim that
# survives all three does not depend on which notion of "aligned" one happens to prefer.
GEOMETRY = {"rsa": rsa, "cca": cca, "procrustes": procrustes}


def permuted(fn, X, n, seed=0):
    """Row-permutation null: shuffle the embedding rows, the alignment dies."""
    return permuted_stats(fn, X, n, seed)[0]


def permuted_stats(fn, X, n, seed=0):
    """(mean, sd) of the null. The sd is what makes a score dimensionless: a measure is
    only as good as its own noise floor, and the three geometries have different ones."""
    rng = np.random.default_rng(seed)
    v = np.array([fn(X[rng.permutation(len(X))]) for _ in range(n)], float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0


# --------------------------------------------------------------------------- isolation

def struct_leak(smi, ods, iks):
    """Mean over class members of the best Tanimoto to a NON-member odorant (ECFP4).

    How much of the held-out class survives in training as a near-analogue. 1.0 would mean
    every member has a structural twin the graph did get to see, and the holdout tests
    nothing. Returns NaN without rdkit rather than pretending the control ran.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, DataStructs
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return float("nan")

    def fp(ik):
        m = Chem.MolFromSmiles(smi.get(ik, ""))
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m is not None else None

    mem = [f for f in (fp(o) for o in ods if o in iks) if f is not None]
    oth = [f for f in (fp(o) for o in ods if o not in iks) if f is not None]
    if not mem or not oth:
        return float("nan")
    return float(np.mean([max(DataStructs.BulkTanimotoSimilarity(f, oth)) for f in mem]))


def func_redund(R, recs, ods, order, iks, target):
    """Correlation across receptors between the class target and general responsiveness.

    The functional twin of struct_leak: if "responds to this class" is just "responds to
    everything", the class carries no separate mechanism and a readout can be satisfied by
    general tuning. Measured against `target` -- the very matrix the readouts score, already
    residualised on the insect matrices -- so it never re-charges a leak the target removed.
    Clipped at 0: an anticorrelated class is not MORE trustworthy.
    """
    o_i = {o: j for j, o in enumerate(ods)}
    r_i = {r: i for i, r in enumerate(recs)}
    keep = [j for j, o in enumerate(ods) if o not in iks]
    if not keep or len(order) < 3:
        return float("nan")
    with np.errstate(invalid="ignore"):
        a = np.nanmean(np.asarray(target, float).reshape(len(order), -1), axis=1)
        b = np.nanmean(R[np.ix_([r_i[r] for r in order], keep)], axis=1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.nanstd(a[ok]) < 1e-9 or np.nanstd(b[ok]) < 1e-9:
        return float("nan")
    return float(max(0.0, np.corrcoef(a[ok], b[ok])[0, 1]))


def pca_esm(Xe, dim=PCA_DIM):
    """Principal scores of the raw ESM matrix, at most `dim` of them.

    Fitted on this holdout's own training universe: the split removes ODORANTS, not
    receptors, so every receptor here is a training receptor. It is fitted on sequence
    alone and never sees a response, so it cannot carry the held-out class either way.
    The rank is capped by n-1, which binds hard on the insects (24 and 50 receptors).
    """
    k = int(min(dim, len(Xe) - 1, np.asarray(Xe).shape[1]))
    return _pca(Xe, k), k


def block_concat(*blocks):
    """Concatenate representations after putting each block on the same scale.

    Without this the wider-variance block decides the geometry outright: ESM principal
    scores carry their raw singular values while the graph's 256 dims are whatever training
    left them at. Each block is centred and divided by its own mean row norm, so the
    concatenation is a genuine 50/50 and not a disguised single source.
    """
    out = []
    for B in blocks:
        B = np.asarray(B, np.float64)
        B = B - B.mean(0)
        out.append(B / (np.sqrt((B ** 2).sum(1).mean()) + 1e-12))
    return np.hstack(out)


def retained_profile(R, recs, ods, order, iks):
    """Function only, with no model in it at all: the receptor's OWN measured responses to
    the odorants that stayed in training.

    The point of the pair (this, GNN one-hot): both know function and nothing else, but this
    one involves no learning, so it says how much of the transfer was already sitting in the
    raw response table. On the sparse matrix an untested pair reads 0 -- the same thing the
    graph sees, an absent edge, not an imputed one.

    Rows are centred, and that matters: a receptor's mean over the retained odorants is
    EXACTLY the quantity the insect target was residualised on, so leaving it in would make
    the baseline's score turn on that convention instead of on function. Measured both ways
    on HC/carboxylic_acid, centring raises the baseline (RSA +0.147 -> +0.272), so this is
    the harder floor for our own method to clear, not the softer one.
    """
    r_i = {r: i for i, r in enumerate(recs)}
    keep = [j for j, o in enumerate(ods) if o not in iks]
    if not keep:
        return None
    X = np.nan_to_num(R[np.ix_([r_i[r] for r in order], keep)], nan=0.0)
    return X - X.mean(1, keepdims=True)


def trust(sl, fr):
    """One weight per class, in [0, 1]: how isolated the holdout actually was.

    Multiplicative because the two leaks are independent routes to the same failure -- a
    class needs BOTH a structural gap and a functional one to be a real test. A missing
    control (NaN) drops out of the product rather than silently scoring 1.
    """
    parts = [1.0 - v for v in (sl, fr) if np.isfinite(v)]
    return float(np.clip(np.prod(parts), 0.0, 1.0)) if parts else float("nan")


def geometry_nulls(X, M, n, seed=0):
    """(mean, sd) of the permutation null for all three geometries, without recomputing
    anything a permutation cannot change.

    Permuting the rows of X permutes the DERIVED objects too, exactly:

        emb_sim(X[p]) == emb_sim(X)[p][:, p]      a cosine matrix is row/col-permuted
        _pca(X[p], k) == _pca(X, k)[p]            column-centring and the SVD scores are
                                                  row-equivariant

    So the similarity matrix, its ranks and the principal scores are built once and indexed
    n times instead of rebuilt n times. Same numbers as calling the measures on shuffled
    inputs; the naive path re-derived a 937x937 cosine matrix and a 937x1280 SVD for every
    one of 100 permutations, per model, per class. Measured at M2OR's size (n=937, 100
    permutations, all three measures): 338s naive, 5s here -- 113 minutes against 2 for a
    whole dataset's model_nulls.csv.
    """
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(X)) for _ in range(n)]
    iu = np.triu_indices(len(X), 1)
    out = {}

    # --- RSA: rank both sides once; Spearman becomes Pearson over a permuted gather
    Mc = M - M.mean(1, keepdims=True)
    Mn = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9)
    b = rankdata((Mn @ Mn.T)[iu])
    S = emb_sim(X)
    RS = np.zeros_like(S)
    RS[iu] = rankdata(S[iu])
    RS = RS + RS.T                       # symmetric; the diagonal never enters iu
    b = b - b.mean()
    bn = np.linalg.norm(b) + 1e-12
    vals = []
    for p in perms:
        a = RS[np.ix_(p, p)][iu]
        a = a - a.mean()
        vals.append(float(a @ b / ((np.linalg.norm(a) + 1e-12) * bn)))
    out["rsa"] = _mean_sd(vals)

    # --- CCA / Procrustes: both sides reduced once, then only the row order moves
    k = geometry_k(X, M)
    A, B = _pca(X, k), _pca(M, k)
    Qb, _ = np.linalg.qr(B)
    cvals, pvals = [], []
    for p in perms:
        Ap = A[p]
        Qa, _ = np.linalg.qr(Ap)
        cvals.append(float(np.mean(np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1))))
        pvals.append(_procrustes_pair(Ap, B))
    out["cca"] = _mean_sd(cvals)
    out["procrustes"] = _mean_sd(pvals)
    return out


def _mean_sd(v):
    v = np.asarray(v, float)
    return (float(np.nanmean(v)),
            float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0)


# --------------------------------------------------------------------------- OOD boost

def ood_masks(p, rec_vec, mol_emb, iks):
    ok = p["receptor"].isin(rec_vec).to_numpy() & p["inchikey"].isin(mol_emb).to_numpy()
    inC = p["inchikey"].isin(iks).to_numpy()
    return ok & ~inC, ok & inC


def _score(task, y, pred):
    from orbind.dataset import METRICS
    try:
        return {k: float(v) for k, v in METRICS[task](y, pred).items()}
    except ValueError:
        # a held-out class with only one label present has no defined AUROC; say so rather
        # than crash the whole sweep on one degenerate class
        return {}


def ood_boost(p, rec_vec, mol_emb, iks, seed, task):
    """Fit on retained-chemistry pairs, score the held-out class. The receptor vector was
    itself built without the class, so nothing in the path has seen that chemistry."""
    from orbind.baselines import train_boost
    tr_mask, te_mask = ood_masks(p, rec_vec, mol_emb, iks)

    def feats(mask):
        sub = p[mask]
        Xr = np.stack([rec_vec[r] for r in sub["receptor"]])
        Xm = np.stack([mol_emb[k] for k in sub["inchikey"]])
        return np.concatenate([Xr, Xm], 1).astype(np.float32), sub["label"].to_numpy(np.float32)

    Xtr, ytr = feats(tr_mask)
    Xte, yte = feats(te_mask)
    pred = train_boost(Xtr, ytr, Xte, seed=seed, task=task)
    m = _score(task, yte, pred)
    m.update(n_train=int(len(ytr)), n_test=int(len(yte)))
    return m


def ood_references(p, rec_vec, mol_emb, iks, task):
    """The two lines that decide how to read every OOD number.

    naive   -- the training mean. Under regression R2 is measured against the TEST mean, so
               this is the honest zero and it is NOT zero when a coherent class sits
               off-centre. Under classification the same constant is the prevalence.
    tuning  -- each receptor's mean response over the RETAINED odorants. A model can score
               well on a class merely by knowing which receptors are promiscuous; anything
               that does not clear this line has learned nothing chemistry-specific.
    """
    tr_mask, te_mask = ood_masks(p, rec_vec, mol_emb, iks)
    tr, te = p[tr_mask], p[te_mask]
    yte = te["label"].to_numpy(np.float32)
    gm = float(tr["label"].mean())
    naive = _score(task, yte, np.full(len(yte), gm, np.float32))
    tune = te["receptor"].map(tr.groupby("receptor")["label"].mean()).fillna(gm).to_numpy(np.float32)
    out = _score(task, yte, tune)
    n = dict(n_train=int(len(tr)), n_test=int(len(yte)))
    return dict(naive, **n), dict(out, **n)


def ood_hladis(spec, ds, p, mol_emb, iks, seed, budget):
    """The same masks, a competitor's features. This is what section 5 buys over RSA: any
    method producing a pair-level prediction can be dropped onto exactly these masks."""
    from orbind.hladis_extractor import HladisExtractor
    from orbind.baselines import train_boost
    max_steps, warmup, every = budget
    inC = p["inchikey"].isin(iks).to_numpy()
    tr_all, te_idx = np.where(~inC)[0], np.where(inC)[0]
    perm = np.random.default_rng(seed).permutation(len(tr_all))
    n_val = max(1, int(0.1 * len(tr_all)))
    va_idx, tr_idx = tr_all[perm[:n_val]], tr_all[perm[n_val:]]
    ext = HladisExtractor(name="cls", protein_path=str(DATA / spec["prot"]), n_models=1,
                          max_steps=max_steps, warmup_steps=warmup, eval_every=every)
    ext.task = spec["task"]
    Xtr, _Xva, Xte = ext.fit_transform(p, tr_idx, va_idx, te_idx, seed)
    ik = p["inchikey"].to_numpy()
    y = p["label"].to_numpy(np.float32)
    Mtr = np.stack([mol_emb[k] for k in ik[tr_idx]])
    Mte = np.stack([mol_emb[k] for k in ik[te_idx]])
    pred = train_boost(np.concatenate([Xtr, Mtr], 1), y[tr_idx],
                       np.concatenate([Xte, Mte], 1), seed=seed, task=spec["task"])
    m = _score(spec["task"], y[te_idx], pred)
    m.update(n_train=int(len(tr_idx)), n_test=int(len(te_idx)))
    return m


# --------------------------------------------------------------------------- targets

def class_targets(kind, p, R, recs, ods, order, iks):
    """What the readouts score against, per dataset kind.

    complete_continuous: the class response block minus each receptor's mean over the
        RETAINED odorants -- its general responsiveness. A model that only knows tuning
        cannot predict that residual. Both readouts use it.
    sparse_binary: a receptor-level 0/1 "responds to the class" label for kNN (AUROC), and
        the per-class-molecule 0/1 profile for RSA. Receptors never tested on the class are
        dropped -- there is no ground truth for them.
    """
    o_i = {o: j for j, o in enumerate(ods)}
    r_i = {r: i for i, r in enumerate(recs)}
    cols = [o_i[o] for o in ods if o in iks]
    keep = [j for j in range(len(ods)) if j not in cols]
    if kind == "complete_continuous":
        ri = [r_i[r] for r in order]
        Y = np.nan_to_num(R[np.ix_(ri, cols)])
        base = np.nan_to_num(np.nanmean(R[np.ix_(ri, keep)], axis=1, keepdims=True))
        resid = Y - base
        return dict(order=order, knn_target=resid, rsa_target=resid, Y=Y, cols=cols)
    sub = R[np.ix_([r_i[r] for r in order], cols)]
    tested = np.isfinite(sub).any(1)
    order2 = [r for r, t in zip(order, tested) if t]
    P = np.nan_to_num(sub[tested])
    y = (np.nanmax(np.where(np.isfinite(sub[tested]), sub[tested], np.nan), axis=1) >= 1).astype(int)
    return dict(order=order2, knn_target=y, rsa_target=P, Y=sub[tested], cols=cols)


def knn_of(kind, X, tgt, k):
    return (knn_auroc(X, tgt, k) if kind == "sparse_binary"
            else col_corr(tgt, knn_pred(X, tgt, k)))


def knn_baseline(kind, p, R, recs, ods, tg, iks, k, n_perm):
    """The line a kNN bar has to clear. On M2OR that is the tuning-only predictor (each
    receptor's overall non-class hit rate); on the insects a row-permutation null."""
    if kind != "sparse_binary":
        return permuted(lambda Xs: col_corr(tg["knn_target"], knn_pred(Xs, tg["knn_target"], k)),
                        tg["_X_esm"], n_perm)
    from sklearn.metrics import roc_auc_score
    hit = p[~p["inchikey"].isin(iks)].groupby("receptor")["label"].mean()
    hr = hit.reindex(tg["order"]).fillna(hit.mean()).to_numpy()
    y = tg["knn_target"]
    return float(roc_auc_score(y, hr)) if 0 < y.sum() < len(y) else np.nan


# --------------------------------------------------------------------------- panels

def build_panels(kind, R, recs, ods, tg, ref_v, fun_v, raw_v, k, max_rows=70):
    """actual | GNN+ESM | GNN one-hot | ESM, reconstructed from each embedding's neighbours.

    On a sparse matrix every model panel is masked to `actual`'s measured support, so each
    coloured cell has a ground-truth counterpart and all four panels share one shape; the
    rows are then cut to the `max_rows` best-responding receptors, because 900+ rows of a
    mostly-untested matrix is not a figure anyone can read. The insect matrices are complete
    and small (50 / 24 receptors), so they keep every row.
    """
    order = tg["order"]
    A = tg["Y"].astype(np.float32)
    P = {}
    for name, vec in [("gnn", ref_v), ("onehot", fun_v), ("esm", raw_v)]:
        X = np.stack([vec[r] for r in order])
        nn = knn_neighbours(X, k)
        P[name] = np.stack([np.nanmean(A[nn[i]], axis=0) for i in range(len(order))])
    if kind == "sparse_binary":
        mask = np.isfinite(A)
        for name in P:
            P[name] = np.where(mask, P[name], np.nan)
    with np.errstate(invalid="ignore"):
        strength = np.nan_to_num(np.nanmean(np.where(np.isfinite(A), A, np.nan), axis=1),
                                 nan=-1e9)
    row_order = np.argsort(-strength)
    if kind == "sparse_binary" and max_rows:
        row_order = row_order[:max_rows]
    return dict(actual=A, gnn=P["gnn"], onehot=P["onehot"], esm=P["esm"],
                row_order=row_order.astype(np.int32))


def build_overview(kind, R, recs, ods, iks, seed=0):
    """The response matrix with the held-out class pulled into a block on the left and the
    receptors sorted by their class response -- the picture that shows, by eye, whether the
    class reorganises the receptors (what funcRedund measures numerically)."""
    o_i = {o: j for j, o in enumerate(ods)}
    ac = [o_i[o] for o in ods if o in iks]
    rest = [j for j in range(len(ods)) if j not in ac]
    if kind == "sparse_binary":
        rng = np.random.default_rng(seed)
        rest = list(rng.choice(rest, size=min(len(ac), len(rest)), replace=False))
        cols = ac + rest
        measured = np.isfinite(R[:, cols]).sum(1)
        rows = np.argsort(-measured)[:70]
    else:
        cols = ac + rest
        rows = np.argsort(-np.nan_to_num(np.nanmean(R[:, ac], axis=1)))
    return dict(M=R[np.ix_(rows, cols)].astype(np.float32),
                n_class=np.int32(len(ac)), n_rows=np.int32(len(rows)))


# --------------------------------------------------------------------------- driver

def run_dataset(ds, args):
    import torch
    from orbind.dataset import load_npz_dict
    spec = DATASETS[ds]
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    out = pathlib.Path(args.out) / ds
    out.mkdir(parents=True, exist_ok=True)

    p, recs, ods, R, smi = load_dataset(ds)
    print(f"\n=== {ds.upper()} === {len(recs)} receptors x {len(ods)} odorants "
          f"| {len(p)} pairs | device {device}", flush=True)
    names = args.classes or spec["classes"]
    classes = class_members(ods, smi, names, spec["min_members"])
    mol_emb = load_npz_dict(str(DATA / spec["mol"]))
    k = spec["k_nn"]

    readouts, nulls, ood, panels_saved = [], [], [], False
    emb_store = {}          # everything a future second-order metric would need
    for cname, iks in classes.items():
        c0 = time.time()
        # the raw-ESM reference embedding also fixes the receptor universe for this class
        ref0, raw0 = train_refined(spec, p, args.seeds[0], iks, "esm", args.epochs, device)
        order0 = [r for r in recs if r in ref0]
        tg = class_targets(spec["kind"], p, R, recs, ods, order0, iks)
        order = tg["order"]
        if len(order) < 5:
            print(f"  {cname:16} skipped: {len(order)} usable receptors", flush=True)
            continue
        if spec["kind"] == "sparse_binary" and not 0 < int(tg["knn_target"].sum()) < len(order):
            print(f"  {cname:16} skipped: degenerate class label", flush=True)
            continue

        Xe = np.stack([raw0[r] for r in order])
        tg["_X_esm"] = Xe
        M = tg["rsa_target"]
        gk = geometry_k(Xe, M)
        readouts.append(dict(cls=cname, model=MODEL_ESM, feat="-", seed="-",
                             knn=knn_of(spec["kind"], Xe, tg["knn_target"], k),
                             **{g: fn(Xe, M) for g, fn in GEOMETRY.items()}))
        gnull = geometry_nulls(Xe, M, args.n_perm)
        sl = struct_leak(smi, ods, iks)
        fr = func_redund(R, recs, ods, order, iks, tg["rsa_target"])
        nulls.append(dict(
            cls=cname, n_receptors=len(order), n_molecules=len(iks), geometry_k=gk,
            knn_baseline=knn_baseline(spec["kind"], p, R, recs, ods, tg, iks, k, args.n_perm),
            struct_leak=sl, func_redund=fr, trust=trust(sl, fr),
            **{f"{g}_null": m for g, (m, _) in gnull.items()},
            **{f"{g}_null_sd": sd for g, (_, sd) in gnull.items()}))
        if args.embeddings:
            # str_, not object: an object array would force allow_pickle=True on every read
            emb_store[f"order__{cname}"] = np.array(order, dtype=np.str_)
            emb_store[f"target__{cname}"] = M.astype(np.float32)
            emb_store[f"emb__{cname}__esm__0"] = Xe.astype(np.float16)

        if args.ood:
            nv, tn = ood_references(p, raw0, mol_emb, iks, spec["task"])
            ood.append(dict(cls=cname, model="naive (train mean)", seed="-", **nv))
            ood.append(dict(cls=cname, model="receptor tuning", seed="-", **tn))

        keep = {}
        for feat, model in [("esm", MODEL_GNN), ("onehot", MODEL_ONEHOT)]:
            for seed in args.seeds:
                ref, raw = (ref0, raw0) if (feat == "esm" and seed == args.seeds[0]) else \
                    train_refined(spec, p, seed, iks, feat, args.epochs, device)
                X = np.stack([ref[r] for r in order])
                readouts.append(dict(cls=cname, model=model, feat=feat, seed=seed,
                                     knn=knn_of(spec["kind"], X, tg["knn_target"], k),
                                     **{g: fn(X, M) for g, fn in GEOMETRY.items()}))
                if args.embeddings:
                    emb_store[f"emb__{cname}__{feat}__{seed}"] = X.astype(np.float16)
                if args.ood:
                    if feat == "esm":
                        # raw ESM vectors are identical every seed; only the head's seed
                        # moves, which is what makes its CI comparable to the graphs'
                        ood.append(dict(cls=cname, model=MODEL_ESM, seed=seed,
                                        **ood_boost(p, raw, mol_emb, iks, seed, spec["task"])))
                    ood.append(dict(cls=cname, model=model, seed=seed,
                                    **ood_boost(p, ref, mol_emb, iks, seed, spec["task"])))
                if seed == args.seeds[0]:
                    keep[feat] = ref
                r_ = readouts[-1]
                print(f"  {cname:16} {model:24} seed {seed}  knn {r_['knn']:+.3f}  "
                      f"rsa {r_['rsa']:+.3f}  cca {r_['cca']:+.3f}  "
                      f"proc {r_['procrustes']:+.3f}", flush=True)

        if args.hladis:
            for seed in args.seeds[:args.hladis_seeds]:
                m = ood_hladis(spec, ds, p, mol_emb, iks, seed, spec["hladis"])
                ood.append(dict(cls=cname, model="Hladis cls+mol", seed=seed, **m))
                print(f"  {cname:16} {'Hladis cls+mol':24} seed {seed}", flush=True)

        if cname == (args.panel_class or list(classes)[0]) and not panels_saved:
            np.savez_compressed(out / "panels.npz", cls=cname,
                                **build_panels(spec["kind"], R, recs, ods, tg,
                                                keep["esm"], keep["onehot"], raw0, k))
            np.savez_compressed(out / "overview.npz", cls=cname,
                                **build_overview(spec["kind"], R, recs, ods, iks))
            panels_saved = True
        print(f"  {cname:16} done in {time.time() - c0:.0f}s", flush=True)

    if emb_store:
        np.savez_compressed(out / "embeddings.npz", **emb_store)
    pd.DataFrame(readouts).to_csv(out / "readouts.csv", index=False)
    pd.DataFrame(nulls).to_csv(out / "nulls.csv", index=False)
    if ood:
        pd.DataFrame(ood).to_csv(out / "ood.csv", index=False)
    mol = pd.DataFrame({"inchikey": ods, "smiles": [smi[o] for o in ods]})
    for cname, iks in classes.items():
        mol[cname] = mol["inchikey"].isin(iks)
    mol.to_csv(out / "molecules.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(
        dataset=ds, kind=spec["kind"], task=spec["task"], classes=list(classes),
        n_receptors=len(recs), n_odorants=len(ods), n_pairs=int(len(p)),
        seeds=args.seeds, epochs=args.epochs, k_nn=k, n_perm=args.n_perm,
        knn_kind="auroc" if spec["kind"] == "sparse_binary" else "specificity",
        geometry=list(GEOMETRY), geometry_k_cap=GEOMETRY_K_CAP, embeddings=bool(emb_store),
        metric_family="classification" if spec["task"] == "classification" else "regression",
        panel_class=(args.panel_class or (list(classes)[0] if classes else None)),
        ood=bool(args.ood), hladis=bool(args.hladis),
        variant=VARIANT, seconds=round(time.time() - t0, 1),
    ), indent=2), encoding="utf-8")
    print(f"=== {ds.upper()} written to {out} in {time.time() - t0:.0f}s", flush=True)
    if args.auto_derive and emb_store:
        derive(ds, args)


def backfill(ds, args):
    """Add the isolation controls and the null spreads to an ALREADY-WRITTEN directory.

    Nothing here needs a trained model: `struct_leak` is chemistry, `func_redund` is the
    response matrix against the stored target, and the null spread is the same row
    permutation applied to the stored raw-ESM embedding. This is the embeddings dump paying
    for itself -- a run made before these columns existed does not have to be repeated.
    """
    spec = DATASETS[ds]
    out = pathlib.Path(args.out) / ds
    npath = out / "nulls.csv"
    if not npath.exists():
        print(f"=== {ds.upper()} skipped: no {npath}", flush=True)
        return
    null = pd.read_csv(npath)
    emb = None
    if (out / "embeddings.npz").exists():
        # allow_pickle: runs written before the order array became str_ stored it as
        # object. It is our own artifact, not foreign input.
        emb = np.load(out / "embeddings.npz", allow_pickle=True)
    else:
        print(f"  {ds}: no embeddings.npz -- func_redund and the null spreads need it, "
              f"only struct_leak can be filled", flush=True)

    p, recs, ods, R, smi = load_dataset(ds)
    names = args.classes or spec["classes"]
    classes = class_members(ods, smi, names, spec["min_members"])
    rows = []
    for _, r in null.iterrows():
        row = dict(r)
        cname = row["cls"]
        iks = classes.get(cname)
        if iks is None:
            print(f"  {cname:16} not a class of this dataset now -- left as is", flush=True)
            rows.append(row)
            continue
        row["struct_leak"] = struct_leak(smi, ods, iks)
        if emb is not None and f"order__{cname}" in emb.files:
            order = [str(x) for x in emb[f"order__{cname}"]]
            M = np.asarray(emb[f"target__{cname}"], np.float64)
            row["func_redund"] = func_redund(R, recs, ods, order, iks, M)
            Xe = np.asarray(emb[f"emb__{cname}__esm__0"], np.float64)
            for g, (m, sd) in geometry_nulls(Xe, M, args.n_perm).items():
                row[f"{g}_null_sd"] = sd
                # the stored mean is authoritative; this one only says the two agree
                if abs(m - float(row.get(f"{g}_null", m))) > 4 * sd + 1e-9:
                    print(f"  {cname:16} {g}: recomputed null {m:+.3f} disagrees with the "
                          f"stored {row[f'{g}_null']:+.3f} -- different data?", flush=True)
        row["trust"] = trust(row.get("struct_leak", np.nan), row.get("func_redund", np.nan))
        print(f"  {cname:16} struct_leak {row['struct_leak']:.3f}  "
              f"func_redund {row.get('func_redund', float('nan')):.3f}  "
              f"trust {row['trust']:.3f}", flush=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(npath, index=False)
    if emb is not None:
        emb.close()
    print(f"=== {ds.upper()} nulls.csv backfilled in place", flush=True)


def derive(ds, args):
    """Two more receptor representations, and per-model nulls, from the stored embeddings.

    Neither needs a trained model, which is the whole point:

        GNN + PCA128(ESM)   the graph's own embedding concatenated with the principal
                            scores of raw ESM. Both sources at once, explicitly, instead of
                            the graph's implicit mixing -- it answers "did refinement DROP
                            something ESM had", which no single-source row can.
        retained profile    the receptor's measured responses to the odorants that stayed.
                            Function with no learning at all: the floor the graph has to
                            clear before "the graph transferred a mechanism" means anything.

    Also writes model_nulls.csv -- the permutation null per (class, MODEL). The class-level
    null in nulls.csv is computed on raw ESM, which was fine for three representations of
    similar width but not for a 1300-column response profile: the null's spread depends on
    the representation, and section 7 divides by it.
    """
    spec = DATASETS[ds]
    out = pathlib.Path(args.out) / ds
    if not (out / "readouts.csv").exists():
        print(f"=== {ds.upper()} skipped: no {out/'readouts.csv'}", flush=True)
        return
    if not (out / "embeddings.npz").exists():
        print(f"=== {ds.upper()} skipped: no embeddings.npz -- the derived models are read "
              f"from it (re-run without --no-embeddings)", flush=True)
        return

    res = pd.read_csv(out / "readouts.csv")
    res = res[~res["model"].isin(DERIVED)]              # idempotent: drop a previous derive
    with np.load(out / "embeddings.npz", allow_pickle=True) as z:
        emb = {kk: z[kk] for kk in z.files}

    p, recs, ods, R, smi = load_dataset(ds)
    names = args.classes or spec["classes"]
    classes = class_members(ods, smi, names, spec["min_members"])
    k = spec["k_nn"]

    rows, mnulls = [], []
    for cname in [c for c in res["cls"].unique() if c in classes]:
        iks = classes[cname]
        order = [str(x) for x in emb[f"order__{cname}"]]
        M = np.asarray(emb[f"target__{cname}"], np.float64)
        tg = class_targets(spec["kind"], p, R, recs, ods, order, iks)
        if list(tg["order"]) != list(order):
            print(f"  {cname:16} receptor order moved since the run -- skipped", flush=True)
            continue
        Xe = np.asarray(emb[f"emb__{cname}__esm__0"], np.float64)
        Pe, pdim = pca_esm(Xe)

        new = {}
        for kk in sorted(emb):
            pre = f"emb__{cname}__esm__"
            if kk.startswith(pre) and kk != f"{pre}0":
                seed = kk[len(pre):]
                new[(MODEL_CONCAT, seed)] = block_concat(np.asarray(emb[kk], np.float64), Pe)
        prof = retained_profile(R, recs, ods, order, iks)
        if prof is not None:
            new[(MODEL_PROFILE, "-")] = prof

        for (model, seed), X in new.items():
            rows.append(dict(cls=cname, model=model, feat="derived", seed=seed,
                             knn=knn_of(spec["kind"], X, tg["knn_target"], k),
                             **{g: fn(X, M) for g, fn in GEOMETRY.items()}))
            emb[f"emb__{cname}__{'concat' if model == MODEL_CONCAT else 'profile'}__{seed}"]                 = X.astype(np.float16)
            print(f"  {cname:16} {model:24} seed {seed:>3}  "
                  + "  ".join(f"{g[:4]} {rows[-1][g]:+.3f}" for g in GEOMETRY), flush=True)

        # one null per (class, model): same permutation, each representation's own geometry
        reps = {MODEL_ESM: Xe, MODEL_CONCAT: new.get((MODEL_CONCAT, str(args.seeds[0]))),
                MODEL_PROFILE: new.get((MODEL_PROFILE, "-"))}
        for feat, model in (("esm", MODEL_GNN), ("onehot", MODEL_ONEHOT)):
            kk = f"emb__{cname}__{feat}__{args.seeds[0]}"
            reps[model] = np.asarray(emb[kk], np.float64) if kk in emb else None
        for model, X in reps.items():
            if X is None:
                continue
            r = dict(cls=cname, model=model, n_dim=int(np.asarray(X).shape[1]))
            for g, (m, sd) in geometry_nulls(np.asarray(X, np.float64), M, args.n_perm).items():
                r[f"{g}_null"], r[f"{g}_null_sd"] = m, sd
            mnulls.append(r)
        print(f"  {cname:16} PCA(ESM) kept {pdim} dims", flush=True)

    if not rows:
        print(f"=== {ds.upper()} nothing derived", flush=True)
        return
    pd.concat([res, pd.DataFrame(rows)], ignore_index=True).to_csv(out / "readouts.csv",
                                                                   index=False)
    pd.DataFrame(mnulls).to_csv(out / "model_nulls.csv", index=False)
    np.savez_compressed(out / "embeddings.npz", **emb)
    mp = out / "meta.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["derived"] = list(DERIVED)
    m["pca_dim"] = PCA_DIM
    mp.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"=== {ds.upper()} derived {len(rows)} rows, model_nulls.csv written", flush=True)


def main():
    global GEOMETRY_K_CAP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["all"],
                    help="m2or / cc / hc / all (default: all)")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="override the per-dataset class list")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=900, help="graph epochs (titular protocol)")
    ap.add_argument("--n-perm", type=int, default=100, help="row permutations for the nulls")
    ap.add_argument("--geometry-k", type=int, default=GEOMETRY_K_CAP,
                    help=f"rank both sides are reduced to for CCA/Procrustes "
                         f"(default {GEOMETRY_K_CAP}; raising it inflates CCA's null -- "
                         f"see the table in the source)")
    ap.add_argument("--no-ood", dest="ood", action="store_false",
                    help="skip the predictive boosting readout (RSA/kNN only)")
    ap.add_argument("--hladis", action="store_true",
                    help="also score Hladis on the same OOD masks -- EXPENSIVE, it trains a "
                         "model per (class, seed) instead of reusing an embedding")
    ap.add_argument("--hladis-seeds", type=int, default=1,
                    help="how many of --seeds to give Hladis (default 1; 5 is ~5x the cost)")
    ap.add_argument("--no-embeddings", dest="embeddings", action="store_false",
                    help="skip embeddings.npz. It is what makes a NEW second-order metric "
                         "free later (no retraining) -- float16, a few MB on the insects, "
                         "~100MB on M2OR's 900+ receptors")
    ap.add_argument("--panel-class", default=None,
                    help="class for the actual-vs-models panels (default: the first one)")
    ap.add_argument("--derive", action="store_true",
                    help="do not train: add the derived representations (GNN+PCA(ESM), "
                         "retained profile) and per-model nulls to an existing run, from "
                         "its embeddings.npz. Runs automatically after a fresh run")
    ap.add_argument("--no-derive", dest="auto_derive", action="store_false",
                    help="skip that automatic step after a fresh run")
    ap.add_argument("--backfill", action="store_true",
                    help="do not train: add struct_leak / func_redund / trust and the "
                         "*_null_sd columns to an existing run's nulls.csv, using its "
                         "embeddings.npz. For runs made before those columns existed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/mechanism_holdout")
    args = ap.parse_args()

    GEOMETRY_K_CAP = args.geometry_k
    todo = list(DATASETS) if "all" in args.dataset else args.dataset
    bad = [d for d in todo if d not in DATASETS]
    if bad:
        ap.error(f"unknown dataset(s) {bad}, have {list(DATASETS)}")
    for ds in todo:
        if args.backfill or args.derive:
            if args.backfill:
                backfill(ds, args)
            if args.derive:
                derive(ds, args)
        else:
            run_dataset(ds, args)


if __name__ == "__main__":
    main()
