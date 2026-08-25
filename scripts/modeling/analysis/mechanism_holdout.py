"""Ligand-class holdout: does a binding-refined receptor carry a transferable MECHANISM?

Hold out every odorant of a chemical class C, train the signed graph on what remains, and
ask what the resulting receptor geometry still knows about C. The graph never saw a single
class-C example, so anything it gets right transferred.

Three receptor representations throughout:

    raw ESM       structure only  -- sequence, sees no binding at all
    GNN+ESM       both            -- our graph, ESM node features refined by binding
    GNN one-hot   function only   -- the same graph fed one-hot receptor identity, so the
                                     geometry comes from binding alone

Three readouts, deliberately different in kind:

    kNN     the original leave-one-out neighbour readout. DEPRECATED -- k, the coarse label
            and the smoothing are all moving parts. Computed and stored so the deprecated
            panel still renders; not a number to quote.
    RSA     Mantel: Spearman between the off-diagonals of embedding similarity and held-out
            class-profile similarity. No head, no hyperparameters, predicts nothing. THE
            METRIC OF RECORD -- this is what `tab:t6` reports.
    OOD     the pipeline's own boosting head (`orbind.baselines.train_boost`, the same
            400-tree XGBoost the paper's tables use) fitted on `[receptor || molecule]` for
            every pair whose odorant is OUTSIDE the class and scored on the class pairs.
            Prediction, in units another method can be scored in -- a CONTROLLED OOD, where
            the pipeline's cold-molecule split hides a random 20% of odorants and this hides
            a chemistry. Optionally the same masks for a competitor (`--hladis`).

Agreement between RSA and OOD is the point: RSA has no moving parts and OOD has both a head
and hyperparameters, so a conclusion surviving in both does not live in either one's.

This script does all the computing and writes small artifacts;
`notebooks/graph/mechanism_holdout/mechanism_holdout.ipynb` only reads and draws them.

    .venv/bin/python scripts/modeling/analysis/mechanism_holdout.py --dataset all

Artifacts land in `results/mechanism_holdout/<dataset>/`:

    readouts.csv   one row per (class, model, seed): knn, rsa
    nulls.csv      one row per class: the null / baseline line for each readout
    ood.csv        one row per (class, model, seed): the metric family for the task,
                   including the `naive (train mean)` and `receptor tuning` references
    panels.npz     the flagship class's actual-vs-three-models reconstruction
    overview.npz   the response matrix with the class pulled into a block, for the heat map
    molecules.csv  inchikey, smiles and class membership, for the RDKit grids
    meta.json      config, shapes, timing, what `knn` means for this dataset
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


def permuted(fn, X, n, seed=0):
    """Row-permutation null: shuffle the embedding rows, the alignment dies."""
    rng = np.random.default_rng(seed)
    return float(np.nanmean([fn(X[rng.permutation(len(X))]) for _ in range(n)]))


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
        readouts.append(dict(cls=cname, model=MODEL_ESM, feat="-", seed="-",
                             knn=knn_of(spec["kind"], Xe, tg["knn_target"], k),
                             rsa=rsa(Xe, tg["rsa_target"])))
        nulls.append(dict(
            cls=cname, n_receptors=len(order), n_molecules=len(iks),
            knn_baseline=knn_baseline(spec["kind"], p, R, recs, ods, tg, iks, k, args.n_perm),
            rsa_null=permuted(lambda Xs: rsa(Xs, tg["rsa_target"]), Xe, args.n_perm)))

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
                                     rsa=rsa(X, tg["rsa_target"])))
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
                print(f"  {cname:16} {model:24} seed {seed}  "
                      f"knn {readouts[-1]['knn']:+.3f}  rsa {readouts[-1]['rsa']:+.3f}", flush=True)

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
        metric_family="classification" if spec["task"] == "classification" else "regression",
        panel_class=(args.panel_class or (list(classes)[0] if classes else None)),
        ood=bool(args.ood), hladis=bool(args.hladis),
        variant=VARIANT, seconds=round(time.time() - t0, 1),
    ), indent=2), encoding="utf-8")
    print(f"=== {ds.upper()} written to {out} in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["all"],
                    help="m2or / cc / hc / all (default: all)")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="override the per-dataset class list")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=900, help="graph epochs (titular protocol)")
    ap.add_argument("--n-perm", type=int, default=100, help="row permutations for the nulls")
    ap.add_argument("--no-ood", dest="ood", action="store_false",
                    help="skip the predictive boosting readout (RSA/kNN only)")
    ap.add_argument("--hladis", action="store_true",
                    help="also score Hladis on the same OOD masks -- EXPENSIVE, it trains a "
                         "model per (class, seed) instead of reusing an embedding")
    ap.add_argument("--hladis-seeds", type=int, default=1,
                    help="how many of --seeds to give Hladis (default 1; 5 is ~5x the cost)")
    ap.add_argument("--panel-class", default=None,
                    help="class for the actual-vs-models panels (default: the first one)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/mechanism_holdout")
    args = ap.parse_args()

    todo = list(DATASETS) if "all" in args.dataset else args.dataset
    bad = [d for d in todo if d not in DATASETS]
    if bad:
        ap.error(f"unknown dataset(s) {bad}, have {list(DATASETS)}")
    for ds in todo:
        run_dataset(ds, args)


if __name__ == "__main__":
    main()
