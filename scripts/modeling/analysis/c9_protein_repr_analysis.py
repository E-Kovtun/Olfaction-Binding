"""C9 -- information criteria on the receptor representation BEFORE vs AFTER our graph.

Runs entirely off an existing *ensemble-structure GNN run*: it reads that run's
`config.json` (dataset, regime, protein/molecule embedding paths, the gnn source
string with q/criterion/n_models/emit, the repeats) and RELOADS the already-trained
per-model checkpoints from `checkpoints/repeat_{R}/` -- so it recomputes the graph
node embeddings without retraining. Nothing in the run (or the pipeline) is touched.

For one repeat it builds a set of per-receptor representations
    R_raw   -- the raw mean-ESM node feature (before the graph)
    R_ref   -- the graph-refined receptor vector z_prot (after the graph)   [reloaded]
    R_pca   -- PCA of R_raw to dim(R_ref)          (compression control)
    R_rand  -- an UNTRAINED graph (epochs=0, random init)  (learning control)
    R_shuf  -- a graph trained on LABEL-PERMUTED edges     (structure control; --no-shuffle to skip)
and runs six information criteria (each isolated in try/except -- a criterion that
fails or is null does not abort the rest):

 1. geometry shift phylogeny->function : dist-corr(rep, seq) vs dist-corr(rep, func) + kNN functional purity
 2. dependence with functional structure: linear/RBF CKA, HSIC, kernel-target alignment vs K_func and K_seq
 3. ligand-profile recoverability        : CV multi-output ridge  R[rec] -> its binding profile
 4. task predictivity (cold molecule)    : reader on [R[rec] || raw molecule] -> binding, linear + MLP
 5. information budget / compression      : participation ratio, spectral entropy, identity-retention R2
 6. mutual information with label         : kNN-MI proxy between the receptor rep and its positive rate (exploratory)

Expected pattern (state up front so nulls are not a surprise): generic predictivity
may not rise (or drop in transductive); the signal should concentrate in (3), (1)/(2)
and (4). Controls decide reality: a real gain survives vs R_pca and dies under R_shuf/R_rand.

Usage:
    python scripts/modeling/analysis/c9_protein_repr_analysis.py \
        --run-dir results/ensemble_logs/<pool>/<gnn_run> [--repeat 42] [--no-shuffle]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts" / "modeling" / "train"))

import torch  # noqa: E402

from orbind import gnn_extractor as GX  # noqa: E402
from orbind.regimes import full_full_pairs, load_split  # noqa: E402
from train_ensemble_boost import parse_source_arg  # noqa: E402


# --------------------------------------------------------------------------- run reconstruction

def load_run(run_dir: pathlib.Path):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    gnn_raw = next((s for s in cfg["sources"] if "=gnn_signed" in s), None)
    if gnn_raw is None:
        raise SystemExit(f"no gnn_signed source in {run_dir}/config.json (sources={cfg['sources']})")
    name, ext = parse_source_arg(gnn_raw)
    ext.name = name
    return cfg, ext


def covered_split(pairs, ext, mode, repeat):
    """(train, val, test) row indices restricted to rows this extractor can embed
    (both a protein and a molecule vector present) -- matches the run's on_missing=drop."""
    tr, va, te = load_split(mode, repeat)
    out = []
    for idx in (tr, va, te):
        idx = np.asarray(idx)
        out.append(idx[ext.covered(pairs, idx)])
    return out


# --------------------------------------------------------------------------- receptor representations

def _bag_encode(ext, pairs, all_idx, mp_edges, seed, n_models, epochs, checkpoint_dir, y_override=None):
    """Encode the whole graph n_models times and concat z_prot / z_mol per node.
    checkpoint_dir!=None with existing files -> reload (no training). epochs=0 ->
    random-init encode. y_override -> train the decoder on permuted labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol_to_i, prot_to_i, x_mol, x_prot = GX._build_universe(pairs, all_idx, ext._proteins, ext._molecules)
    train_df = pairs.iloc[mp_edges]
    mi = train_df["inchikey"].map(mol_to_i).to_numpy().copy()
    pi = train_df["receptor"].map(prot_to_i).to_numpy().copy()
    y = train_df["label"].to_numpy(dtype=np.float32)
    if y_override is not None:
        y = y_override
    pos, neg = GX._mp_edges(pairs, mp_edges, mol_to_i, prot_to_i, ext.q,
                            getattr(ext, "criterion", "coverage"))
    pos_eidx, neg_eidx = GX._edge_index_dict(pos, neg)

    prot_blocks, mol_blocks = [], []
    for m in range(n_models):
        hp = ext._hp(seed + ext.seed_offset * m); hp["epochs"] = epochs
        ckpt = (pathlib.Path(checkpoint_dir) / f"gnn_{ext.name}_model{m}.pt"
                if checkpoint_dir is not None else None)
        z_mol, z_prot, _ = GX._train_one(ext._build_model, x_mol, x_prot, pos_eidx, neg_eidx,
                                         mi, pi, y, hp, device, checkpoint_path=ckpt, hidden=ext.hidden)
        prot_blocks.append(z_prot); mol_blocks.append(z_mol)
    return (np.concatenate(prot_blocks, 1), np.concatenate(mol_blocks, 1),
            mol_to_i, prot_to_i, x_prot.cpu().numpy(), x_mol.cpu().numpy())


def build_representations(cfg, ext, pairs, tr, va, te, repeat, run_dir, do_shuffle):
    all_idx = np.concatenate([tr, va, te])
    ckpt_dir = run_dir / "checkpoints" / f"repeat_{repeat}"
    if not ckpt_dir.exists():
        raise SystemExit(f"no checkpoints at {ckpt_dir} -- rerun the GNN run with checkpoints on")

    # R_ref: reload the trained models (epochs value is irrelevant when checkpoints load)
    print("  building ref (reloading trained checkpoints)...", flush=True)
    z_prot, z_mol, mol_to_i, prot_to_i, x_prot, x_mol = _bag_encode(
        ext, pairs, all_idx, tr, repeat, ext.n_models, ext.epochs, ckpt_dir)

    reps = {"raw": x_prot, "ref": z_prot}
    print(f"    raw {x_prot.shape}  ref {z_prot.shape}", flush=True)

    # R_pca: PCA of raw to dim(ref) -- "is the gain just compression?"
    try:
        from sklearn.decomposition import PCA
        k = min(z_prot.shape[1], x_prot.shape[1], x_prot.shape[0] - 1)
        reps["pca"] = PCA(n_components=k, random_state=0).fit_transform(x_prot)
        print(f"    pca {reps['pca'].shape}", flush=True)
    except Exception as e:
        print(f"  [pca control skipped] {e}")

    # R_rand: untrained graph (epochs=0, random init), same n_models -- cheap (forward only)
    print("  building rand control (untrained graph, forward-only)...", flush=True)
    try:
        reps["rand"] = _bag_encode(ext, pairs, all_idx, tr, repeat, ext.n_models, 0, None)[0]
        print(f"    rand {reps['rand'].shape}", flush=True)
    except Exception as e:
        print(f"  [rand control skipped] {e}")

    # R_shuf: graph trained on label-permuted edges (single model, real training) -- THE slow step
    if do_shuffle:
        print(f"  building shuf control -- RETRAINING one model {ext.epochs} epochs on shuffled "
              f"labels (slow on CPU; pass --no-shuffle to skip)...", flush=True)
        try:
            y_tr = pairs.iloc[tr]["label"].to_numpy(dtype=np.float32)
            y_perm = np.random.RandomState(repeat).permutation(y_tr)
            reps["shuf"] = _bag_encode(ext, pairs, all_idx, tr, repeat, 1, ext.epochs, None,
                                       y_override=y_perm)[0]
            print(f"    shuf {reps['shuf'].shape}", flush=True)
        except Exception as e:
            print(f"  [shuf control skipped] {e}")

    return reps, mol_to_i, prot_to_i, z_mol, x_mol


# --------------------------------------------------------------------------- shared targets (functional structure)

def receptor_profiles(pairs, tr, prot_to_i, mol_to_i):
    """Signed train binding profile P[receptor, molecule] in {+1,-1,0} and the
    positive-only 0/1 profile. Only receptors present in prot_to_i / molecules in mol_to_i."""
    n_p, n_m = len(prot_to_i), len(mol_to_i)
    P = np.zeros((n_p, n_m), np.float32)
    sub = pairs.iloc[tr]
    for r, mol, lab in zip(sub["receptor"], sub["inchikey"], sub["label"]):
        if r in prot_to_i and mol in mol_to_i:
            P[prot_to_i[r], mol_to_i[mol]] = 1.0 if lab == 1 else -1.0
    return P


def _cos_gram(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return Xn @ Xn.T


def _funcsim(P):
    return _cos_gram(P)


# --------------------------------------------------------------------------- criteria 1-6

def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def crit1_geometry(reps, K_func, K_seq, P, k=10):
    """dist-corr(rep, seq) vs dist-corr(rep, func) + kNN functional purity."""
    iu = np.triu_indices(K_func.shape[0], k=1)
    d_seq, d_func = (1 - K_seq)[iu], (1 - K_func)[iu]
    pos = (P > 0).astype(np.float32)
    rows = {}
    for name, R in reps.items():
        D = 1 - _cos_gram(R)
        rows[name] = dict(corr_seq=_spearman(D[iu], d_seq),
                          corr_func=_spearman(D[iu], d_func),
                          knn_purity=_knn_purity(R, pos, k))
    return rows


def _knn_purity(R, pos, k):
    G = _cos_gram(R); np.fill_diagonal(G, -np.inf)
    nn = np.argsort(-G, axis=1)[:, :k]
    num = (pos @ pos.T); den = (pos.sum(1)[:, None] + pos.sum(1)[None, :] - num) + 1e-9
    J = num / den                                   # Jaccard of positive-ligand sets
    return float(np.mean([J[i, nn[i]].mean() for i in range(len(R))]))


def crit2_dependence(reps, K_func, K_seq):
    """Centered kernel alignment (CKA) and raw kernel-target alignment (KTA) of each
    representation's cosine Gram with the functional kernel K_func and the sequence
    kernel K_seq. Thesis: refinement raises alignment with K_func (functional), and
    typically lowers it with K_seq (phylogeny)."""
    def kta(G, K):
        return float(np.sum(G * K) / (np.linalg.norm(G) * np.linalg.norm(K) + 1e-12))
    rows = {}
    for name, R in reps.items():
        G = _cos_gram(R)
        rows[name] = dict(cka_func=_cka_gram(G, K_func), cka_seq=_cka_gram(G, K_seq),
                          kta_func=kta(G, K_func), kta_seq=kta(G, K_seq))
    return rows


def _cka_gram(G, K):
    n = G.shape[0]; H = np.eye(n) - 1.0 / n
    Gc, Kc = H @ G @ H, H @ K @ H
    return float(np.sum(Gc * Kc) / (np.linalg.norm(Gc) * np.linalg.norm(Kc) + 1e-12))


def crit3_profile_recoverability(reps, P):
    """CV multi-output ridge  R[receptor] -> its signed binding profile; mean R2."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    rows = {}
    keep = np.where(np.abs(P).sum(1) > 0)[0]           # receptors with any train edge
    Y = P[keep]
    for name, R in reps.items():
        X = R[keep]; scores = []
        for trn, tst in KFold(5, shuffle=True, random_state=0).split(X):
            m = Ridge(alpha=10.0).fit(X[trn], Y[trn])
            pred = m.predict(X[tst])
            ss_res = ((Y[tst] - pred) ** 2).sum(); ss_tot = ((Y[tst] - Y[trn].mean(0)) ** 2).sum()
            scores.append(1 - ss_res / (ss_tot + 1e-9))
        rows[name] = dict(profile_R2=float(np.mean(scores)))
    return rows


def crit4_task_probe(reps, pairs, tr, te, prot_to_i, mol_to_i, x_mol):
    """Reader on [R[receptor] || raw molecule] -> binding, cold-molecule test; linear + MLP."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score

    def pair_xy(idx, R):
        sub = pairs.iloc[idx]
        mask = sub["receptor"].isin(prot_to_i) & sub["inchikey"].isin(mol_to_i)
        sub = sub[mask]
        pr = R[sub["receptor"].map(prot_to_i).to_numpy()]
        mo = x_mol[sub["inchikey"].map(mol_to_i).to_numpy()]
        return np.concatenate([pr, mo], 1), sub["label"].to_numpy(np.int32)

    rows = {}
    for name, R in reps.items():
        Xtr, ytr = pair_xy(tr, R); Xte, yte = pair_xy(te, R)
        if len(np.unique(yte)) < 2:
            rows[name] = dict(auroc_linear=np.nan, auroc_mlp=np.nan); continue
        lin = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0).fit(Xtr, ytr)
        rows[name] = dict(
            auroc_linear=float(roc_auc_score(yte, lin.predict_proba(Xte)[:, 1])),
            auroc_mlp=float(roc_auc_score(yte, mlp.predict_proba(Xte)[:, 1])))
    return rows


def crit5_info_budget(reps):
    """Participation ratio, spectral entropy, and identity-retention R2 (reconstruct raw from ref)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict
    raw = reps["raw"]
    rows = {}
    for name, R in reps.items():
        Rc = R - R.mean(0); cov = Rc.T @ Rc / len(R)
        ev = np.clip(np.linalg.eigvalsh(cov), 0, None); ev = ev[ev > 0]
        pr = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))
        p = ev / ev.sum(); ent = float(-np.sum(p * np.log(p + 1e-12)))
        rows[name] = dict(participation_ratio=pr, spectral_entropy=ent)
    # how much raw identity is linearly retained in ref (and controls)
    for name, R in reps.items():
        if name == "raw":
            rows[name]["identity_retention_R2"] = 1.0; continue
        try:
            pred = cross_val_predict(Ridge(alpha=10.0), R, raw, cv=5)
            ss_res = ((raw - pred) ** 2).sum(); ss_tot = ((raw - raw.mean(0)) ** 2).sum()
            rows[name]["identity_retention_R2"] = float(1 - ss_res / (ss_tot + 1e-9))
        except Exception:
            rows[name]["identity_retention_R2"] = np.nan
    return rows


def crit6_mutual_info(reps, P):
    """Exploratory: kNN-MI between the receptor rep and its positive rate."""
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.decomposition import PCA
    pos_rate = (P > 0).sum(1) / (np.abs(P).sum(1) + 1e-9)
    rows = {}
    for name, R in reps.items():
        try:
            Z = PCA(n_components=min(16, R.shape[1]), random_state=0).fit_transform(R)
            mi = mutual_info_regression(Z, pos_rate, random_state=0)
            rows[name] = dict(mi_posrate=float(mi.sum()))
        except Exception as e:
            rows[name] = dict(mi_posrate=np.nan)
    return rows


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="results/ensemble_logs/<pool>/<gnn_run>")
    ap.add_argument("--repeat", type=int, default=None, help="which repeat/seed (default: first with checkpoints)")
    ap.add_argument("--k", type=int, default=10, help="k for kNN functional purity")
    ap.add_argument("--no-shuffle", action="store_true", help="skip the (retraining) label-shuffle control")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _root / run_dir
    cfg, ext = load_run(run_dir)
    if ext.emit != "prot":
        print(f"  note: run emit={ext.emit!r}; analysis uses node z_prot directly, unaffected.")

    dataset = cfg.get("dataset", "m2or")
    if dataset not in ("m2or", None):
        raise SystemExit(f"this script currently wires M2OR/full_full only (config dataset={dataset!r})")
    mode = cfg.get("full_full_mode") or cfg.get("split")
    ckpt_root = run_dir / "checkpoints"
    avail = sorted(int(p.name.split("_")[1]) for p in ckpt_root.glob("repeat_*")) if ckpt_root.exists() else []
    repeat = args.repeat if args.repeat is not None else (avail[0] if avail else None)
    if repeat is None:
        raise SystemExit(f"no checkpoints under {ckpt_root}")
    print(f"run={run_dir.name}  dataset={dataset}  mode={mode}  repeat={repeat}\n"
          f"gnn: q={ext.q} criterion={getattr(ext,'criterion','coverage')} n_models={ext.n_models} "
          f"emit={ext.emit} prot={ext.protein_path} mol={ext.molecule_path}", flush=True)

    pairs = full_full_pairs(pool_fold=cfg.get("pool_fold", 1))
    tr, va, te = covered_split(pairs, ext, mode, repeat)
    print(f"covered rows: train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    reps, mol_to_i, prot_to_i, z_mol, x_mol = build_representations(
        cfg, ext, pairs, tr, va, te, repeat, run_dir, do_shuffle=not args.no_shuffle)
    print("representations:", {k: v.shape for k, v in reps.items()}, flush=True)

    P = receptor_profiles(pairs, tr, prot_to_i, mol_to_i)
    K_func = _funcsim(P)
    K_seq = _cos_gram(reps["raw"])

    criteria = [
        ("1_geometry",        lambda: crit1_geometry(reps, K_func, K_seq, P, args.k)),
        ("2_dependence",      lambda: crit2_dependence(reps, K_func, K_seq)),
        ("3_profile_recover", lambda: crit3_profile_recoverability(reps, P)),
        ("4_task_probe",      lambda: crit4_task_probe(reps, pairs, tr, te, prot_to_i, mol_to_i, x_mol)),
        ("5_info_budget",     lambda: crit5_info_budget(reps)),
        ("6_mutual_info",     lambda: crit6_mutual_info(reps, P)),
    ]

    results = {}
    for name, fn in criteria:
        print(f"\n=== {name} ===", flush=True)
        try:
            r = fn(); results[name] = r
            print(pd.DataFrame(r).T.round(4).to_string())
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}
            print(f"  [criterion {name} failed: {e}]")

    out = pathlib.Path(args.out) if args.out else (run_dir / "c9_analysis" / f"repeat_{repeat}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": run_dir.name, "repeat": repeat, "mode": mode,
                               "reps": {k: list(v.shape) for k, v in reps.items()},
                               "results": results}, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved -> {out}")
    print("\nread the reps side-by-side per criterion: signal = ref beats raw AND survives vs pca, "
          "dies under rand/shuf. Expect it to concentrate in criteria 1/2/3 and cold-molecule 4.")


if __name__ == "__main__":
    main()
