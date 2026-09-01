#!/usr/bin/env python
"""v8: sweep the alpha gate on Carey / Hallem-Carlson, both primary regimes.

WHAT THE GATE IS. The receptor vector the boost eventually reads is

    z_prot = (1 - alpha) * frozen_SVD(ESM)  +  alpha * signed_graph(...)

with both branches RMS-normalised, the structural branch fit on TRAIN receptors and
frozen, and alpha fixed before training and used unchanged at inference. See
`orbind.gnn_extractor.GnnSignedExtractor.alpha`.

WHY IT EXISTS. In the historical graph the only path from ESM to the receptor
embedding is trainable, and 900 epochs of binding loss empty it: measured on the
(k, phi) grid, the refined receptor cloud sits AT the permutation null against ESM
(z = +0.4 on CC) while scoring z = +11 against the response profile, in every cell of
a two-axis ablation. Both of those axes removed information; neither could pull back
toward structure, because nothing in the architecture ever pulled that way. The
structural end was reachable only by not training. A frozen branch is a path the
optimizer cannot drain, which is what turns "structure vs function" from a pair of
ablations into one dial with two known ends:

    alpha = 0   the receptor cloud IS ESM's geometry -- exactly, on the train span,
                since the branch is an isometry there (uncentered, so cosines and
                not merely distances are preserved)
    alpha = 1   the historical graph, up to one global scalar no geometry sees

WHAT IS MEASURED. Every cell reports both halves at once:

  PREDICTION  the pipeline's own head -- `train_boost` on [ z_prot || raw molecule ],
              the `cls+mol` combo, R2/RMSE/MAE/Pearson/Spearman on the fold's test
              rows. Alongside it two references that are not the graph at all:
              `boost_full` = [ raw ESM || molecule ] (what the graph must beat) and
              `naive` = the constant train mean (R2's honest zero).
  GEOMETRY    the three second-order readouts (RSA, CCA, Procrustes) of the receptor
              cloud against raw ESM and against the TRAIN response profile, each with
              its own permutation null, so the dial can be watched moving. Train
              columns only -- the diagnostic must not see test odorants.

`graph_legacy` (alpha=None, the pre-v8 model) is run as its own arm so the
comparison to the previous implementation is a row in the same table under the same
folds, not a number quoted from another run.

DATASETS. cc/hc are the complete continuous insect panels: regression, R2/RMSE/MAE/
Pearson/Spearman, upstream's `rand` and our stratified `our_inductive` folds. m2or is
the sparse binary pool: classification, AUROC/AUPRC/MCC/F1, LORaX folds 1-5 for
transductive and cold-molecule seeds 42-46 for `inductive_molecule_v5`. The MP-edge
variant defaults per dataset -- q99/greedy on m2or, where coverage is heavy-tailed and
the quantile picks a hub core, q0/coverage on the insects, whose complete matrix leaves
the quantile nothing to cut -- and on m2or it is part of the filename, so both coexist.

    python scripts/modeling/train/run_alpha_gate_sweep.py --dataset hc cc \\
        --regime transductive inductive --max-parallel 4 --gpus 0 1 2 3
    python scripts/modeling/train/run_alpha_gate_sweep.py --dataset m2or \\
        --variant q99greedy --nodes onehot --max-parallel 4 --gpus 0 1 2 3

Resumable: a finished (arm, alpha, fold) is skipped, the CSV is rewritten after every
cell, and the parent process is its sole writer.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind.baselines import train_boost                          # noqa: E402
from orbind.dataset import METRICS as METRIC_FNS, load_npz_dict    # noqa: E402
from orbind.gnn_extractor import GnnSignedExtractor                # noqa: E402
from orbind.regimes import full_full_pairs, load_split           # noqa: E402
from orbind.regimes_ofm import ofm_indices, ofm_pairs              # noqa: E402
from scripts.modeling.analysis.mechanism_holdout import (          # noqa: E402
    GEOMETRY, geometry_nulls)

# What each regime name means per dataset. On the insects `rand` is i.i.d. and
# `our_inductive` is the stratified cold-molecule family -- upstream's `scaf` is
# cold-molecule too but its fold 1 is degenerate (test sd 0.215), so a sweep must not
# be read off it. On M2OR they are the LORaX folds and the v5 cold-molecule split, the
# two the paper's tables use. The ligand-class holdout ("special-inductive") is a
# different script: scripts/modeling/analysis/mechanism_holdout.py
FAMILY = {"cc": {"transductive": "rand", "inductive": "our_inductive"},
          "hc": {"transductive": "rand", "inductive": "our_inductive"},
          "m2or": {"transductive": "transductive",
                   "inductive": "inductive_molecule_v5"}}
# M2OR's repeats are cold-molecule SEEDS, not folds, for the inductive regime.
REPEATS = {"cc": {}, "hc": {},
           "m2or": {"transductive": [1, 2, 3, 4, 5],
                    "inductive": [42, 43, 44, 45, 46]}}
TASK = {"cc": "regression", "hc": "regression", "m2or": "classification"}
TASK_METRICS = {"regression": ["R2", "RMSE", "MAE", "Pearson", "Spearman"],
                "classification": ["AUROC", "AUPRC", "MCC", "F1"]}
# The MP-edge variants. On M2OR coverage is heavy-tailed and q99+greedy picks a hub
# core -- that is the graph every M2OR number in the paper stands on. On the complete
# insect matrices the quantile has nothing to cut, so q0/coverage is theirs.
VARIANTS = {"q99greedy": dict(q=0.99, criterion="greedy_pair_cover",
                              k_mode="coverage_quantile"),
            "q0cov": dict(q=0.0, criterion="coverage", k_mode="coverage_quantile")}
DEFAULT_VARIANT = {"cc": "q0cov", "hc": "q0cov", "m2or": "q99greedy"}
GEOMS = ["rsa", "cca", "procrustes"]
REFS = ["esm", "fun"]
KEY = ["arm", "alpha", "fold"]
DEFAULTS = {"prot": "data/embeddings/proteins/esm1b_650m_mean_{ds}.npz",
            "mol": "data/embeddings/molecules/gin_supervised_contextpred_{ds}.npz"}
# M2OR's files carry no dataset suffix, and its molecule source is the GIN the paper
# uses for every M2OR number.
DEFAULTS_M2OR = {"prot": "data/embeddings/proteins/esm1b_650m_mean.npz",
                 "mol": "data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz"}


def paths(ds, args):
    if args.prot_embeddings or args.mol_embeddings:
        d = dict(DEFAULTS_M2OR if ds == "m2or" else DEFAULTS)
        if args.prot_embeddings:
            d["prot"] = args.prot_embeddings
        if args.mol_embeddings:
            d["mol"] = args.mol_embeddings
    else:
        d = DEFAULTS_M2OR if ds == "m2or" else DEFAULTS
    return d["prot"].format(ds=ds), d["mol"].format(ds=ds)


def repeats(ds, regime, args):
    if args.folds:
        return list(args.folds)
    return REPEATS[ds].get(regime, [1, 2, 3, 4, 5])


def _fmt(sec):
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def _mat(emb, keys):
    return np.stack([emb[k] for k in keys]).astype(np.float32)


def _prepare(ds, args):
    """Pool + embeddings + coverage mask, once per process."""
    pairs = ofm_pairs(ds) if ds != "m2or" else full_full_pairs(pool_fold=args.pool_fold)
    pp, mp = paths(ds, args)
    mol = load_npz_dict(str(_root / mp))
    prot = load_npz_dict(str(_root / pp))
    cov = pairs["inchikey"].isin(mol).to_numpy() & pairs["receptor"].isin(prot).to_numpy()
    return pairs, mol, prot, cov


def _split(ds, regime, rep):
    """(train, val, test) row indices. M2OR keeps its splits in a persisted npz keyed
    by regime+repeat; the insects derive theirs from upstream's fold files."""
    if ds == "m2or":
        return load_split(FAMILY[ds][regime], int(rep))
    return ofm_indices(ds, FAMILY[ds][regime], int(rep))


def _fold_prep(ds, regime, fold, args, data):
    """Everything a fold needs, shared by every arm run on it."""
    pairs, mol, prot, cov = data
    ik, rc = pairs["inchikey"].to_numpy(), pairs["receptor"].to_numpy()
    lab = pairs["label"].to_numpy(np.float32)
    tr, va, te = (np.asarray(a)[cov[np.asarray(a)]]
                  for a in _split(ds, regime, fold))
    # Receptor order for the geometry readouts: the graph's own universe order, so
    # z_prot rows line up with the reference clouds without a second lookup.
    order = list(pd.unique(np.concatenate([rc[tr], rc[va], rc[te]])))
    rank = {r: i for i, r in enumerate(order)}
    # The functional reference: the TRAIN response matrix, receptors x train
    # odorants, row-centred. No test odorant enters it -- a diagnostic that peeked
    # would report the leak as a success.
    #
    # CAUTION on M2OR. There the matrix is SPARSE, so most cells are "not assayed"
    # rather than "no response", and the imputation below writes the global mean into
    # them. What is left is partly WHICH PAIRS WERE MEASURED -- assay design, not
    # binding. The `tested mask` control in mechanism_holdout measured that share at
    # 76-103% of the trained graph's own gain on M2OR. So on M2OR read the `fun`
    # geometry columns as an upper bound contaminated by design, and lean on the
    # prediction columns instead. On the complete insect matrices this does not arise.
    tr_mols = list(pd.unique(ik[tr]))
    mrank = {m: i for i, m in enumerate(tr_mols)}
    R = np.full((len(order), len(tr_mols)), np.nan)
    R[[rank[r] for r in rc[tr]], [mrank[m] for m in ik[tr]]] = lab[tr]
    R = np.nan_to_num(R, nan=float(np.nanmean(R)) if np.isfinite(R).any() else 0.0)
    return {"pairs": pairs, "tr": tr, "va": va, "te": te,
            "y_tr": lab[tr], "y_te": lab[te],
            "Xm_tr": _mat(mol, ik[tr]), "Xm_te": _mat(mol, ik[te]),
            "Xp_tr": _mat(prot, rc[tr]), "Xp_te": _mat(prot, rc[te]),
            "order": order, "prank": rank,
            "ref": {"esm": _mat(prot, order).astype(np.float64),
                    "fun": R - R.mean(1, keepdims=True)}}


def _geometry(Z, P, n_perm):
    """RSA / CCA / Procrustes of the receptor cloud against both reference clouds,
    each as a raw value and as z against its own permutation null. The nulls are
    what make three measures on different scales comparable at all."""
    out = {}
    for ref, M in P["ref"].items():
        stats = geometry_nulls(Z, M, n_perm) if n_perm else {}
        for g in GEOMS:
            v = float(GEOMETRY[g](Z, M))
            out[f"{g}_{ref}"] = v
            if g in stats:
                mu, sd = stats[g]
                out[f"{g}_{ref}_z"] = float((v - mu) / sd) if sd > 1e-12 else np.nan
    return out


def _seed(args, fold):
    """The seed handed to BOTH the graph and the boosting head.

    Default 42 for every fold, which is not an aesthetic choice: under `--regime ofm`
    (and full_full) `train_ensemble_boost.py` calls `run_ensemble` WITHOUT a `seed=`
    argument, so it keeps that function's default of 42 and passes the same 42 to the
    extractor and to `fit_boost` on every fold -- only the split changes with the
    fold. Every cc/hc number in the tables was produced that way. `fit_boost` draws
    subsample=0.8 / colsample_bytree=0.8 from `random_state`, so seeding by fold
    instead re-rolls the head on each fold and moves R2 by up to 0.045 per fold in
    either direction; that is exactly what made this sweep's first `boost_full` arm
    read 0.501 against the table's 0.516 on CC and 0.453 against 0.468 on HC, with
    identical features and an identical head. `--seed-per-fold` restores the other
    convention, which is arguably the better experiment but is NOT the one the
    reported numbers come from."""
    return int(fold) if args.seed_per_fold else int(args.seed)


def _row(arm, alpha, fold, **rest):
    return {"arm": arm, "alpha": np.nan if alpha is None else float(alpha),
            "fold": int(fold), "status": "ok", **rest}


def _score(y_te, pred, task):
    m = METRIC_FNS[task](y_te, pred)
    return {k: float(m[k]) for k in TASK_METRICS[task]}


def _baseline_rows(fold, P, args):
    """The two references the graph is judged against, on this fold's own rows.

    `naive` is the constant train mean: under regression that is upstream's own naive
    baseline and the only thing that makes an R2 near zero readable; under
    classification the same constant is the class prevalence, i.e. AUROC 0.5 and the
    AUPRC floor."""
    task = TASK[args._ds]
    rows = []
    pred = train_boost(np.concatenate([P["Xp_tr"], P["Xm_tr"]], 1), P["y_tr"],
                       np.concatenate([P["Xp_te"], P["Xm_te"]], 1),
                       seed=_seed(args, fold), task=task)
    rows.append(_row("boost_full", None, fold, n_receptors=len(P["order"]),
                     **_score(P["y_te"], pred, task)))
    const = np.full(len(P["y_te"]), float(P["y_tr"].mean()), dtype=np.float32)
    rows.append(_row("naive", None, fold, n_receptors=len(P["order"]),
                     **_score(P["y_te"], const, task)))
    return rows


def _graph_row(arm, alpha, fold, ds, P, args):
    """One trained graph -> the cls+mol boost feature -> metrics + geometry."""
    task = TASK[ds]
    pp, mp = paths(ds, args)
    ext = GnnSignedExtractor(
        name="cls", protein_path=pp, molecule_path=mp, **VARIANTS[args._variant],
        task=task, n_models=args.n_models, epochs=args.epochs,
        emit="prot", alpha=alpha, onehot_nodes=(args.nodes == "onehot"))
    seed = _seed(args, fold)
    Zp_tr, Zp_va, Zp_te = ext.fit_transform(P["pairs"], P["tr"], P["va"], P["te"], seed)
    pred = train_boost(np.concatenate([Zp_tr, P["Xm_tr"]], 1), P["y_tr"],
                       np.concatenate([Zp_te, P["Xm_te"]], 1),
                       seed=seed, task=task)
    # One receptor, one row: the per-pair features repeat the receptor vector, so
    # collapse back to the universe order the reference clouds are in. All three
    # splits are walked -- a receptor that appears only in val would otherwise be
    # missing and silently shift every row below it.
    seen = {}
    for idx, part in ((P["tr"], Zp_tr), (P["va"], Zp_va), (P["te"], Zp_te)):
        rc = P["pairs"]["receptor"].to_numpy()[idx]
        for i, r in enumerate(rc):
            seen.setdefault(r, part[i])
    missing = [r for r in P["order"] if r not in seen]
    Z = np.stack([seen[r] for r in P["order"] if r in seen]).astype(np.float64)
    geo = {} if missing else _geometry(Z, P, args.n_perm)
    return _row(arm, alpha, fold, n_receptors=len(P["order"]),
                k_pca=int(getattr(ext, "_k_pca", 0)), variant=args._variant,
                **_score(P["y_te"], pred, task), **geo)


def _failed(arm, alpha, fold, err):
    r = _row(arm, alpha, fold, **{k: float("nan") for k in TASK_METRICS["regression"]},
             **{k: float("nan") for k in TASK_METRICS["classification"]})
    r["status"] = f"failed: {err}"
    return r


def _run_job(job, ds, regime, args, cache, data):
    arm, alpha, fold = job
    if fold not in cache:
        cache[fold] = _fold_prep(ds, regime, fold, args, data)
    P = cache[fold]
    if arm == "baselines":
        return _baseline_rows(fold, P, args)
    return [_graph_row(arm, alpha, fold, ds, P, args)]


def _worker(job_q, res_q, ds, regime, args):
    """Persistent GPU-pinned worker; `worker_done` from `finally` so a crash cannot
    leave the parent blocked forever waiting for a row that will never come."""
    try:
        data, cache = _prepare(ds, args), {}
        while True:
            job = job_q.get()
            if job is None:
                break
            try:
                for row in _run_job(job, ds, regime, args, cache, data):
                    res_q.put(("row", row))
            except Exception as e:                  # noqa: BLE001 -- one cell must not kill the sweep
                res_q.put(("row", _failed(job[0], job[1], job[2], e)))
    finally:
        res_q.put(("worker_done", None))


def sweep(ds, regime, args):
    args._ds, args._variant = ds, (args.variant or DEFAULT_VARIANT[ds])
    reps = repeats(ds, regime, args)
    # The variant is in the filename only where more than one is in play. The insects
    # were only ever run at their own q0cov, and renaming those files now would orphan
    # the series already on disk; every row carries a `variant` column regardless.
    tag = "" if args.nodes == "esm" else f"_{args.nodes}"
    if ds == "m2or":
        tag = f"_{args._variant}{tag}"
    out = pathlib.Path(args.out or (_root / "results/graph/v8_alpha_gate")) / \
        f"metrics_{ds}_{FAMILY[ds][regime]}{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, done = [], set()
    if out.exists() and not args.force:
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {(r["arm"], "" if pd.isna(r["alpha"]) else round(float(r["alpha"]), 6),
                 int(r["fold"])) for _, r in prev.iterrows()}

    def key(arm, alpha, fold):
        return (arm, "" if alpha is None else round(float(alpha), 6), int(fold))

    jobs = []
    for f in reps:
        if not all(key(a, None, f) in done for a in ("boost_full", "naive")):
            jobs.append(("baselines", None, f))
        if args.baselines_only:
            continue
        if args.legacy and key("graph_legacy", None, f) not in done:
            jobs.append(("graph_legacy", None, f))
        for a in args.alphas:
            if key("gate", a, f) not in done:
                jobs.append(("gate", float(a), f))

    print(f"\n=== {ds.upper()} / {regime} ({FAMILY[ds][regime]}) ===\n"
          f"    repeats {reps}  alphas {args.alphas}  legacy {args.legacy}\n"
          f"    task {TASK[ds]}  nodes {args.nodes}  variant {args._variant} "
          f"{VARIANTS[args._variant]}\n"
          f"    {len(jobs)} jobs ({len(done)} cells already done)  ->  {out}", flush=True)
    if not jobs:
        return out

    heavy = sum(1 for j in jobs if j[0] != "baselines")
    t0, state = time.time(), {"n": 0}

    def record(row):
        k = key(row["arm"], None if pd.isna(row["alpha"]) else row["alpha"], row["fold"])
        if k in done:
            return
        rows.append(row); done.add(k)
        tmp = out.with_suffix(".tmp.csv")
        pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(out)
        if row["arm"] in ("boost_full", "naive"):
            tag = f"[{row['arm']}]"
        else:
            state["n"] += 1
            el = time.time() - t0
            tag = (f"[{state['n']}/{heavy} {_fmt(el)} "
                   f"ETA {_fmt(el / state['n'] * (heavy - state['n']))}]")
        shown = ("R2", "Spearman") if TASK[ds] == "regression" else ("AUROC", "AUPRC")
        body = (row["status"] if str(row["status"]).startswith("failed")
                else " ".join(f"{m}={row[m]:+.3f}" for m in shown)
                + "".join(f"  {g}/{r}={row.get(f'{g}_{r}_z', float('nan')):+.1f}"
                          for r in REFS for g in ("rsa",)))
        a = "" if pd.isna(row["alpha"]) else f"a={row['alpha']:.2f}"
        print(f"  fold {row['fold']} {row['arm']:<12} {a:<7} {body}  {tag}", flush=True)

    if args.max_parallel <= 1:
        data, cache = _prepare(ds, args), {}
        for job in jobs:
            try:
                for row in _run_job(job, ds, regime, args, cache, data):
                    record(row)
            except Exception as e:                  # noqa: BLE001
                record(_failed(job[0], job[1], job[2], e))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        job_q, res_q = ctx.Queue(), ctx.Queue()
        gpus = args.gpus or [0]
        n = min(args.max_parallel, len(jobs))
        for j in jobs:
            job_q.put(j)
        for _ in range(n):
            job_q.put(None)
        procs = []
        for i in range(n):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
            p = ctx.Process(target=_worker, args=(job_q, res_q, ds, regime, args))
            p.start(); procs.append(p)
            print(f"    worker {i} -> GPU {gpus[i % len(gpus)]}", flush=True)
        alive = n
        while alive:
            kind, payload = res_q.get()
            if kind == "worker_done":
                alive -= 1
            else:
                record(payload)
        for p in procs:
            p.join()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["hc", "cc"],
                    choices=["hc", "cc", "m2or"],
                    help="cc/hc are the continuous insect panels (regression); m2or is "
                         "the sparse binary pool (classification, AUROC/AUPRC/MCC/F1)")
    ap.add_argument("--variant", choices=list(VARIANTS), default=None,
                    help="MP-edge variant. Default per dataset: q99greedy on m2or (the "
                         "graph its tables stand on), q0cov on the insects (their complete "
                         "matrix leaves the quantile nothing to cut). On m2or the variant "
                         "is part of the output filename, so both can coexist")
    ap.add_argument("--regime", nargs="+", default=["transductive", "inductive"],
                    choices=["transductive", "inductive"],
                    help="the two PRIMARY regimes. The ligand-class holdout "
                         "(special-inductive) lives in mechanism_holdout.py")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="repeats to run. Default: folds 1-5, except m2or/inductive "
                         "whose repeats are the cold-molecule seeds 42-46")
    ap.add_argument("--pool-fold", type=int, default=1,
                    help="m2or only: which LORaX fold reconstructs the pool")
    ap.add_argument("--no-legacy", dest="legacy", action="store_false",
                    help="skip the pre-v8 graph arm (alpha=None)")
    ap.add_argument("--prot-embeddings", default=None,
                    help="override; {ds} is filled in per dataset")
    ap.add_argument("--mol-embeddings", default=None,
                    help="override; also the graph's MP node features")
    ap.add_argument("--n-models", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutations per geometry null; 0 skips the nulls")
    ap.add_argument("--nodes", choices=["esm", "onehot"], default="esm",
                    help="receptor NODE features. `onehot` removes ESM from the graph "
                         "entirely, so with the gate on it reaches the receptor vector "
                         "ONLY through the frozen branch at weight (1-alpha) -- the "
                         "decomposition in which alpha is an honest fraction of "
                         "structure. Writes its own metrics_*_onehot.csv")
    ap.add_argument("--baselines-only", action="store_true",
                    help="only boost_full + naive (no graph, so seconds not hours) -- the "
                         "cheap way to check this sweep reproduces the table of record "
                         "before spending a GPU on the rest")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the graph AND the boosting head on every fold -- the "
                         "ensembler's own convention under --regime ofm, which is what the "
                         "reported cc/hc numbers were produced with. See _seed()")
    ap.add_argument("--seed-per-fold", action="store_true",
                    help="seed by fold number instead; a different experiment, not "
                         "comparable to the tables")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--gpus", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="recompute cells already in the CSV")
    args = ap.parse_args()

    if args.variant and any(d != "m2or" for d in args.dataset):
        print("NOTE: --variant is being applied to an insect dataset too; its canonical "
              "graph is q0cov and the filename will NOT record the variant there.",
              flush=True)
    written = []
    for ds in args.dataset:
        for regime in args.regime:
            written.append(sweep(ds, regime, args))
    print("\nwrote:")
    for w in written:
        print(f"  {w}")
    print("\nread with:\n  python scripts/analysis/alpha_gate_summary.py -c")


if __name__ == "__main__":
    main()
