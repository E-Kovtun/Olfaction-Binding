"""Quantile x criterion sweep of the APPROVED pipeline GNN.

Runs the exact ensemble-pipeline signed GNN (`orbind.gnn_extractor.GnnSignedExtractor`
-> `_SignedSage`, emit=prot, 5-model bag) across every molecule-ranking criterion
and a set of coverage quantiles, and writes the compact per-(criterion, quantile,
seed) CSV that `notebooks/graph/alternatives/protein_based_graph.ipynb` DISPLAYS.
The notebook no longer trains anything -- this script is the single producer.

The feature is built exactly as the pipeline's `cls+mol` combo: `[ bagged
graph-refined protein || raw molecule ]` (column order matches `--combos "12"`,
source 1 = gnn cls, source 2 = mol), fitted with the fixed head `train_boost` at
`seed=repeat` -- the same seed `orbind.ensemble.run_ensemble` uses for a combo.

Configurable: regime (inductive/transductive), molecule-embedding source (also the
GNN's MP node features), quantiles, criteria, seeds, and GPU parallelism
(`--max-parallel`/`--gpus`). Hardwired: the 7 molecule-ranking criteria
(`orbind.mol_selection`) and the established GNN architecture + hyperparameters
(`GnnSignedExtractor` defaults).

Parallelism: `--max-parallel` persistent worker processes, each pinned to one GPU
(`CUDA_VISIBLE_DEVICES` set in the parent before start -- see train_ensemble_boost.py
for why the initializer approach fails on Linux). Each **(seed, quantile, criterion)
GNN cell** is a separate queued job (boost_full is one cheap job per seed), so GNN
training -- not just the boost baseline -- stays spread across every GPU regardless of
how the seed count divides the GPUs. Workers cache per-seed prep and stream finished
rows back; the PARENT is the sole CSV writer, so the incremental/resumable CSV stays
race-free. Re-running skips finished (criterion, quantile, seed) cells.

    python scripts/modeling/train/run_quantile_criteria_sweep.py \
        --regime inductive --boost-full \
        --mol-embeddings data/embeddings/molecules/chemberta_77m_m2or.npz \
        --max-parallel 4 --gpus 0 1 2 3

Carey / Hallem-Carlson (`--dataset cc|hc`)
------------------------------------------
Same sweep on a continuous target: repeats become upstream's folds 1..5, the
head becomes an XGBRegressor, and the reported columns become R2/RMSE/MAE/
Pearson/Spearman. Two things change by necessity, not by taste:

* **k_mode defaults to `fraction`.** These matrices are COMPLETE (CC 50x110,
  HC 24x110), so every train molecule has identical coverage, the coverage
  quantile has nothing to cut on, and `cov >= quantile(cov, q)` keeps all of
  them -- on CC/our_inductive, q=0.99 and q=0 both give K=70 of 70. `fraction`
  keeps the top (1-q) share outright, which is the tie-free reading of the same
  intent. Passing `--k-mode coverage_quantile` there is allowed but warns.
* **Only `coverage` and `greedy_pair_cover` are available.** The other five
  criteria score molecules by positive/negative counts, which need a 0/1 label.

    python scripts/modeling/train/run_quantile_criteria_sweep.py \
        --dataset cc --regime inductive --boost-full \
        --criteria coverage greedy_pair_cover --n-models 1 \
        --mol-embeddings data/embeddings/molecules/chemberta_77m_cc.npz \
        --prot-embeddings data/embeddings/proteins/esm1b_650m_mean_cc.npz \
        --max-parallel 4 --gpus 0 1 2 3
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

from orbind.gnn_extractor import GnnSignedExtractor              # noqa: E402
from orbind.mol_selection import CRITERIA, K_MODES, resolve_K    # noqa: E402
from orbind.baselines import train_boost                         # noqa: E402
from orbind.dataset import METRICS as METRIC_FNS, load_npz_dict  # noqa: E402
from orbind.regimes import full_full_pairs, load_split           # noqa: E402
from orbind.regimes_ofm import ofm_pairs, ofm_indices            # noqa: E402

TASK_METRICS = {"classification": ["AUROC", "AUPRC", "MCC", "F1"],
                "regression": ["R2", "RMSE", "MAE", "Pearson", "Spearman"]}
REGIME_KEY = {"inductive": "inductive_molecule_v5", "transductive": "transductive"}
DEFAULT_REPEATS = {"inductive": [42, 43, 44, 45, 46], "transductive": [1, 2, 3, 4, 5]}

# On the ofm datasets `--regime` names a split family instead of an M2OR regime.
# `rand` is i.i.d. (transductive); `our_inductive` is our stratified cold-molecule
# family (scripts/preprocessing/03_build_ofm_our_inductive_splits.py) -- upstream's
# `scaf` is cold-molecule too but has a degenerate fold 1 (test sd 0.215), so it is
# not what a sweep should be read off.
OFM_FAMILY = {"inductive": "our_inductive", "transductive": "rand"}


def _pair_matrix(emb: dict, keys) -> np.ndarray:
    return np.stack([emb[k] for k in keys]).astype(np.float32)


def _fmt(sec: float) -> str:
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def _prepare(args):
    """Load the pool + embeddings + coverage mask once (per process)."""
    pairs = (ofm_pairs(args.dataset) if args.dataset != "m2or"
             else full_full_pairs(pool_fold=args.pool_fold))
    mol_emb = load_npz_dict(str(_root / args.mol_embeddings))
    prot_emb = load_npz_dict(str(_root / args.prot_embeddings))
    cov = pairs["inchikey"].isin(mol_emb).to_numpy() & pairs["receptor"].isin(prot_emb).to_numpy()
    return pairs, mol_emb, prot_emb, cov


def _split(args, seed):
    """`seed` is an M2OR split seed, or an ofm FOLD number (1..5)."""
    if args.dataset != "m2or":
        return ofm_indices(args.dataset, OFM_FAMILY[args.regime], int(seed))
    return load_split(REGIME_KEY[args.regime], seed)


def _seed_prep(seed, args, data):
    """Per-seed tensors reused across all quantiles/criteria of that seed."""
    pairs, mol_emb, prot_emb, cov = data
    ik = pairs["inchikey"].to_numpy(); rc = pairs["receptor"].to_numpy()
    lab = pairs["label"].to_numpy().astype(np.float32)
    tr, va, te = (np.asarray(a)[cov[np.asarray(a)]] for a in _split(args, seed))
    y_tr, y_te = lab[tr], lab[te]
    Xm_tr, Xm_te = _pair_matrix(mol_emb, ik[tr]), _pair_matrix(mol_emb, ik[te])
    Xp_tr_raw, Xp_te_raw = _pair_matrix(prot_emb, rc[tr]), _pair_matrix(prot_emb, rc[te])
    uniq = pd.unique(ik[tr]); loc = {k: i for i, k in enumerate(uniq)}
    cov_counts = np.bincount([loc[k] for k in ik[tr]], minlength=len(uniq))
    Kq = {q: resolve_K(cov_counts, q / 100.0, args.k_mode) for q in args.quantiles}
    return {"pairs": pairs, "tr": tr, "va": va, "te": te, "y_tr": y_tr, "y_te": y_te,
            "Xm_tr": Xm_tr, "Xm_te": Xm_te, "Xp_tr_raw": Xp_tr_raw, "Xp_te_raw": Xp_te_raw,
            "uniq": uniq, "Kq": Kq}


def _boost_rows(seed, args, P):
    """No-graph [protein||molecule] baseline; q-independent, replicated per quantile.

    Under regression this doubles as the honest reference the naive row plays in
    the ensembler: R2 here is against the test mean, so a negative number means
    the features lost to predicting a constant."""
    cols = TASK_METRICS[args.task]
    sc = train_boost(np.concatenate([P["Xp_tr_raw"], P["Xm_tr"]], 1), P["y_tr"],
                     np.concatenate([P["Xp_te_raw"], P["Xm_te"]], 1), seed=seed,
                     task=args.task)
    mb = METRIC_FNS[args.task](P["y_te"], sc)
    return [{"criterion": "boost_full", "quantile": q, "K": len(P["uniq"]), "seed": seed,
             "n_models": args.n_models, "status": "ok",
             **{k: float(mb[k]) for k in cols}}
            for q in args.quantiles]


def _gnn_row(seed, q, crit, args, P):
    """One (seed, quantile, criterion) GNN cell -> boost head; the schedulable unit."""
    cols = TASK_METRICS[args.task]
    ext = GnnSignedExtractor(
        name="gnn", protein_path=args.prot_embeddings,
        molecule_path=args.mol_embeddings, q=q / 100.0, criterion=crit,
        k_mode=args.k_mode, task=args.task,
        n_models=args.n_models, epochs=args.epochs, emit="prot")
    Zp_tr, _Zp_va, Zp_te = ext.fit_transform(P["pairs"], P["tr"], P["va"], P["te"], seed)
    sc = train_boost(np.concatenate([Zp_tr, P["Xm_tr"]], 1), P["y_tr"],
                     np.concatenate([Zp_te, P["Xm_te"]], 1), seed=seed, task=args.task)
    m = METRIC_FNS[args.task](P["y_te"], sc)
    return {"criterion": crit, "quantile": q, "K": int(P["Kq"][q]), "seed": seed,
            "n_models": args.n_models, "status": "ok",
            **{k: float(m[k]) for k in cols}}


def _failed_row(seed, q, crit, args, K, err):
    """A cell that cannot be computed is still an answer -- record it as NaN with
    the reason, so the CSV documents the hole and a resume doesn't retry it.

    The one that actually happens: at a small K the kept molecules can all sit on
    one side of `edge_threshold`, and signed message passing needs both signs. On
    Carey at q=99 K is a SINGLE molecule, so whether its 50 receptor responses
    straddle zero is a per-fold coin flip."""
    return {"criterion": crit, "quantile": q, "K": int(K), "seed": seed,
            "n_models": args.n_models, "status": f"failed: {err}",
            **{k: float("nan") for k in TASK_METRICS[args.task]}}


def _worker_loop(job_q, res_q, args):
    """Persistent worker: the parent pinned CUDA_VISIBLE_DEVICES before start(), so
    every torch/XGBoost op here lands on this worker's GPU. Loads the pool once and
    caches per-seed prep, then pulls (seed, quantile, criterion) jobs -- so GNN
    training, not just the boost baseline, stays spread across every GPU.

    Every job is guarded and `worker_done` is sent from `finally`: a worker that
    dies silently would leave the parent blocked forever in its collection loop,
    which is exactly what an uncaught degenerate-graph error used to do."""
    try:
        data = _prepare(args)
        prep = {}
        while True:
            job = job_q.get()
            if job is None:
                break
            kind, seed, q, crit = job
            try:
                if seed not in prep:
                    prep[seed] = _seed_prep(seed, args, data)
                P = prep[seed]
                if kind == "boost":
                    for row in _boost_rows(seed, args, P):
                        res_q.put(("row", row, False))
                else:
                    res_q.put(("row", _gnn_row(seed, q, crit, args, P), True))
            except Exception as e:                       # noqa: BLE001 -- one cell must not kill the sweep
                K = prep.get(seed, {}).get("Kq", {}).get(q, -1)
                res_q.put(("row", _failed_row(seed, q, crit, args, K, e), True))
    finally:
        res_q.put(("worker_done", None, None))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["inductive", "transductive"], default="inductive",
                    help="M2OR: inductive_molecule_v5 / transductive. ofm datasets: the "
                         "our_inductive / rand split family.")
    ap.add_argument("--dataset", choices=["m2or", "cc", "hc"], default="m2or",
                    help="cc = Carey, hc = Hallem-Carlson (both continuous -> regression, "
                         "repeats are folds 1..5)")
    ap.add_argument("--task", choices=["classification", "regression"], default=None,
                    help="default: regression for cc/hc, classification for m2or")
    ap.add_argument("--k-mode", choices=list(K_MODES), default=None,
                    help="how q becomes K. Default: coverage_quantile for m2or (the "
                         "historical reading), fraction for cc/hc -- their matrices are "
                         "complete, so the coverage quantile is a no-op there and the "
                         "whole sweep would collapse to a single point.")
    ap.add_argument("--mol-embeddings", default="data/embeddings/molecules/chemberta_77m_m2or.npz",
                    help="molecule embedding npz (inchikey-keyed): GNN MP node features AND "
                         "the raw-molecule half of the boost feature")
    ap.add_argument("--prot-embeddings", default="data/embeddings/proteins/esm1b_650m_mean.npz")
    ap.add_argument("--quantiles", type=float, nargs="+", default=[50, 80, 85, 90, 95, 99],
                    help="coverage quantiles as PERCENTS (matches the notebook x-axis)")
    ap.add_argument("--criteria", nargs="+", default=list(CRITERIA))
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default = inductive 42-46 / transductive folds 1-5")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--pool-fold", type=int, default=1)
    ap.add_argument("--boost-full", action="store_true",
                    help="also the no-graph [protein||molecule] baseline (replicated per quantile)")
    ap.add_argument("--max-parallel", type=int, default=1,
                    help="seeds to run concurrently, each in its own GPU-pinned process")
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="GPU ids to round-robin seeds across when --max-parallel > 1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    unknown = [c for c in args.criteria if c not in CRITERIA]
    if unknown:
        ap.error(f"unknown criteria {unknown}; choose from {list(CRITERIA)}")

    is_ofm = args.dataset != "m2or"
    if args.task is None:
        args.task = "regression" if is_ofm else "classification"
    if args.k_mode is None:
        args.k_mode = "fraction" if is_ofm else "coverage_quantile"
    if is_ofm and args.k_mode == "coverage_quantile":
        print("WARNING: k_mode=coverage_quantile on a complete matrix keeps every "
              "molecule at every q -- all quantiles will produce the same model.",
              flush=True)
    # `greedy_pair_cover` and `coverage` are the only criteria that never read y,
    # and the others are built from npos/nneg, i.e. from a 0/1 label.
    if args.task == "regression":
        bad = [c for c in args.criteria if c not in ("coverage", "greedy_pair_cover")]
        if bad:
            ap.error(f"criteria {bad} score molecules by positive/negative counts and "
                     f"are undefined on a continuous target; use coverage and/or "
                     f"greedy_pair_cover under --task regression")

    regime_key = (f"{args.dataset}_{OFM_FAMILY[args.regime]}" if is_ofm
                  else REGIME_KEY[args.regime])
    seeds = args.seeds or (list(range(1, 6)) if is_ofm else DEFAULT_REPEATS[args.regime])
    mol_stem = pathlib.Path(args.mol_embeddings).stem
    nm_tag = "" if args.n_models == 5 else f"__nm{args.n_models}"   # 5-model bag keeps the legacy name
    out = pathlib.Path(args.out) if args.out else (
        _root / "results/graph/full_full/v7/protein_based_graph"
        / f"metrics_{regime_key}__{mol_stem}{nm_tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"regime={args.regime} ({regime_key})  task={args.task}  k_mode={args.k_mode}\n"
          f"seeds={seeds}  quantiles={args.quantiles}\n"
          f"criteria={args.criteria}\nmol={args.mol_embeddings}\nout={out}\n"
          f"max_parallel={args.max_parallel}  gpus={args.gpus}", flush=True)

    # ---- resumable state (parent is the sole reader/writer) --------------
    rows, done = [], set()
    if out.exists():
        prev = pd.read_csv(out)
        if "n_models" not in prev.columns:
            prev["n_models"] = 5          # legacy files predate n_models; they were all 5-model bags
        if "status" not in prev.columns:
            prev["status"] = "ok"         # ... and predate status; a written row was a good row
        rows = prev.to_dict("records")
        # `done` is scoped to THIS run's n_models: a cell computed at n_models=5 must
        # not skip its n_models=1 twin. All rows (any n_models) are kept for save().
        done = {(r["criterion"], float(r["quantile"]), int(r["seed"]))
                for _, r in prev.iterrows() if int(r["n_models"]) == args.n_models}
        print(f"resuming: {len(done)} cells at n_models={args.n_models} in {out.name} "
              f"({len(rows)} rows total across all n_models)", flush=True)

    def save():
        tmp = out.with_suffix(".tmp.csv")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(out)

    # progress counter over HEAVY (GNN) cells only
    # A complete matrix makes the CRITERION axis degenerate as well as the quantile
    # one: `coverage` is constant, and `greedy_pair_cover`'s gain is
    # n_touch^2 - covered_pairs, which is equal for every molecule when they all
    # touch every receptor. Both then reduce to their own tie-break, so a
    # criterion-vs-criterion difference on such a split is arbitrary, not
    # informative. Say so once, up front, rather than let the plot imply meaning.
    if len(args.criteria) > 1:
        probe = _seed_prep(seeds[0], args, _prepare(args))
        ik = probe["pairs"]["inchikey"].to_numpy()[probe["tr"]]
        cnt = pd.Series(ik).value_counts().to_numpy()
        if len(set(cnt.tolist())) == 1:
            print(f"WARNING: every train molecule has identical coverage ({cnt[0]} receptors), "
                  f"so both molecule-ranking criteria are ties broken arbitrarily -- read the "
                  f"K axis, not the criterion comparison.", flush=True)

    total = sum(1 for s in seeds for q in args.quantiles for c in args.criteria
                if (c, float(q), int(s)) not in done)
    t0 = time.time()
    state = {"heavy_done": 0}

    def record(row, heavy):
        key = (row["criterion"], float(row["quantile"]), int(row["seed"]))
        if key in done:
            return
        rows.append(row); done.add(key); save()
        if heavy:
            state["heavy_done"] += 1
            el = time.time() - t0
            eta = el / state["heavy_done"] * (total - state["heavy_done"]) if state["heavy_done"] else 0
            tag = f"[{state['heavy_done']}/{total}  {_fmt(el)} elapsed  ETA {_fmt(eta)}]"
        else:
            tag = "[boost_full]"
        body = (row.get("status", "ok") if str(row.get("status", "ok")).startswith("failed")
                else " ".join(f"{k}={row[k]:.3f}" for k in TASK_METRICS[args.task]))
        print(f"  seed {row['seed']} q{int(row['quantile'])} {row['criterion']:18} "
              f"K={row['K']} {body}  {tag}", flush=True)

    # ---- schedulable jobs: one GNN cell per (seed, quantile, criterion) so GNN
    #      training saturates every GPU; boost_full is one cheap job per seed.
    jobs = []
    for s in seeds:
        seed_done = {(c, q) for (c, q, ss) in done if ss == int(s)}
        if args.boost_full and any(("boost_full", float(q)) not in seed_done for q in args.quantiles):
            jobs.append(("boost", int(s), None, None))
        for q in args.quantiles:
            for c in args.criteria:
                if (c, float(q)) not in seed_done:
                    jobs.append(("gnn", int(s), float(q), c))

    if not jobs:
        print("nothing to do -- all cells already cached", flush=True)
    elif args.max_parallel <= 1:
        data = _prepare(args); prep = {}
        for kind, s, q, c in jobs:
            if s not in prep:
                prep[s] = _seed_prep(s, args, data)
            P = prep[s]
            try:
                if kind == "boost":
                    for row in _boost_rows(s, args, P):
                        record(row, False)
                else:
                    record(_gnn_row(s, q, c, args, P), True)
            except Exception as e:                       # noqa: BLE001 -- same contract as the workers
                record(_failed_row(s, q, c, args, P["Kq"].get(q, -1), e), True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        job_q, res_q = ctx.Queue(), ctx.Queue()
        gpus = args.gpus or [0]
        n_workers = min(args.max_parallel, len(jobs))
        for j in jobs:
            job_q.put(j)
        for _ in range(n_workers):
            job_q.put(None)            # one sentinel per worker
        procs = []
        for i in range(n_workers):
            gpu = gpus[i % len(gpus)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)   # inherited by the child at start()
            p = ctx.Process(target=_worker_loop, args=(job_q, res_q, args))
            p.start(); procs.append(p)
            print(f"  worker {i} -> GPU {gpu}", flush=True)
        finished = 0
        while finished < n_workers:
            kind, payload, extra = res_q.get()
            if kind == "row":
                record(payload, extra)
            elif kind == "worker_done":
                finished += 1
        for p in procs:
            p.join()

    print(f"\ndone -> {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
