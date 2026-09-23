#!/usr/bin/env python
"""How the GRAPH IS BUILT: a sweep over (molecule-ranking criterion x quantile).

The alpha sweep moves what the receptor vector is MIXED FROM. This one moves which
edges exist at all. The pipeline's message-passing graph keeps only K train molecules;
the quantile says HOW MANY survive and the criterion says WHICH (orbind/mol_selection).
Every number the paper reports stands on one point of this grid -- `q=0.99` with
`greedy_pair_cover` on M2OR, `q=0` with `coverage` on the insects -- and this script is
what makes that a measured choice instead of an inherited one.

WHY A SECOND SWEEP AND NOT A FLAG ON THE FIRST. The alpha sweep hardwires the edge
variant (`VARIANTS[args._variant]`) precisely so that a dial run cannot silently change
two things at once. Rather than loosen that, this script borrows the alpha sweep's fold
preparation, metric battery and boosting reference **as a module** and varies only the
construction knob. Same folds, same coverage mask, same head, same columns -- so a row
here is comparable with a row there, and nothing in the older code had to move.

    .venv/bin/python scripts/article_sweeps/run_quantile_criteria.py \\
        --dataset m2or --regime inductive \\
        --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \\
        --max-parallel 4 --gpus 0 1 2 3

One CSV per (dataset, regime) under results/article_sweeps/quantile_criteria/, one row
per (criterion, quantile, fold, seed, head, split). Resumable at cell granularity: a
re-run skips what is already there, so a killed sweep is restarted by repeating the
command. `notebooks/article_figures/quantile_criteria.ipynb` draws it and aggregates
nothing -- `quantile_grid.py` in this folder does that.

THREE THINGS THAT ARE EASY TO GET WRONG HERE

* **Quantiles are FRACTIONS** (0.99), because that is what the extractor and
  `VARIANTS` take. The old study script took percents; mixing the two silently
  gives a different graph.
* **The paper's own construction is a grid point, not a separate arm.** On M2OR
  that is `--criteria greedy_pair_cover --quantiles ... 0.99`; leave it out of the
  grid and the figure has nothing to compare against.
* **On the insect matrices most criteria are ties.** CC/HC are complete matrices, so
  coverage is constant over train molecules: `coverage_quantile` cuts nothing and the
  coverage-shaped criteria all score every molecule the same. The run probes this
  before it starts and names the flat ones -- read the K axis there, not the
  criterion comparison. Use `--k-mode fraction` to get a knob that actually moves.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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

from orbind.baselines import (check_xgboost_version, fit_boost,   # noqa: E402
                              predict_scores)
from orbind.gnn_extractor import GnnSignedExtractor               # noqa: E402
from orbind.mol_selection import (CRITERIA, K_MODES,              # noqa: E402
                                  compute_mol_scores, resolve_K)

#: Default grid. Coarse at the bottom (nothing happens there) and dense at the top,
#: where the paper's own point sits and where K falls off a cliff.
QUANTILES = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
#: The head the paper compares on: our refined receptor beside the raw molecule, which
#: is the same shape as a competitor's own cls row.
DEFAULT_COMBOS = ("cls+mol",)


def _load_sweep():
    """The alpha sweep, as a module. It is the definition of a fold, a metric and a
    boosting reference in this repo; re-implementing any of them here would produce a
    table that looks comparable and is not."""
    path = _root / "scripts/modeling/train/run_alpha_gate_sweep.py"
    spec = importlib.util.spec_from_file_location("_alpha_sweep_for_qsweep", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _fmt(sec):
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


# ------------------------------------------------------------------ the grid

def repeats_for(sw, ds, regime, args):
    """Which folds this (dataset, regime) has. M2OR's inductive 'folds' are the
    cold-molecule SEEDS 42-46, so they cannot be named by number across datasets --
    `--n-folds` takes the first N of whatever this cell actually uses."""
    if args.folds:
        return list(args.folds)
    reps = sw.REPEATS[ds].get(regime, [1, 2, 3, 4, 5])
    return reps[:args.n_folds] if args.n_folds else list(reps)


def cell_K(P, q, k_mode):
    """How many train molecules survive this (q, k_mode) on this fold.

    Recorded on every row because it is the quantity the knob actually controls, and
    because two quantiles that resolve to the same K are one experiment, not two."""
    mols = pd.unique(P["mol_tr"])
    loc = {m: i for i, m in enumerate(mols)}
    cov = np.bincount([loc[m] for m in P["mol_tr"]], minlength=len(mols))
    return int(resolve_K(cov, float(q), k_mode))


def flat_criteria(P, task, criteria):
    """Criteria that give every train molecule the SAME score on this fold.

    Not a failure -- an honest reading of a complete matrix. But a figure that puts
    seven such criteria side by side implies they rank differently, when what actually
    separated them was an arbitrary tie-break.
    """
    mols = sorted(pd.unique(P["mol_tr"]))
    recs = sorted(pd.unique(P["rec_tr"]))
    mi = {m: i for i, m in enumerate(mols)}
    ri = {r: i for i, r in enumerate(recs)}
    sc = compute_mol_scores(np.array([mi[m] for m in P["mol_tr"]]),
                            np.array([ri[r] for r in P["rec_tr"]]),
                            np.asarray(P["y_tr"]), len(mols), len(recs),
                            need_greedy=False,
                            pos_threshold=(0.0 if task == "regression" else None))
    flat = [c for c in criteria if c != "greedy_pair_cover"
            and len(np.unique(np.round(sc["SCORE"][c], 10))) == 1]
    if "greedy_pair_cover" in criteria and len(np.unique(sc["cov"])) == 1:
        flat.append("greedy_pair_cover")
    return flat


# ------------------------------------------------------------------ one cell

def _base(ds, regime, args, P, **rest):
    return dict(dataset=ds, regime=regime, mol_source=args.mol_source,
                k_mode=args.k_mode, n_models=args.n_models, epochs=args.epochs,
                n_receptors=len(P["order"]), status="ok", **rest)


def boost_rows(ds, regime, fold, seed, sw, args, P):
    """The no-graph reference on this fold: the boosting head over [prot || mol].

    It does not depend on the construction knob, so it is written once per (fold, seed)
    with an empty criterion and a NaN quantile -- the notebook draws it as a horizontal
    line, not as a seventh curve.
    """
    task = sw.TASK[ds]
    X = {k: np.concatenate([P[f"Xp_{k}"], P[f"Xm_{k}"]], 1) for k in ("tr", "va", "te")}
    t0 = time.time()
    est = fit_boost(X["tr"], P["y_tr"], seed=seed, task=task)
    t_head = time.time() - t0
    rows = []
    for split in sw.splits_wanted(args, P):
        k = sw.SPLIT_SHORT[split]
        rows.append(_base(ds, regime, args, P, criterion="boost_full",
                          quantile=np.nan, K=-1, fold=int(fold), seed=int(seed),
                          combo="prot+mol", split=split, t_graph=0.0, t_head=t_head,
                          n_rows=len(P[f"y_{k}"]),
                          **sw._score_split(P, predict_scores(est, X[k], task),
                                            task, split)))
    return rows


def graph_rows(ds, regime, fold, seed, crit, q, sw, args, P):
    """One (criterion, quantile) graph on one fold, then every head asked for.

    `alpha=None` is the pipeline graph exactly -- no gate branch, the historical
    computation. That is the right thing for a CONSTRUCTION sweep: the question is
    which edges to build from, and adding a frozen ESM branch on top would answer it
    through a second knob. It is also the same object the node dial calls `prot_mix=1`.
    """
    task = sw.TASK[ds]
    pp, mp = sw.paths(ds, args)
    ext = GnnSignedExtractor(
        name="cls", protein_path=pp, molecule_path=mp,
        q=float(q), criterion=crit, k_mode=args.k_mode,
        task=task, n_models=args.n_models, epochs=args.epochs, emit="prot",
        alpha=args.alpha, deterministic_init=args.seed_graph,
        # the `random` control draws a different set of hubs per seed, so its spread
        # over seeds is visible instead of one lucky draw standing in for a baseline.
        # Every other criterion ignores this.
        select_seed=int(seed))
    t0 = time.time()
    Z = dict(zip(("tr", "va", "te"),
                 ext.fit_transform(P["pairs"], P["tr"], P["va"], P["te"], seed)))
    t_graph = time.time() - t0

    rows = []
    K = cell_K(P, q, args.k_mode)
    for combo in args.combos:
        t1 = time.time()
        est = fit_boost(sw._head_features(combo, Z["tr"], P, "tr"), P["y_tr"],
                        seed=seed, task=task)
        t_head = time.time() - t1
        for split in sw.splits_wanted(args, P):
            k = sw.SPLIT_SHORT[split]
            pred = predict_scores(est, sw._head_features(combo, Z[k], P, k), task)
            rows.append(_base(ds, regime, args, P, criterion=crit, quantile=float(q),
                              K=K, fold=int(fold), seed=int(seed), combo=combo,
                              split=split, t_graph=t_graph, t_head=t_head,
                              n_rows=len(P[f"y_{k}"]),
                              **sw._score_split(P, pred, task, split)))
    return rows


def failed_rows(ds, regime, fold, seed, crit, q, sw, args, P, err):
    """A cell that cannot be computed is an answer too -- NaN with the reason.

    The one that actually happens: at a tiny K every kept molecule can land on one
    side of `edge_threshold`, and signed message passing needs both signs. Recording
    it stops a resume from retrying it forever and stops the figure from quietly
    interpolating across the hole.
    """
    task = sw.TASK[ds]
    cols = sw.TASK_METRICS[task]
    return [_base(ds, regime, args, P, criterion=crit, quantile=float(q),
                  K=cell_K(P, q, args.k_mode), fold=int(fold), seed=int(seed),
                  combo=combo, split="test", t_graph=np.nan, t_head=np.nan,
                  n_rows=len(P["y_te"]), **{c: np.nan for c in cols})
            | {"status": f"failed: {type(err).__name__}: {err}"}
            for combo in args.combos]


# ------------------------------------------------------------------ bookkeeping

def out_path(ds, regime, args):
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = _root / out
    out.mkdir(parents=True, exist_ok=True)
    return out / f"metrics_{ds}_{regime}__{args.mol_source}.csv"


def cell_key(row):
    """What makes a cell unique. The head and the split are in it: a run that adds
    `cls+prot+mol` later must fill those heads without redoing the graphs."""
    q = row["quantile"]
    return (str(row["criterion"]), None if pd.isna(q) else round(float(q), 6),
            int(row["fold"]), int(row["seed"]), str(row["combo"]), str(row["split"]))


def load_done(path):
    if not path.exists():
        return [], set()
    prev = pd.read_csv(path)
    return prev.to_dict("records"), {cell_key(r) for _, r in prev.iterrows()}


def write_config(path, args, xgb_version):
    """Provenance beside the CSV, named after it. One config per directory would be
    overwritten by the next (dataset, regime) and would then describe a run that is
    not the one in the file next to it."""
    body = {**{k: v for k, v in vars(args).items() if not k.startswith("_")},
            "python": sys.executable, "xgboost": xgb_version,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (path.parent / (path.stem.replace("metrics_", "config_") + ".json")).write_text(
        json.dumps(body, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------ workers

def _run_job(job, sw, args, prep, data, ds, regime):
    kind, fold, seed, q, crit = job
    if fold not in prep:
        prep[fold] = sw._fold_prep(ds, regime, fold, args, data)
    P = prep[fold]
    if kind == "boost":
        return boost_rows(ds, regime, fold, seed, sw, args, P)
    try:
        return graph_rows(ds, regime, fold, seed, crit, q, sw, args, P)
    except Exception as e:                  # noqa: BLE001 -- one cell must not kill a sweep
        return failed_rows(ds, regime, fold, seed, crit, q, sw, args, P, e)


def _worker(job_q, res_q, args, ds, regime):
    """Persistent worker. The parent set CUDA_VISIBLE_DEVICES before start(), so every
    torch and XGBoost op in here lands on this worker's card.

    `worker_done` is sent from `finally`: a worker that dies silently leaves the parent
    blocked forever in its collection loop, which is exactly what an uncaught
    degenerate-graph error used to do in the older sweep.
    """
    try:
        sw = _load_sweep()
        data = sw._prepare(ds, args)
        prep = {}
        while True:
            job = job_q.get()
            if job is None:
                break
            res_q.put(("rows", _run_job(job, sw, args, prep, data, ds, regime),
                       job[0] == "gnn"))
    finally:
        res_q.put(("worker_done", None, None))


# ------------------------------------------------------------------ the sweep

def sweep(ds, regime, sw, args, xgb_version):
    path = out_path(ds, regime, args)
    rows, done = load_done(path)
    if rows:
        print(f"  resuming: {len(rows)} rows already in {path.name}")

    def save():
        tmp = path.with_suffix(".tmp.csv")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(path)

    folds = repeats_for(sw, ds, regime, args)
    jobs = []
    for fold in folds:
        for seed in args.seeds:
            if args.boost_full and                     ("boost_full", None, fold, seed, "prot+mol", "test") not in done:
                jobs.append(("boost", fold, seed, None, None))
            for q in args.quantiles:
                for crit in args.criteria:
                    if not all((crit, round(float(q), 6), fold, seed, c, "test") in done
                               for c in args.combos):
                        jobs.append(("gnn", fold, seed, float(q), crit))

    heavy = sum(1 for j in jobs if j[0] == "gnn")
    print(f"\n=== {ds} / {regime}: {len(folds)} folds x {len(args.seeds)} seeds x "
          f"{len(args.quantiles)} quantiles x {len(args.criteria)} criteria "
          f"-> {heavy} graphs to train ({len(jobs)} jobs)\n    -> {path}")
    if not jobs:
        print("  nothing to do -- every cell is cached")
        return
    write_config(path, args, xgb_version)

    t0, state = time.time(), {"n": 0}

    def record(new, was_heavy):
        added = 0
        for r in new:
            k = cell_key(r)
            if k in done:
                continue
            rows.append(r); done.add(k); added += 1
        if added:
            save()
        if was_heavy:
            state["n"] += 1
            el = time.time() - t0
            eta = el / state["n"] * (heavy - state["n"])
            # THE TEST ROW, not new[0]: `splits_wanted` puts train first when train
            # scoring is on, and a progress line showing the training score reads as
            # a result when it is not one.
            head = next((r for r in new if str(r.get("split")) == "test"), new[0])
            body = head["status"] if str(head["status"]).startswith("failed") else \
                " ".join(f"{m}={head[m]:.3f}" for m in sw.TASK_METRICS[sw.TASK[ds]]
                         if m in head and np.isfinite(head[m]))
            print(f"  f{head['fold']} s{head['seed']} [{head['split']}] q={head['quantile']:<5g} "
                  f"{head['criterion']:<18} K={head['K']:<4} {body}   "
                  f"[{state['n']}/{heavy} {_fmt(el)} elapsed, ETA {_fmt(eta)}]",
                  flush=True)

    if args.max_parallel <= 1:
        data = sw._prepare(ds, args)
        prep = {}
        for job in jobs:
            record(_run_job(job, sw, args, prep, data, ds, regime), job[0] == "gnn")
        return

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    job_q, res_q = ctx.Queue(), ctx.Queue()
    gpus = args.gpus or [0]
    n = min(args.max_parallel, len(jobs))
    for j in jobs:
        job_q.put(j)
    for _ in range(n):
        job_q.put(None)                       # one sentinel per worker
    procs = []
    for i in range(n):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
        p = ctx.Process(target=_worker, args=(job_q, res_q, args, ds, regime))
        p.start(); procs.append(p)
        print(f"  worker {i} -> GPU {gpus[i % len(gpus)]}")
    finished = 0
    while finished < n:
        kind, payload, was_heavy = res_q.get()
        if kind == "rows":
            record(payload, was_heavy)
        else:
            finished += 1
    for p in procs:
        p.join()


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["m2or"], choices=["m2or", "cc", "hc"])
    ap.add_argument("--regime", nargs="+", default=["inductive", "transductive"])
    ap.add_argument("--quantiles", type=float, nargs="+", default=QUANTILES,
                    help="FRACTIONS, not percents -- 0.99 is the paper's M2OR point")
    ap.add_argument("--criteria", nargs="+", default=list(CRITERIA), choices=list(CRITERIA))
    ap.add_argument("--k-mode", default=None, choices=list(K_MODES),
                    help="how q becomes K. Default: coverage_quantile on m2or, "
                         "fraction on cc/hc -- their matrices are complete, so the "
                         "coverage quantile keeps every molecule at every q and the "
                         "whole sweep collapses to one point")
    ap.add_argument("--combos", nargs="+", default=list(DEFAULT_COMBOS),
                    choices=["cls+mol", "cls+prot+mol"])
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="literal fold ids; cannot be mixed across datasets")
    ap.add_argument("--n-folds", type=int, default=None,
                    help="the first N folds of whatever this cell uses -- the portable "
                         "way to ask for a cheaper slice")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="MODEL seeds, averaged inside a fold by the reader. 42 is what "
                         "every reported boosting number uses")
    ap.add_argument("--alpha", type=float, default=None,
                    help="v8 gate on the graph output. Default None = the pipeline "
                         "graph, which is what a construction sweep should move")
    ap.add_argument("--seed-graph", action="store_true",
                    help="seed the graph init from the model seed (removes the init "
                         "lottery; changes numbers against older unseeded runs)")
    ap.add_argument("--n-models", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--mol-source", default="chemberta", choices=["chemberta", "gin", "ecfp"])
    ap.add_argument("--mol-embeddings", default=None)
    ap.add_argument("--prot-embeddings", default=None,
                    help="THE SAME npz the comparison run used -- it decides the "
                         "coverage mask. ESM3: 'data/embeddings/proteins/esm3_{ds}.npz'")
    ap.add_argument("--pool-fold", type=int, default=1)
    ap.add_argument("--no-boost-full", dest="boost_full", action="store_false",
                    help="skip the [prot||mol] reference rows")
    ap.add_argument("--no-train-scores", dest="score_train", action="store_false",
                    help="score val and test only")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--gpus", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="results/article_sweeps/quantile_criteria")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    # Fail here, not three hours from now inside an XGBoost destructor.
    xgb_version = check_xgboost_version()
    sw = _load_sweep()
    if args.folds:
        args.n_folds = None
    requested_k_mode = args.k_mode
    for ds in args.dataset:
        # Per dataset, and re-derived every time: M2OR's coverage is heavy-tailed so a
        # coverage quantile cuts, while the insect matrices are complete so only a
        # fraction does. Deriving it into `args` once would hand the second dataset
        # the first one's reading of the same number.
        args.k_mode = requested_k_mode or (
            "coverage_quantile" if ds == "m2or" else "fraction")
        for regime in args.regime:
            data = sw._prepare(ds, args)
            P = sw._fold_prep(ds, regime, repeats_for(sw, ds, regime, args)[0],
                              args, data)
            flat = flat_criteria(P, sw.TASK[ds], args.criteria)
            if flat:
                print(f"\nWARNING: on {ds}/{regime} these criteria score every train "
                      f"molecule the same: {flat}.\n         Their selection is an "
                      f"arbitrary tie-break, not a ranking -- read the K axis, not the "
                      f"criterion comparison.")
            if args.k_mode == "coverage_quantile" and ds != "m2or":
                print(f"WARNING: k_mode=coverage_quantile on {ds}'s complete matrix "
                      f"keeps every molecule at every q.")
            del data, P
            sweep(ds, regime, sw, args, xgb_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
