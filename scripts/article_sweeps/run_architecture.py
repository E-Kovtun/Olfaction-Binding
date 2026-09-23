#!/usr/bin/env python
"""WHICH MESSAGE-PASSING OPERATOR: the architecture ablation, six numbers per row.

    .venv/bin/python scripts/article_sweeps/run_architecture.py \\
        --dataset m2or cc hc --regime transductive inductive \\
        --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \\
        --seeds 42 43 44 45 46 --max-parallel 4 --gpus 0 1 2 3

WHAT MOVES AND WHAT DOES NOT. The operator moves: GraphSAGE (ours), GAT, GraphConv,
GIN. Everything else is pinned -- the signed two-stack structure, the subtraction, the
decoder, the epoch budget, the folds, the boosting head, and above all THE GRAPH, whose
construction is fixed per dataset at the point the paper reports (`VARIANT` below). An
ablation in which the graph and the operator move together answers neither question.

WHY THESE FOUR. `orbind.gnn_extractor.CONVS` carries the argument in full; in short,
GAT is the attention epoch of this project (v1--v5, `orbind/legacy/hetero_gat.py`) and
is ported with the two details that make attention work on a bipartite graph;
`graphconv` is the GCN-shaped operator that is actually defined on two node sets, plain
`GCNConv` being undefined here; `gin` is the molecular domain's standard and the most
expressive of the four.

THE `:paper` ROWS. Two of the operators are also run the way their papers run them:
`sage:paper` and `gat:paper` add NEIGHBOUR SAMPLING (fan-out 25 then 10, redrawn every
epoch, inference still full-neighbourhood) and PER-LAYER L2 NORMALISATION -- the two
things our encoder took from neither. They are extra rows, not a change to ours: the
question they answer is how much of the gap between "GraphSAGE" and "our GraphSAGE" is
the operator and how much is the regime around it. Both additions are opt-in fields on
the extractor, so nothing outside this sweep moves.

WHAT THE "SAGE" ROW IS. Stock `SAGEConv` -- mean aggregation, a root weight, no
post-normalisation -- inside OUR algorithm, which is not the paper's: no neighbourhood
sampling (full batch, so every train pair is both an edge and a target), no per-layer
L2 normalisation, signed stacks subtracted, bipartite and heterogeneous, LeakyReLU, an
MLP decoder at a fixed epoch budget. The full list is in `orbind/gnn_extractor.py`
under WHAT "GRAPHSAGE" MEANS HERE. This table compares operators inside one fixed
algorithm and claims nothing about reproducing any of their papers.

WIDTH IS FREE, DEPTH IS NOT. `--hidden` sweeps the width with no new code, and each
width is its own row. Depth is NOT a knob: the encoder is two layers by construction,
and making it variable is a refactor of the module rather than a flag. It is out of
this sweep deliberately rather than by oversight.

WHAT COMES OUT. One CSV per (dataset, regime) under `--out`, one row per
(operator, width, fold, seed, head, split), plus the boosting reference on the same
folds. `scripts/article_tables/08_architecture.py` renders the six-column table from
them: the metric of record per dataset, one column per (dataset, regime).

Resumable at cell granularity: rows already in the CSV are kept and only the missing
ones are trained, so adding an operator later costs that operator alone.
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
from orbind.gnn_extractor import CONVS, GnnSignedExtractor        # noqa: E402

#: The graph every operator is run on: the construction the paper reports, per dataset.
#: M2OR keeps its hub core; the insect matrices are complete, so there is nothing to
#: filter and q=0 is the whole graph. Pinned here rather than taken from the command
#: line because an operator comparison on two different graphs is not one.
VARIANT = {
    "m2or": dict(q=0.99, criterion="greedy_pair_cover", k_mode="coverage_quantile"),
    "cc": dict(q=0.0, criterion="coverage", k_mode="coverage_quantile"),
    "hc": dict(q=0.0, criterion="coverage", k_mode="coverage_quantile"),
}
#: What `:paper` adds to an operator. Fan-out 25 then 10 is GraphSAGE's own setting;
#: the normalisation is its per-layer L2. Pinned here so the two `:paper` rows differ
#: from their plain rows in exactly this and nothing else.
PAPER = dict(fanout=(25, 10), normalize_layers=True)
PAPER_SUFFIX = ":paper"
#: The rows this sweep trains by default: four operators as we run them, plus the two
#: whose papers define a regime we could adopt.
DEFAULT_SPECS = ("sage", "gat", "graphconv", "gin", "sage:paper", "gat:paper")

#: The head the paper compares on: our refined receptor beside the raw molecule.
DEFAULT_COMBOS = ("cls+mol",)
#: The width every reported number uses. Other widths are extra ROWS, not a rescaling
#: of the comparison.
DEFAULT_HIDDEN = (256,)


def _load_sweep():
    """The alpha sweep, as a module. It is the definition of a fold, a metric and a
    boosting reference in this repo; re-implementing any of them here would produce a
    table that looks comparable and is not."""
    path = _root / "scripts/modeling/train/run_alpha_gate_sweep.py"
    spec = importlib.util.spec_from_file_location("_alpha_sweep_for_arch", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _fmt(sec):
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def repeats_for(sw, ds, regime, args):
    """Which folds this (dataset, regime) has. M2OR's inductive 'folds' are the
    cold-molecule SEEDS 42-46, so they cannot be named by number across datasets."""
    if args.folds:
        return list(args.folds)
    reps = sw.REPEATS[ds].get(regime, [1, 2, 3, 4, 5])
    return reps[:args.n_folds] if args.n_folds else list(reps)


def parse_spec(spec):
    """`sage` -> ('sage', False); `sage:paper` -> ('sage', True). Refused otherwise,
    at parse time rather than three hours into a run."""
    conv, sep, tail = str(spec).partition(":")
    if conv not in CONVS:
        raise SystemExit(f"unknown operator {conv!r} in {spec!r}; have {list(CONVS)}")
    if sep and tail != "paper":
        raise SystemExit(f"unknown suffix {tail!r} in {spec!r}; only ':paper' exists")
    return conv, bool(sep)


def arch_label(conv, hidden, paper=False):
    """How a row is named. The width appears only when it is not the reported one, so
    the default table reads as operators and not as widths; `:paper` always appears,
    because two rows of one operator that differ in regime must never share a name."""
    name = conv if int(hidden) == DEFAULT_HIDDEN[0] else f"{conv}@{int(hidden)}"
    return name + PAPER_SUFFIX if paper else name


# ------------------------------------------------------------------ one cell

def _base(ds, regime, args, P, **rest):
    v = VARIANT[ds]
    return dict(dataset=ds, regime=regime, mol_source=args.mol_source,
                q=v["q"], criterion=v["criterion"], k_mode=v["k_mode"],
                n_models=args.n_models, epochs=args.epochs,
                n_receptors=len(P["order"]), status="ok", **rest)


def boost_rows(ds, regime, fold, seed, sw, args, P):
    """The no-graph reference on this fold: the boosting head over [prot || mol].

    It has no operator, so it is written once per (fold, seed) with conv='boost_full'
    and NaN width -- the reader shows it as the anchor row, not as a fifth operator.
    """
    task = sw.TASK[ds]
    X = {k: np.concatenate([P[f"Xp_{k}"], P[f"Xm_{k}"]], 1) for k in ("tr", "va", "te")}
    t0 = time.time()
    est = fit_boost(X["tr"], P["y_tr"], seed=seed, task=task)
    t_head = time.time() - t0
    rows = []
    for split in sw.splits_wanted(args, P):
        k = sw.SPLIT_SHORT[split]
        rows.append(_base(ds, regime, args, P, conv="boost_full", hidden=np.nan,
                          arch="boost_full", heads=np.nan, paper=False,
                          fanout="", normalize_layers=False,
                          fold=int(fold), seed=int(seed), combo="prot+mol",
                          split=split, t_graph=0.0, t_head=t_head,
                          n_rows=len(P[f"y_{k}"]),
                          **sw._score_split(P, predict_scores(est, X[k], task),
                                            task, split)))
    return rows


def graph_rows(ds, regime, fold, seed, spec, hidden, sw, args, P):
    """One operator at one width on one fold, then every head asked for.

    `alpha=None` is the pipeline graph exactly -- no gate branch. That is the right
    thing for an architecture sweep: the question is which operator aggregates the
    messages, and a frozen ESM branch on top would answer it through a second knob.
    """
    task = sw.TASK[ds]
    pp, mp = sw.paths(ds, args)
    v = VARIANT[ds]
    conv, paper = parse_spec(spec)
    extra = dict(fanout=tuple(args.fanout), normalize_layers=True) if paper else {}
    ext = GnnSignedExtractor(
        name="cls", protein_path=pp, molecule_path=mp,
        q=float(v["q"]), criterion=v["criterion"], k_mode=v["k_mode"],
        conv=conv, heads=args.heads, hidden=int(hidden),
        task=task, n_models=args.n_models, epochs=args.epochs, emit="prot",
        alpha=args.alpha, deterministic_init=args.seed_graph, **extra)
    t0 = time.time()
    Z = dict(zip(("tr", "va", "te"),
                 ext.fit_transform(P["pairs"], P["tr"], P["va"], P["te"], seed)))
    t_graph = time.time() - t0

    rows = []
    for combo in args.combos:
        t1 = time.time()
        est = fit_boost(sw._head_features(combo, Z["tr"], P, "tr"), P["y_tr"],
                        seed=seed, task=task)
        t_head = time.time() - t1
        for split in sw.splits_wanted(args, P):
            k = sw.SPLIT_SHORT[split]
            pred = predict_scores(est, sw._head_features(combo, Z[k], P, k), task)
            rows.append(_base(ds, regime, args, P, conv=conv, hidden=int(hidden),
                              arch=arch_label(conv, hidden, paper),
                              heads=int(args.heads), paper=paper,
                              fanout=("-".join(map(str, args.fanout)) if paper else ""),
                              normalize_layers=paper,
                              fold=int(fold), seed=int(seed), combo=combo,
                              split=split, t_graph=t_graph, t_head=t_head,
                              n_rows=len(P[f"y_{k}"]),
                              **sw._score_split(P, pred, task, split)))
    return rows


def failed_rows(ds, regime, fold, seed, spec, hidden, sw, args, P, err):
    """A cell that cannot be computed is an answer too -- NaN with the reason.

    Recording it stops a resume from retrying it forever and stops the table from
    quietly showing an operator's column as if it had simply not been run yet.
    """
    task = sw.TASK[ds]
    cols = sw.TASK_METRICS[task]
    conv, paper = parse_spec(spec)
    return [_base(ds, regime, args, P, conv=conv, hidden=int(hidden),
                  arch=arch_label(conv, hidden, paper), heads=int(args.heads),
                  paper=paper,
                  fanout=("-".join(map(str, args.fanout)) if paper else ""),
                  normalize_layers=paper,
                  fold=int(fold), seed=int(seed), combo=combo, split="test",
                  t_graph=np.nan, t_head=np.nan, n_rows=len(P["y_te"]),
                  **{c: np.nan for c in cols})
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
    """What makes a cell unique. The width is in it because two widths of one operator
    are two models; the head and the split are in it so a run that adds
    `cls+prot+mol` later fills those heads without retraining a single graph."""
    h = row["hidden"]
    return (str(row["arch"]), None if pd.isna(h) else int(h),
            int(row["fold"]), int(row["seed"]), str(row["combo"]), str(row["split"]))


def load_done(path):
    if not path.exists():
        return [], set()
    prev = pd.read_csv(path)
    missing = [c for c in ("conv", "hidden", "arch", "paper") if c not in prev.columns]
    if missing:
        raise SystemExit(
            f"{path} predates this script ({missing} absent). Move it aside or pass a "
            f"fresh --out: resuming onto it would mix two row formats silently.")
    return prev.to_dict("records"), {cell_key(r) for _, r in prev.iterrows()}


def write_config(path, args, xgb_version):
    """Provenance beside the CSV, named after it."""
    body = {**{k: v for k, v in vars(args).items() if not k.startswith("_")},
            "variant": VARIANT, "python": sys.executable, "xgboost": xgb_version,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (path.parent / (path.stem.replace("metrics_", "config_") + ".json")).write_text(
        json.dumps(body, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------ workers

def _run_job(job, sw, args, prep, data, ds, regime):
    kind, fold, seed, conv, hidden = job
    if fold not in prep:
        prep[fold] = sw._fold_prep(ds, regime, fold, args, data)
    P = prep[fold]
    if kind == "boost":
        return boost_rows(ds, regime, fold, seed, sw, args, P)
    try:
        return graph_rows(ds, regime, fold, seed, conv, hidden, sw, args, P)
    except Exception as e:                  # noqa: BLE001 -- one cell must not kill a sweep
        return failed_rows(ds, regime, fold, seed, conv, hidden, sw, args, P, e)


def _worker(job_q, res_q, args, ds, regime):
    """Persistent worker. The parent set CUDA_VISIBLE_DEVICES before start(), so every
    torch and XGBoost op in here lands on this worker's card. `worker_done` is sent
    from `finally`: a worker that dies silently would leave the parent blocked."""
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

def plan(folds, done, args):
    """Every cell still missing, boost first so the anchor exists early."""
    jobs = []
    for fold in folds:
        for seed in args.seeds:
            if args.boost_full and ("boost_full", None, fold, seed,
                                    "prot+mol", "test") not in done:
                jobs.append(("boost", fold, seed, None, None))
            for spec in args.conv:
                conv, paper = parse_spec(spec)
                for hidden in args.hidden:
                    name = arch_label(conv, hidden, paper)
                    if not all((name, int(hidden), fold, seed, c, "test") in done
                               for c in args.combos):
                        jobs.append(("gnn", fold, seed, spec, int(hidden)))
    return jobs


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
    jobs = plan(folds, done, args)
    heavy = sum(1 for j in jobs if j[0] == "gnn")
    v = VARIANT[ds]
    print(f"\n=== {ds} / {regime}: {len(folds)} folds x {len(args.seeds)} seeds x "
          f"{len(args.conv)} row spec(s) x {len(args.hidden)} width(s) "
          f"-> {heavy} graphs to train ({len(jobs)} jobs)"
          f"\n    graph fixed at q={v['q']}, criterion={v['criterion']}, "
          f"k_mode={v['k_mode']}\n    -> {path}")
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
            head = new[0]
            body = head["status"] if str(head["status"]).startswith("failed") else \
                " ".join(f"{m}={head[m]:.3f}" for m in sw.TASK_METRICS[sw.TASK[ds]]
                         if m in head and np.isfinite(head[m]))
            print(f"  f{head['fold']} s{head['seed']} {head['arch']:<14} {body}   "
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
    ap.add_argument("--dataset", nargs="+", default=["m2or", "cc", "hc"],
                    choices=["m2or", "cc", "hc"])
    ap.add_argument("--regime", nargs="+", default=["transductive", "inductive"])
    ap.add_argument("--conv", nargs="+", default=list(DEFAULT_SPECS),
                    metavar="SPEC",
                    help=f"rows to train: an operator from {list(CONVS)}, optionally "
                         f"with ':paper' for neighbour sampling + per-layer L2 "
                         f"normalisation. Default: {list(DEFAULT_SPECS)}")
    ap.add_argument("--fanout", type=int, nargs=2, default=list(PAPER["fanout"]),
                    metavar=("L1", "L2"),
                    help="the ':paper' rows' fan-out, layer 1 then layer 2. "
                         "GraphSAGE's own setting is 25 10")
    ap.add_argument("--hidden", type=int, nargs="+", default=list(DEFAULT_HIDDEN),
                    help="widths. Each is its own ROW: 256 is what the paper reports, "
                         "and another width answers a different question")
    ap.add_argument("--heads", type=int, default=4,
                    help="GAT heads on layer 1, concatenated back to --hidden. "
                         "Ignored by every other operator")
    ap.add_argument("--combos", nargs="+", default=list(DEFAULT_COMBOS),
                    choices=["cls+mol", "cls+prot+mol"])
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="literal fold ids; cannot be mixed across datasets")
    ap.add_argument("--n-folds", type=int, default=None,
                    help="the first N folds of whatever this cell uses")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="MODEL seeds, averaged inside a fold by the reader")
    ap.add_argument("--alpha", type=float, default=None,
                    help="v8 gate on the graph output. Default None = the pipeline "
                         "graph, which is what an architecture sweep should keep")
    ap.add_argument("--seed-graph", action="store_true",
                    help="seed the graph init from the model seed. Worth it here: the "
                         "init lottery is the noise this table is most exposed to")
    ap.add_argument("--n-models", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--mol-source", default="chemberta",
                    choices=["chemberta", "gin", "ecfp"])
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
    ap.add_argument("--out", default="results/article_sweeps/architecture")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    # Fail here, not three hours from now inside an XGBoost destructor.
    xgb_version = check_xgboost_version()
    sw = _load_sweep()
    if args.folds:
        args.n_folds = None
    for ds in args.dataset:
        for regime in args.regime:
            sweep(ds, regime, sw, args, xgb_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
