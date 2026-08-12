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

Parallelism: each **seed** is one work unit run in its own spawned process pinned
to a GPU (`CUDA_VISIBLE_DEVICES` set in the parent before start -- see
train_ensemble_boost.py for why the initializer approach fails on Linux). Workers
stream finished rows back over a queue; the PARENT is the sole CSV writer, so the
incremental/resumable CSV stays race-free. Re-running skips finished (criterion,
quantile, seed) cells.

    python scripts/modeling/train/run_quantile_criteria_sweep.py \
        --regime inductive --boost-full \
        --mol-embeddings data/embeddings/molecules/chemberta_77m_m2or.npz \
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

from orbind.gnn_extractor import GnnSignedExtractor          # noqa: E402
from orbind.mol_selection import CRITERIA, quality_K         # noqa: E402
from orbind.baselines import train_boost                     # noqa: E402
from orbind.dataset import metrics, load_npz_dict            # noqa: E402
from orbind.regimes import full_full_pairs, load_split       # noqa: E402

METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
REGIME_KEY = {"inductive": "inductive_molecule_v5", "transductive": "transductive"}
DEFAULT_REPEATS = {"inductive": [42, 43, 44, 45, 46], "transductive": [1, 2, 3, 4, 5]}


def _pair_matrix(emb: dict, keys) -> np.ndarray:
    return np.stack([emb[k] for k in keys]).astype(np.float32)


def _fmt(sec: float) -> str:
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def _prepare(args):
    """Load the pool + embeddings + coverage mask once (per process)."""
    pairs = full_full_pairs(pool_fold=args.pool_fold)
    mol_emb = load_npz_dict(str(_root / args.mol_embeddings))
    prot_emb = load_npz_dict(str(_root / args.prot_embeddings))
    cov = pairs["inchikey"].isin(mol_emb).to_numpy() & pairs["receptor"].isin(prot_emb).to_numpy()
    return pairs, mol_emb, prot_emb, cov


def _run_seed(seed, args, per_seed_done, emit, data):
    """Compute one seed's whole sub-sweep, calling `emit(row_dict)` per finished
    cell. `per_seed_done` is the set of (criterion, quantile) already cached for
    this seed (skipped). `data` = (pairs, mol_emb, prot_emb, cov)."""
    regime_key = REGIME_KEY[args.regime]
    pairs, mol_emb, prot_emb, cov = data
    ik = pairs["inchikey"].to_numpy(); rc = pairs["receptor"].to_numpy()
    lab = pairs["label"].to_numpy().astype(np.float32)

    tr, va, te = (np.asarray(a)[cov[np.asarray(a)]] for a in load_split(regime_key, seed))
    y_tr, y_te = lab[tr], lab[te]
    Xm_tr, Xm_te = _pair_matrix(mol_emb, ik[tr]), _pair_matrix(mol_emb, ik[te])
    Xp_tr_raw, Xp_te_raw = _pair_matrix(prot_emb, rc[tr]), _pair_matrix(prot_emb, rc[te])

    uniq = pd.unique(ik[tr]); loc = {k: i for i, k in enumerate(uniq)}
    cov_counts = np.bincount([loc[k] for k in ik[tr]], minlength=len(uniq))
    Kq = {q: quality_K(cov_counts, q / 100.0) for q in args.quantiles}

    if args.boost_full and any(("boost_full", q) not in per_seed_done for q in args.quantiles):
        sc = train_boost(np.concatenate([Xp_tr_raw, Xm_tr], 1), y_tr,
                         np.concatenate([Xp_te_raw, Xm_te], 1), seed=seed)
        mb = metrics(y_te, sc)
        for q in args.quantiles:
            if ("boost_full", q) not in per_seed_done:
                emit({"criterion": "boost_full", "quantile": q, "K": len(uniq), "seed": seed,
                      **{k: float(mb[k]) for k in METRICS}}, heavy=False)

    for q in args.quantiles:
        for crit in args.criteria:
            if (crit, q) in per_seed_done:
                continue
            ext = GnnSignedExtractor(
                name="gnn", protein_path=args.prot_embeddings,
                molecule_path=args.mol_embeddings, q=q / 100.0, criterion=crit,
                n_models=args.n_models, epochs=args.epochs, emit="prot")
            Zp_tr, _Zp_va, Zp_te = ext.fit_transform(pairs, tr, va, te, seed)
            sc = train_boost(np.concatenate([Zp_tr, Xm_tr], 1), y_tr,
                             np.concatenate([Zp_te, Xm_te], 1), seed=seed)
            m = metrics(y_te, sc)
            emit({"criterion": crit, "quantile": q, "K": int(Kq[q]), "seed": seed,
                  **{k: float(m[k]) for k in METRICS}}, heavy=True)


def _worker(q, seed, args, per_seed_done):
    """Spawned process: CUDA_VISIBLE_DEVICES already pinned by the parent."""
    data = _prepare(args)
    _run_seed(seed, args, per_seed_done, lambda row, heavy: q.put(("row", row, heavy)), data)
    q.put(("done", seed, None))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["inductive", "transductive"], default="inductive")
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

    regime_key = REGIME_KEY[args.regime]
    seeds = args.seeds or DEFAULT_REPEATS[args.regime]
    mol_stem = pathlib.Path(args.mol_embeddings).stem
    out = pathlib.Path(args.out) if args.out else (
        _root / "results/graph/full_full/v7/protein_based_graph"
        / f"metrics_{regime_key}__{mol_stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"regime={args.regime} ({regime_key})  seeds={seeds}  quantiles={args.quantiles}\n"
          f"criteria={args.criteria}\nmol={args.mol_embeddings}\nout={out}\n"
          f"max_parallel={args.max_parallel}  gpus={args.gpus}", flush=True)

    # ---- resumable state (parent is the sole reader/writer) --------------
    rows, done = [], set()
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {(r["criterion"], float(r["quantile"]), int(r["seed"])) for _, r in prev.iterrows()}
        print(f"resuming: {len(done)} cells already in {out.name}", flush=True)

    def save():
        tmp = out.with_suffix(".tmp.csv")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(out)

    # progress counter over HEAVY (GNN) cells only
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
        print(f"  seed {row['seed']} q{int(row['quantile'])} {row['criterion']:18} "
              + " ".join(f"{k}={row[k]:.3f}" for k in METRICS) + f"  {tag}", flush=True)

    def per_seed_done(seed):
        return {(c, q) for (c, q, s) in done if s == int(seed)}

    todo = [s for s in seeds if any((c, float(q), int(s)) not in done
                                    for q in args.quantiles for c in args.criteria)]

    if args.max_parallel <= 1 or len(todo) <= 1:
        data = _prepare(args)
        for seed in todo:
            print(f"\n--- seed {seed} ---", flush=True)
            _run_seed(seed, args, per_seed_done(seed), record, data)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        gpus = args.gpus or [0]
        procs, pending, gi = {}, list(todo), 0

        def spawn(seed):
            nonlocal gi
            gpu = gpus[gi % len(gpus)]; gi += 1
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)   # baked into the child at start
            p = ctx.Process(target=_worker, args=(q, seed, args, per_seed_done(seed)))
            p.start(); procs[seed] = p
            print(f"  launched seed {seed} -> GPU {gpu}", flush=True)

        while pending and len(procs) < args.max_parallel:
            spawn(pending.pop(0))
        while procs:
            kind, payload, extra = q.get()
            if kind == "row":
                record(payload, extra)
            elif kind == "done":
                procs.pop(payload).join()
                if pending:
                    spawn(pending.pop(0))

    print(f"\ndone -> {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
