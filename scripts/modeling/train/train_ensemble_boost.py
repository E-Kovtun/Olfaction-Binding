"""Configurable multi-source boosting ensemble (see orbind/ensemble.py).

Each --source registers one embedding extractor under a short name; the
1-based position among --source flags is what the --combos digit-string
mini-language refers to. Two families of source type:

  esm, gin              -- label-independent, entity-level lookups
                            ("name=type:path[:model_name[:pooling]]").
  attn_noisy_or, attn_lse -- supervised, pair-level "cls" sources (see
                            orbind/attention_extractor.py). Each trains
                            n_models (default 5) independently-seeded
                            whole-train torch models -- always concurrent --
                            and concatenates their pooled embeddings; every
                            model embeds train/val/test alike (accepting
                            mild train-row leakage, bounded by early
                            stopping, same tradeoff ProSmith's own cls model
                            makes). n_models=1 reduces to ProSmith's exact
                            scheme. Bare "name=type" uses its baked-in
                            embedding paths (LoRaX ESM-1b mean + our per-atom
                            GIN) and n_models=5; pass
                            "name=type:protein_path:molecule_sites_path:n_models"
                            to override any of them (leave a field blank to
                            keep its default, e.g. "cls=attn_lse:::1").

Two regimes, two split "traditions" (see orbind/dataset.py vs orbind/regimes.py
for why they aren't unified):

  curated_full  -- pairs from a csv (pairs_curated.csv / pairs_m2or_full.csv),
                   split via --split {stratified,group_molecule,group_receptor}
                   + --seeds (fresh random split per seed).
  full_full     -- pairs reconstructed from LoRaX's own data
                   (orbind.regimes.full_full_pairs), split via
                   --full-full-mode {transductive,inductive_molecule} +
                   --repeats (LoRaX fold 1-5 for transductive, our own
                   cold-molecule seed for inductive_molecule).

Every invocation creates one timestamped run folder under
results/ensemble_logs/<run_id>/:

  config.json           -- every CLI arg, for reproducibility
  log.txt                -- this process's own stdout (setup + final summary)
  logs/repeat_{R}.log     -- one per repeat, captures its combo/ensemble lines
                            even when --max-parallel runs it in another process
  metrics.csv             -- one row per (repeat, combo) + (repeat, ensemble method)
  checkpoints/repeat_{R}/
    boost_{combo}.json           -- one XGBoost booster per combo (unless --skip-checkpoints)
    attn_{source}_model{k}.pt     -- one torch state_dict per model, per pair-level source

--max-parallel N runs repeats concurrently as separate OS processes (own CUDA
context each); a pair-level source's n_models are always trained concurrently
regardless (see orbind/attention_extractor.py), independent of
this flag. With >1 GPU visible (or --gpus explicitly given), repeats are
pinned round-robin one GPU per worker process via CUDA_VISIBLE_DEVICES --
each worker keeps its GPU for every repeat it picks up. With <=1 GPU, no
pinning happens and all workers share the default device as before.

Examples
--------
Solo ESM, solo GIN, and their concatenation, on curated/group_molecule::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime curated_full --pairs data/processed/pairs_curated.csv \\
        --source prot=esm:data/embeddings/proteins/esm2_650m_mean.npz \\
        --source mol=gin:data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz \\
        --combos "1 2 12" --split group_molecule --seeds 42 43 44

Same combos on full_full, both split traditions, LoRaX's own ESM-1b::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime full_full --full-full-mode transductive \\
        --source prot=esm:data/embeddings/proteins/esm1b_650m_mean.npz:esm1b_t33_650M_UR50S \\
        --source mol=gin:data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz \\
        --combos "1 2 12" --on-missing drop

Adding the cls (attention) source -- solo, paired with each of prot/mol, and
all three together -- on full_full/transductive, all 5 folds, 3 at a time::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime full_full --full-full-mode transductive --max-parallel 3 \\
        --source cls=attn_noisy_or \\
        --source prot=esm:data/embeddings/proteins/esm1b_650m_mean.npz:esm1b_t33_650M_UR50S \\
        --source mol=gin:data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz \\
        --combos "1 2 3 12 13 23 123" --on-missing drop

Same, but cls as a single ProSmith-style model (n_models=1) instead of the
default 5-model bagging ensemble -- note the blank protein/molecule-path
fields to keep their defaults::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime full_full --full-full-mode transductive --max-parallel 3 \\
        --source cls=attn_noisy_or:::1 \\
        --source prot=esm:data/embeddings/proteins/esm1b_650m_mean.npz:esm1b_t33_650M_UR50S \\
        --source mol=gin:data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz \\
        --combos "1 2 3 12 13 23 123" --on-missing drop
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind.ensemble import EsmExtractor, GinExtractor, run_ensemble
from orbind.regimes import full_full_pairs, load_split
from orbind.attention_extractor import MilNoisyOrExtractor, MilLseExtractor
from orbind.gnn_extractor import GnnSignedExtractor

# entity-level: "name=type:path[:model_name[:pooling]]" -- a static npz lookup.
TYPE_FACTORIES = {"esm": EsmExtractor, "gin": GinExtractor}
# pair-level, supervised: "name=type" (bakes in its own embedding paths/hparams;
# optionally "name=type:protein_path:molecule_sites_path" to override them).
ATTENTION_FACTORIES = {"attn_noisy_or": MilNoisyOrExtractor, "attn_lse": MilLseExtractor}
# pair-level, supervised, graph-based: "name=type" (bakes in mean-pooled ESM/GIN
# paths; "name=type:protein_path:molecule_path[:n_models]" to override).
GNN_FACTORIES = {"gnn_signed": GnnSignedExtractor}
DEFAULT_REPEATS = {"transductive": [1, 2, 3, 4, 5], "inductive_molecule": [42, 43, 44, 45, 46]}


def parse_source_arg(raw: str):
    """"name=type:path[:model_name[:pooling]]" (entity-level) or
    "name=type[:protein_path:molecule_sites_path[:n_models]]" (attention,
    pair-level) -> (name, extractor). n_models=1 is ProSmith's own scheme
    (one whole-train model embeds train/val/test); n_models>1 (default 5)
    bags that many independently-seeded models -- see attention_extractor's
    module docstring for why this replaced true OOF fold-holdout retraining."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--source {raw!r} must look like name=type[:...]")
    name, rest = raw.split("=", 1)
    parts = rest.split(":")
    type_ = parts[0]

    if type_ in ATTENTION_FACTORIES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["molecule_sites_path"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        return name, ATTENTION_FACTORIES[type_](name=name, **kwargs)

    if type_ in GNN_FACTORIES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["molecule_path"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        return name, GNN_FACTORIES[type_](name=name, **kwargs)

    if type_ not in TYPE_FACTORIES:
        raise argparse.ArgumentTypeError(
            f"unknown source type {type_!r}, have {list(TYPE_FACTORIES) + list(ATTENTION_FACTORIES) + list(GNN_FACTORIES)}")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--source {raw!r} must look like name=type:path")
    path = parts[1]
    kwargs = {}
    if len(parts) > 2 and parts[2]:
        kwargs["model_name"] = parts[2]
    if len(parts) > 3 and parts[3]:
        kwargs["pooling"] = parts[3]
    extractor = TYPE_FACTORIES[type_](name=name, path=path, **kwargs)
    return name, extractor


def _detect_gpus(explicit):
    """GPU ids to round-robin repeats across, or None (no pinning needed --
    either the user gave none/one explicitly, or there's <=1 visible GPU)."""
    if explicit is not None:
        return explicit if len(explicit) > 1 else None
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return list(range(n)) if n > 1 else None


def _pin_worker_gpu(gpu_ids, counter, lock):
    """ProcessPoolExecutor initializer: each worker process claims the next
    GPU id round-robin and restricts itself to it via CUDA_VISIBLE_DEVICES,
    set before any CUDA-touching code (torch/xgboost) runs in this process.
    Workers are long-lived, so a worker keeps the same GPU across every
    repeat it picks up -- no per-task reassignment needed."""
    with lock:
        idx = counter.value
        counter.value += 1
    gpu = gpu_ids[idx % len(gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


class _Tee:
    """Writes to several streams at once -- lets a repeat's own log file
    capture its prints while they still show up live in the console."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _run_one_repeat(regime, pairs, extractors, combos, split, repeat, test_size, val_size,
                     weight_method, on_missing, full_full_mode, run_dir, save_checkpoints,
                     tune_boost_hp, n_trials, optuna_storage):
    """One repeat's full run_ensemble call -- top-level (not a closure) so it
    can be pickled and sent to a separate process by --max-parallel. Owns its
    own log file and checkpoint subdir regardless of which process runs it."""
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = None
    if save_checkpoints:
        checkpoint_dir = run_dir / "checkpoints" / f"repeat_{repeat}"

    with open(log_dir / f"repeat_{repeat}.log", "w", encoding="utf-8") as logf:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, logf)
        try:
            gpu_note = os.environ.get("CUDA_VISIBLE_DEVICES")
            tag = f" (pinned to GPU {gpu_note})" if gpu_note is not None else ""
            print(f"=== repeat {repeat} ==={tag}", flush=True)
            if regime == "curated_full":
                result = run_ensemble(pairs, extractors, combos, split_kind=split, seed=repeat,
                                       test_size=test_size, val_size=val_size,
                                       weight_method=weight_method, on_missing=on_missing,
                                       checkpoint_dir=checkpoint_dir,
                                       tune_boost_hp=tune_boost_hp, n_trials=n_trials,
                                       optuna_storage=optuna_storage, run_id=str(repeat))
            else:
                train_idx, val_idx, test_idx = load_split(full_full_mode, repeat)
                result = run_ensemble(pairs, extractors, combos,
                                       train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                                       weight_method=weight_method, on_missing=on_missing,
                                       checkpoint_dir=checkpoint_dir,
                                       tune_boost_hp=tune_boost_hp, n_trials=n_trials,
                                       optuna_storage=optuna_storage, run_id=str(repeat))
        finally:
            sys.stdout = old_stdout
    return repeat, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="curated_full", choices=["curated_full", "full_full"])
    ap.add_argument("--source", action="append", required=True, dest="sources",
                     help="name=type:path[:model_name[:pooling]], repeatable; order fixes combo digits")
    ap.add_argument("--combos", required=True, help='e.g. "1 2 12"')
    ap.add_argument("--weight-method", default="both", choices=["both", "simplex", "logreg"])
    ap.add_argument("--on-missing", default="raise", choices=["raise", "drop"],
                     help="raise: fail loudly on any embedding gap (default). "
                          "drop: warn with coverage %% and drop uncovered rows, "
                          "same reduced set for every combo + the ensemble.")
    ap.add_argument("--tune-boost", action="store_true",
                     help="tune each combo's XGBoost head independently via optuna "
                          "(see orbind/baselines.py::tune_boost) instead of fixed "
                          "hyperparameters. Off by default -- multiplies runtime by "
                          "roughly --n-trials per combo per repeat.")
    ap.add_argument("--n-trials", type=int, default=30,
                     help="optuna trials per combo when --tune-boost is set (default 30)")
    ap.add_argument("--out-dir", default="results/ensemble_logs",
                     help="base dir; each run gets its own timestamped subfolder here")
    ap.add_argument("--run-name", default=None,
                     help="override the run folder name (default: <regime>-<split_or_mode>-<timestamp>)")
    ap.add_argument("--skip-checkpoints", action="store_true",
                     help="don't save boosters/torch weights, only metrics (smaller, faster to iterate)")
    ap.add_argument("--max-parallel", type=int, default=1,
                     help="run this many repeats concurrently, each in its own OS process "
                          "(separate CUDA context). Default 1 (sequential). "
                          "Independent of OOF parallelism, which is always on inside each repeat.")
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                     help="GPU ids to round-robin repeats across when --max-parallel > 1, "
                          "e.g. --gpus 0 1. Default: auto-detect all visible GPUs; "
                          "with 0 or 1 visible, no pinning (all workers share the default device).")

    g1 = ap.add_argument_group("curated_full")
    g1.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    g1.add_argument("--split", default="stratified", choices=["stratified", "group_molecule", "group_receptor"])
    g1.add_argument("--seeds", type=int, nargs="+", default=[42])
    g1.add_argument("--test-size", type=float, default=0.2)
    g1.add_argument("--val-size", type=float, default=0.2)

    g2 = ap.add_argument_group("full_full")
    g2.add_argument("--full-full-mode", default="transductive", choices=["transductive", "inductive_molecule"])
    g2.add_argument("--repeats", type=int, nargs="+", default=None,
                     help="fold 1-5 for transductive, cold-molecule seed for inductive_molecule. "
                          "Default: 1..5 / 42..46 respectively.")
    args = ap.parse_args()

    tag = args.split if args.regime == "curated_full" else args.full_full_mode
    run_id = args.run_name or f"{args.regime}-{tag}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = _root / args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run: {run_dir}")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    log_path = run_dir / "log.txt"
    with open(log_path, "w", encoding="utf-8") as logf:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, logf)
        try:
            _run(args, run_dir)
        finally:
            sys.stdout = old_stdout


def _run(args, run_dir) -> None:
    extractors = dict(parse_source_arg(raw) for raw in args.sources)
    for name, ex in extractors.items():
        dim_out = getattr(ex, "dim_out", getattr(ex, "dim", "?"))
        print(f"source {name!r}: type={type(ex).__name__} model={ex.model_name} "
              f"pooling={ex.pooling} dim_out={dim_out} path={ex.path}")

    if args.regime == "curated_full":
        pairs_path = pathlib.Path(args.pairs)
        if not pairs_path.is_absolute():
            pairs_path = _root / pairs_path
        pairs = pd.read_csv(pairs_path)
        print(f"pairs: {len(pairs)} rows from {pairs_path}")
        repeats = args.seeds
    else:
        pairs = full_full_pairs()
        print(f"pairs: {len(pairs)} rows from orbind.regimes.full_full_pairs()")
        repeats = args.repeats or DEFAULT_REPEATS[args.full_full_mode]

    save_checkpoints = not args.skip_checkpoints

    optuna_storage = None
    if args.tune_boost:
        db_path = (run_dir / "optuna_studies.db").as_posix()
        optuna_storage = f"sqlite:///{db_path}"
        # Create the schema (tables) right now, before any repeat starts --
        # otherwise it's only lazily created on the first real tune_boost
        # call (which may be minutes away, behind the cls model's own
        # training), and `optuna-dashboard` opened before that point fails
        # with "no such table: version_info" on the still-schemaless file.
        import optuna as _optuna
        _optuna.create_study(storage=optuna_storage, study_name="_init", load_if_exists=True)
        print(f"\noptuna storage: {optuna_storage}")
        print(f"  live dashboard: optuna-dashboard {optuna_storage!r}")

    def collect(repeat, result):
        for combo, m in result["combos"].items():
            rows.append({"repeat": repeat, "kind": "combo", "name": "+".join(combo), **m})
        for method, m in result["ensemble"].items():
            w = result["weights"][method]
            rows.append({
                "repeat": repeat, "kind": "ensemble", "name": f"ensemble[{method}]", **m,
                "weights": json.dumps({"+".join(c): round(v, 4) for c, v in w.items()}),
            })

    rows = []
    if args.max_parallel <= 1:
        for repeat in repeats:
            print(f"\n--- repeat {repeat} (see logs/repeat_{repeat}.log) ---", flush=True)
            _, result = _run_one_repeat(args.regime, pairs, extractors, args.combos, args.split, repeat,
                                         args.test_size, args.val_size, args.weight_method,
                                         args.on_missing, args.full_full_mode, run_dir, save_checkpoints,
                                         args.tune_boost, args.n_trials, optuna_storage)
            collect(repeat, result)
    else:
        gpu_ids = _detect_gpus(args.gpus)
        # "spawn", not the platform default -- on Linux that default is "fork", and
        # _detect_gpus above already touched torch.cuda in this (parent) process to
        # count devices; forking a child that inherits an initialized CUDA context
        # is unsupported and fails with "CUDA error: initialization error". Windows
        # already defaults to spawn, which is why this only surfaces on Linux.
        ctx = multiprocessing.get_context("spawn")
        pool_kwargs = {"max_workers": args.max_parallel, "mp_context": ctx}
        if gpu_ids:
            print(f"\nrunning {len(repeats)} repeats, up to {args.max_parallel} concurrently "
                  f"(separate processes; each writes logs/repeat_{{R}}.log; "
                  f"pinned round-robin across GPUs {gpu_ids})...", flush=True)
            manager = ctx.Manager()
            pool_kwargs["initializer"] = _pin_worker_gpu
            pool_kwargs["initargs"] = (gpu_ids, manager.Value("i", 0), manager.Lock())
        else:
            print(f"\nrunning {len(repeats)} repeats, up to {args.max_parallel} concurrently "
                  f"(separate processes; each writes logs/repeat_{{R}}.log)...", flush=True)
        with ProcessPoolExecutor(**pool_kwargs) as pool:
            futures = {
                pool.submit(_run_one_repeat, args.regime, pairs, extractors, args.combos, args.split, repeat,
                            args.test_size, args.val_size, args.weight_method, args.on_missing,
                            args.full_full_mode, run_dir, save_checkpoints,
                            args.tune_boost, args.n_trials, optuna_storage): repeat
                for repeat in repeats
            }
            for f in as_completed(futures):
                repeat = futures[f]
                _, result = f.result()
                print(f"--- repeat {repeat} done (see logs/repeat_{repeat}.log) ---", flush=True)
                collect(repeat, result)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    print(f"\nwrote -> {run_dir / 'metrics.csv'} ({len(df)} rows)")
    print(df.groupby(["kind", "name"])[["AUROC", "AUPRC", "MCC", "F1"]].mean().round(3))
    if save_checkpoints:
        print(f"checkpoints -> {run_dir / 'checkpoints'}/repeat_*/")


if __name__ == "__main__":
    main()
