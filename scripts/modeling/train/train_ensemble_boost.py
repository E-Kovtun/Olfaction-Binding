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
                   --full-full-mode
                   {transductive,inductive_molecule,inductive_molecule_v5} +
                   --repeats (LoRaX fold 1-5 for transductive, cold-molecule
                   seed for the two inductive modes). `inductive_molecule` is
                   our own cold-molecule split (30% holdout, stratified);
                   `inductive_molecule_v5` reproduces the v5 graph screen's
                   own split exactly (20% test / 10% val molecules,
                   unstratified) so ensemble numbers can be compared to that
                   screen head-on -- see orbind/regimes.py.

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

The ProSmith/MPP baseline itself, reproducing upstream's own second stage
(its `training_GB.py` builds exactly `cls`, `prot+mol` and `prot+mol+cls`).
The cls source takes a *per-residue* protein npz -- upstream's own ESM-1b,
imported by scripts/embedding_generation/proteins/06_import_ofm_esm1b.py --
while prot stays mean-pooled; both defaults, so only the BindingDB
checkpoint (which upstream's published numbers use) has to be named::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime full_full --full-full-mode transductive \\
        --source cls=prosmith:::1:saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl \\
        --source prot=esm:data/embeddings/proteins/esm1b_650m_mean.npz:esm1b_t33_650M_UR50S \\
        --source mol=gin:data/embeddings/molecules/chemberta_77m_m2or.npz:chemberta_77m \\
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
import shutil
import sys
from datetime import datetime

import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import importlib

from orbind.ensemble import run_ensemble
from orbind.regimes import full_full_pairs, load_split

# Source types dispatch LAZILY: parse_source_arg imports an extractor's module
# only when a source of that type is actually requested, so a run pulls just the
# heavy deps it needs -- gnn_* -> torch_geometric, lorax -> transformers+peft,
# prosmith/attn/esm/gin -> torch only. This is what lets the ProSmith/LORAX
# "controls" run in a PyG-free environment separate from the graph pipeline.
_FACTORY_SPEC = {
    # entity-level: "name=type:path[:model_name[:pooling]]" -- a static npz lookup.
    "esm": ("orbind.ensemble", "EsmExtractor"),
    "gin": ("orbind.ensemble", "GinExtractor"),
    # attention MIL (torch only): "name=type[:protein_path:molecule_sites_path[:n_models]]".
    "attn_noisy_or": ("orbind.attention_extractor", "MilNoisyOrExtractor"),
    "attn_lse": ("orbind.attention_extractor", "MilLseExtractor"),
    # graph-based (pulls torch_geometric): "name=type[:protein_path:molecule_path[:n_models[:emit]]]".
    # emit: "prot" (default, v5 probe shape) or "both". gnn_signed_dgi adds a
    # DeepGraphInfomax auxiliary loss (shared scope, lambda=0.5).
    "gnn_signed": ("orbind.gnn_extractor", "GnnSignedExtractor"),
    "gnn_signed_dgi": ("orbind.gnn_extractor", "GnnSignedDgiExtractor"),
    # ProSmith/MPP transformer over a *per-residue* protein npz + pooled molecule npz
    # (torch only): "name=prosmith[:protein_path:molecule_path[:n_models[:pretrained_path[:faithful_bugs]]]]".
    # pretrained_path = upstream BindingDB checkpoint; empty trains from scratch
    # (weaker than upstream's published numbers). faithful_bugs=1 restores the
    # double sigmoid + unmasked padding (slower). See orbind/prosmith_extractor.py.
    "prosmith": ("orbind.prosmith_extractor", "ProSmithExtractor"),
    # LORAX (pulls transformers+peft): LoRA-ChemBERTa molecule + cross-attention
    # over frozen per-residue ESM-1b protein. LoRA on the MOLECULE side only and
    # protein frozen ESM-1b, so directly comparable to the ProSmith baseline (both
    # on ESM-1b). "name=lorax[:protein_path:chemberta_card[:n_models[:lora_r[:epochs]]]]"
    # (epochs exposed so a smoke can pass e.g. cls=lorax:::::2).
    # See orbind/lorax_extractor.py.
    "lorax": ("orbind.lorax_extractor", "LoraxExtractor"),
    # MolOR (pulls dgl+dgllife+rdkit): GCN molecule encoder (trained live) cross-
    # attended with frozen per-residue ESM-1b protein -- same ESM-1b as ProSmith
    # and LORAX, so all three are directly comparable. Molecule is a 2D graph, not
    # ChemBERTa (that IS MolOR's identity). "name=molor[:protein_path[:n_models[:epochs]]]".
    # See orbind/molor_extractor.py.
    "molor": ("orbind.molor_extractor", "MolorExtractor"),
    # Hladis et al. ICLR 2023 (pulls rdkit): MPNN-attention over the molecule
    # graph, with the receptor vector broadcast onto every atom -- no cross-
    # attention block at all. Protein defaults to mean-pooled ESM-1b (upstream
    # uses ProtBERT CLS; ESM-1b keeps it comparable to the other three).
    # Emitted feature is only node_d_model=72 wide by default -- upstream's own
    # size, far narrower than prosmith/lorax/molor.
    # "name=hladis[:protein_path[:n_models[:epochs]]]". See orbind/hladis_extractor.py.
    "hladis": ("orbind.hladis_extractor", "HladisExtractor"),
}
_ENTITY_TYPES    = {"esm", "gin"}
_ATTENTION_TYPES = {"attn_noisy_or", "attn_lse"}
_GNN_TYPES       = {"gnn_signed", "gnn_signed_dgi"}
_PROSMITH_TYPES  = {"prosmith"}
_LORAX_TYPES     = {"lorax"}
# Same CLI shape for both: "name=type[:protein_path[:n_models[:epochs]]]" --
# one protein npz, no molecule npz (each builds its own molecule graphs).
_MOLOR_TYPES     = {"molor", "hladis"}


def _factory(type_):
    """Import the extractor module lazily and return its class."""
    module, cls = _FACTORY_SPEC[type_]
    return getattr(importlib.import_module(module), cls)
DEFAULT_REPEATS = {"transductive": [1, 2, 3, 4, 5],
                    "inductive_molecule": [42, 43, 44, 45, 46],
                    "inductive_molecule_v5": [42, 43, 44, 45, 46]}


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

    if type_ in _ATTENTION_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["molecule_sites_path"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _GNN_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["molecule_path"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        if len(parts) > 4 and parts[4]:
            kwargs["emit"] = parts[4]
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _PROSMITH_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["molecule_path"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        if len(parts) > 4 and parts[4]:
            kwargs["pretrained_path"] = parts[4]
        if len(parts) > 5 and parts[5]:
            kwargs["faithful_bugs"] = parts[5] not in ("0", "false", "False")
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _LORAX_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["chemberta_card"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        if len(parts) > 4 and parts[4]:
            kwargs["lora_r"] = int(parts[4])
        if len(parts) > 5 and parts[5]:
            kwargs["epochs"] = int(parts[5])
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _MOLOR_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["n_models"] = int(parts[2])
        if len(parts) > 3 and parts[3]:
            kwargs["epochs"] = int(parts[3])
        return name, _factory(type_)(name=name, **kwargs)

    if type_ not in _ENTITY_TYPES:
        raise argparse.ArgumentTypeError(
            f"unknown source type {type_!r}, have {list(_FACTORY_SPEC)}")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--source {raw!r} must look like name=type:path")
    path = parts[1]
    kwargs = {}
    if len(parts) > 2 and parts[2]:
        kwargs["model_name"] = parts[2]
    if len(parts) > 3 and parts[3]:
        kwargs["pooling"] = parts[3]
    extractor = _factory(type_)(name=name, path=path, **kwargs)
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

    def isatty(self):
        # Never a TTY: keeps libraries (e.g. transformers' loading report) from
        # emitting ANSI colour codes into the captured log file, and avoids the
        # AttributeError they raise when they probe sys.stdout.isatty().
        return False

    def __getattr__(self, name):
        # Delegate anything else a library might probe (fileno, encoding, ...) to
        # the real console stream.
        return getattr(self.streams[0], name)


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


def _run_one_repeat_to_queue(q, *call_args):
    """Wraps _run_one_repeat for a plain multiprocessing.Process worker (see
    the GPU-pinning note in _run() for why this replaced ProcessPoolExecutor's
    initializer-based pinning): puts (repeat, result) on `q` instead of
    returning it, since a bare Process has no return-value channel."""
    q.put(_run_one_repeat(*call_args))


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
    g2.add_argument("--full-full-mode", default="transductive",
                     choices=["transductive", "inductive_molecule", "inductive_molecule_v5"])
    g2.add_argument("--repeats", type=int, nargs="+", default=None,
                     help="fold 1-5 for transductive, cold-molecule seed for the inductive modes. "
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

    # metrics.csv is written incrementally, one repeat at a time, so a run
    # killed partway through (e.g. server preemption) doesn't lose already-
    # finished repeats -- rerunning the same command later only has to redo
    # (cheaply, thanks to checkpoint/optuna reuse -- see attention_extractor's
    # and gnn_extractor's checkpoint_path handling and baselines.tune_boost's
    # trial-count resume) whatever this repeat's combos hadn't finished yet.
    # A repeat's rows fully replace any earlier rows for that same repeat
    # (this run's --combos is authoritative for what "done" means now); any
    # existing metrics.csv is preserved first as metrics.csv.bak.
    metrics_path = run_dir / "metrics.csv"
    all_rows = []
    if metrics_path.exists():
        shutil.copy(metrics_path, run_dir / "metrics.csv.bak")
        all_rows = pd.read_csv(metrics_path).to_dict("records")

    def collect(repeat, result):
        rows = []
        for combo, m in result["combos"].items():
            rows.append({"repeat": repeat, "kind": "combo", "name": "+".join(combo), **m})
        for method, m in result["ensemble"].items():
            w = result["weights"][method]
            rows.append({
                "repeat": repeat, "kind": "ensemble", "name": f"ensemble[{method}]", **m,
                "weights": json.dumps({"+".join(c): round(v, 4) for c, v in w.items()}),
            })
        all_rows[:] = [r for r in all_rows if r["repeat"] != repeat] + rows
        pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
        print(f"  wrote -> {metrics_path} ({len(all_rows)} rows so far)", flush=True)

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

        # Deliberately NOT ProcessPoolExecutor(initializer=...): that initializer
        # only runs *after* the child has already imported this module to be able
        # to unpickle the initializer/task callables in the first place, and that
        # import chain (torch_geometric -> torch_scatter/torch_sparse) can touch
        # CUDA and bind the process to the default device before the initializer
        # ever gets a chance to restrict CUDA_VISIBLE_DEVICES -- confirmed on the
        # real server: the pinning log line printed the right GPU per repeat, but
        # nvidia-smi showed every worker on GPU0 regardless. Setting the env var
        # in the *parent*, immediately before each ctx.Process(...).start(), bakes
        # it into that child's environment before its interpreter (and therefore
        # any import) even begins, which does work.
        if gpu_ids:
            print(f"\nrunning {len(repeats)} repeats, up to {args.max_parallel} concurrently "
                  f"(separate processes; each writes logs/repeat_{{R}}.log; "
                  f"pinned round-robin across GPUs {gpu_ids} via CUDA_VISIBLE_DEVICES set "
                  f"before each process starts)...", flush=True)
        else:
            print(f"\nrunning {len(repeats)} repeats, up to {args.max_parallel} concurrently "
                  f"(separate processes; each writes logs/repeat_{{R}}.log)...", flush=True)

        for start in range(0, len(repeats), args.max_parallel):
            chunk = list(enumerate(repeats))[start:start + args.max_parallel]
            procs = []
            for i, repeat in chunk:
                gpu = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
                if gpu is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                q = ctx.Queue()
                p = ctx.Process(target=_run_one_repeat_to_queue, args=(q, args.regime, pairs, extractors,
                                                                        args.combos, args.split, repeat,
                                                                        args.test_size, args.val_size,
                                                                        args.weight_method, args.on_missing,
                                                                        args.full_full_mode, run_dir,
                                                                        save_checkpoints, args.tune_boost,
                                                                        args.n_trials, optuna_storage))
                p.start()
                procs.append((repeat, p, q))
            for repeat, p, q in procs:
                r_repeat, result = q.get()
                p.join()
                print(f"--- repeat {r_repeat} done (see logs/repeat_{r_repeat}.log) ---", flush=True)
                collect(r_repeat, result)

    df = pd.DataFrame(all_rows)
    print(f"\nfinal -> {metrics_path} ({len(df)} rows)")
    print(df.groupby(["kind", "name"])[["AUROC", "AUPRC", "MCC", "F1"]].mean().round(3))
    if save_checkpoints:
        print(f"checkpoints -> {run_dir / 'checkpoints'}/repeat_*/")


if __name__ == "__main__":
    main()
