"""Rebuild the command line of every run that did not finish, from its own config.json.

Why this exists: when a block of baselines dies partway (2026-09-20: xgboost 3.x
aborting in a CUDA destructor, see orbind/docs/gotchas.md), the question "what
exactly do I rerun?" is answered badly from shell history and well from disk. Each
run wrote `config.json` at start-up, so the arguments are already recorded -- this
just turns them back into a command.

It reconstructs against `train_ensemble_boost.build_parser()`, so a value equal to
the parser's default is omitted and every other one is spelled with the flag that
set it. Keys that are not CLI arguments (`python`, `xgboost`) are read, not
re-emitted: they say which environment produced the run, and the rebuilt command
reuses that same interpreter unless you override it.

Nothing is executed. It prints commands; you read them and run them.

    python scripts/modeling/train/relaunch_incomplete.py \
        --root results/ensemble_logs_esm3 --repeats 5

    # only the insect pools, all on one interpreter:
    python scripts/modeling/train/relaunch_incomplete.py \
        --root results/ensemble_logs_esm3 --glob '{cc,hc}-*' \
        --python .venv-controls/bin/python
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib
import shlex
import sys

import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_teb", _root / "scripts" / "modeling" / "train" / "train_ensemble_boost.py")
_teb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_teb)

#: Recorded by the trainer for provenance, not accepted by it as flags.
NON_CLI_KEYS = {"python", "xgboost"}

#: Which env each pair-level source needs, for runs written BEFORE config.json
#: started recording its interpreter. Guessing wrong here does not always crash --
#: it can load a checkpoint and "train" a fold in 5 s (see README) -- so the guess
#: is made explicitly and printed, never left to whatever shell happens to be open.
ENV_FOR_SOURCE = {
    "molor": ".venv-molor/bin/python",      # dgl 2.4 + dgllife
    "hladis": ".venv/bin/python",           # needs rdkit, which controls lacks
    "gnn_signed": ".venv/bin/python",       # PyG
    "lorax": ".venv-controls/bin/python",
    "prosmith": ".venv-controls/bin/python",
}


def infer_python(config: dict) -> str | None:
    """The env a run needs, read off its cls source. None when nothing matches."""
    for spec in config.get("sources") or []:
        _, _, rhs = str(spec).partition("=")
        kind = rhs.split(":")[0]
        if kind in ENV_FOR_SOURCE:
            return ENV_FOR_SOURCE[kind]
    return None


def done_repeats(run_dir: pathlib.Path) -> set:
    """Which repeats actually produced rows. Empty when the run wrote nothing.

    Reads metrics.csv rather than counting checkpoints: a checkpoint proves an
    extractor trained, which is exactly the half of the run that did NOT fail.
    """
    path = run_dir / "metrics.csv"
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    return set(df["repeat"].unique()) if "repeat" in df.columns else set()


def flags_for(dest: str, parser: argparse.ArgumentParser) -> str | None:
    """The long option that sets `dest`, or None when it is positional/unknown."""
    for action in parser._actions:
        if action.dest == dest and action.option_strings:
            return max(action.option_strings, key=len)
    return None


def is_store_true(dest: str, parser: argparse.ArgumentParser) -> bool:
    for action in parser._actions:
        if action.dest == dest:
            return isinstance(action, argparse._StoreTrueAction)
    return False


def canonical_repeats(config: dict, n: int = 5) -> list[int]:
    """The full fold set a run of this kind is supposed to have.

    NOT the one config.json holds. config.json records the LAST invocation, so a
    run that was once topped up fold-by-fold has a narrowed `repeats` list, and
    copying it would quietly rebuild a three-fold run out of a five-fold one --
    which is how a paper row loses two folds without anything looking wrong.
    The trainer's own convention: 42..46 for the cold-molecule modes, 1..n
    everywhere else.
    """
    if config.get("regime") == "full_full" and \
            str(config.get("full_full_mode", "")).startswith("inductive"):
        return list(range(42, 42 + n))
    return list(range(1, n + 1))


def rebuild(config: dict, parser: argparse.ArgumentParser, python: str,
            script: str, repeats: list[int] | None = None) -> list[str]:
    """config.json -> argv. Defaults are dropped so the command stays readable.

    `repeats` overrides whatever the config recorded; pass None to keep it.
    """
    config = dict(config)
    if repeats is not None:
        config["repeats"] = repeats
    argv = [python, script]
    for dest, value in config.items():
        if dest in NON_CLI_KEYS or value is None:
            continue
        flag = flags_for(dest, parser)
        if flag is None:
            continue
        if value == parser.get_default(dest):
            continue
        if is_store_true(dest, parser):
            if value:
                argv.append(flag)
            continue
        if dest == "sources":          # repeatable: one flag per value
            for v in value:
                argv += [flag, str(v)]
        elif isinstance(value, list):  # nargs="+": one flag, many values
            argv += [flag, *[str(v) for v in value]]
        else:
            argv += [flag, str(value)]
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="an ensemble_logs root; runs are found at <root>/<pool>/<run>")
    ap.add_argument("--repeats", type=int, default=5,
                    help="how many repeats a complete run has (default 5)")
    ap.add_argument("--glob", default="*", help="shell pattern over pool names")
    ap.add_argument("--python", default=None,
                    help="interpreter for the rebuilt commands. Default: the one "
                         "config.json recorded; for older runs, the env the cls "
                         "source needs (ENV_FOR_SOURCE); failing both, this one.")
    ap.add_argument("--keep-recorded-repeats", action="store_true",
                    help="use the fold list config.json holds instead of the full "
                         "canonical set. config.json records the LAST invocation, so "
                         "a run topped up fold-by-fold has a narrowed list -- the "
                         "default rebuilds all of them.")
    ap.add_argument("--show-complete", action="store_true",
                    help="also list the runs that need nothing")
    args = ap.parse_args(argv)

    parser = _teb.build_parser()
    script = "scripts/modeling/train/train_ensemble_boost.py"
    root = pathlib.Path(args.root)
    if not root.is_absolute():
        root = _root / root

    incomplete = []
    for config_path in sorted(root.glob("*/*/config.json")):
        run_dir = config_path.parent
        if not fnmatch.fnmatch(run_dir.parent.name, args.glob):
            continue
        done = done_repeats(run_dir)
        label = f"{run_dir.parent.name}/{run_dir.name}"
        if len(done) >= args.repeats:
            if args.show_complete:
                print(f"# complete  {label}  ({len(done)} repeats)")
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        python = (args.python or config.get("python") or infer_python(config)
                  or sys.executable)
        want = None if args.keep_recorded_repeats else canonical_repeats(config, args.repeats)
        note = ""
        if want is not None and sorted(config.get("repeats") or want) != sorted(want):
            note = (f"  [config records repeats {config.get('repeats')}; rebuilding with "
                    f"the full set {want}]")
        incomplete.append((label, len(done),
                           rebuild(config, parser, python, script, repeats=want), note))

    if not incomplete:
        print("# nothing to relaunch")
        return 0

    print(f"# {len(incomplete)} incomplete run(s) under {root}")
    print("# checkpoints are reused, so a relaunch redoes only what is missing.\n")
    for label, n_done, cmd, note in incomplete:
        print(f"# {label}  ({n_done}/{args.repeats} repeats present){note}")
        print(" ".join(shlex.quote(c) for c in cmd))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
