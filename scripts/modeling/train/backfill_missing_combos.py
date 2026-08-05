"""Backfill combos a finished ensemble run never computed, without re-tuning.

Some runs were launched with only the cls-containing combos (e.g. "1 12 13
123"), on the reasoning that the label-independent ones -- solo `prot`, solo
`mol`, and especially `prot+mol`, the raw-boost reference every other number
is judged against -- had already been tuned to death in a sibling run over
the exact same rows. This script imports those combos instead of paying for
them again, then relaunches the run so its ensemble is refit over the full
combo set.

What actually gets imported is the *optuna study*, not the booster: the
ensemble engine never loads a saved booster, it always refits, so copying
`boost_{combo}.json` would achieve nothing. Copying study
`repeat{R}_combo{name}` into the target's `optuna_studies.db` makes
`baselines.tune_boost` see a completed trial budget, add zero trials, and go
straight to the one final refit -- on the *target* run's own train rows,
scoring its own val/test. So no test data crosses runs; the only thing
borrowed is the hyperparameter choice.

That borrowing is only sound when the two runs' features for the imported
combo are literally the same array, which needs all of:
  * same regime and split mode (hence the same train/val/test positions);
  * byte-identical `--source` strings for every source the combo uses;
  * the same surviving row set after `--on-missing drop` -- the coverage
    intersection spans *every* source in a run, cls included, so a sibling
    whose cls reads a different molecule file can silently retain a
    different number of rows.
The third is the one that bites, so it is checked against what each run's
own logs recorded rather than assumed.

Usage
-----
uv run python scripts/modeling/train/backfill_missing_combos.py            # plan only
uv run python scripts/modeling/train/backfill_missing_combos.py --apply    # copy + relaunch
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import re
import shlex
import subprocess
import sys
import time

import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

TRAINER = "scripts/modeling/train/train_ensemble_boost.py"
RECENT_WRITE_MINUTES = 20


def load_runs(base: pathlib.Path) -> dict[str, dict]:
    """One record per run folder: its config, the combos its metrics.csv
    actually holds, the coverage line each repeat logged, and how long ago
    anything in it was written (to leave a live run alone)."""
    runs = {}
    for d in sorted(base.iterdir()):
        cfg_path = d / "config.json"
        if not d.is_dir() or not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        combos_done = set()
        met = d / "metrics.csv"
        if met.exists():
            m = pd.read_csv(met)
            combos_done = set(m.loc[m["kind"] == "combo", "name"].unique())

        coverage = {}
        for log in sorted((d / "logs").glob("repeat_*.log")) if (d / "logs").exists() else []:
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if "after coverage" in line:
                    coverage[log.stem] = re.sub(r"\s+", " ", line.strip())

        files = [f for f in d.rglob("*") if f.is_file()]
        idle_min = (time.time() - max((f.stat().st_mtime for f in files), default=0)) / 60

        runs[d.name] = {"dir": d, "cfg": cfg, "combos_done": combos_done,
                        "coverage": coverage, "idle_min": idle_min,
                        "sources": {s.split("=", 1)[0]: s for s in cfg.get("sources", [])}}
    return runs


def full_combo_set(source_names: list[str]) -> list[tuple[str, ...]]:
    """Every non-empty subset, in the engine's own digit order (shorter
    first, then by source position) -- the same order parse_combo_spec
    produces for "1 2 3 12 13 23 123"."""
    out = []
    for size in range(1, len(source_names) + 1):
        out.extend(itertools.combinations(source_names, size))
    return out


def combo_digits(combo: tuple[str, ...], source_names: list[str]) -> str:
    return "".join(str(source_names.index(n) + 1) for n in combo)


def compatible(target: dict, donor: dict, combo: tuple[str, ...]) -> str | None:
    """None if `donor` may supply `combo` to `target`, else why not."""
    tc, dc = target["cfg"], donor["cfg"]
    if tc.get("regime") != dc.get("regime"):
        return f"regime {tc.get('regime')} != {dc.get('regime')}"
    if tc.get("regime") == "full_full":
        if tc.get("full_full_mode") != dc.get("full_full_mode"):
            return f"mode {tc.get('full_full_mode')} != {dc.get('full_full_mode')}"
    elif tc.get("split") != dc.get("split"):
        return f"split {tc.get('split')} != {dc.get('split')}"
    for name in combo:
        if target["sources"].get(name) != donor["sources"].get(name):
            return f"--source {name} differs"
    if not target["coverage"] or not donor["coverage"]:
        return "no coverage lines logged, cannot verify row alignment"
    if target["coverage"] != donor["coverage"]:
        return "retained rows differ after --on-missing drop"
    return None


def rebuild_cli(cfg: dict, combos: str) -> list[str]:
    """The original invocation, with --combos swapped for the full set."""
    cmd = [sys.executable, TRAINER, "--regime", cfg["regime"]]
    for s in cfg["sources"]:
        cmd += ["--source", s]
    cmd += ["--combos", combos,
            "--on-missing", cfg.get("on_missing", "raise"),
            "--weight-method", cfg.get("weight_method", "both"),
            "--out-dir", cfg.get("out_dir", "results/ensemble_logs"),
            "--run-name", cfg["run_name"],
            "--max-parallel", str(cfg.get("max_parallel", 1))]
    if cfg.get("tune_boost"):
        cmd += ["--tune-boost", "--n-trials", str(cfg.get("n_trials", 30))]
    if cfg.get("skip_checkpoints"):
        cmd += ["--skip-checkpoints"]
    if cfg.get("gpus"):
        cmd += ["--gpus", *[str(g) for g in cfg["gpus"]]]
    if cfg["regime"] == "full_full":
        cmd += ["--full-full-mode", cfg.get("full_full_mode", "transductive")]
        if cfg.get("repeats"):
            cmd += ["--repeats", *[str(r) for r in cfg["repeats"]]]
    else:
        cmd += ["--pairs", cfg.get("pairs", ""), "--split", cfg.get("split", "stratified"),
                "--test-size", str(cfg.get("test_size", 0.2)),
                "--val-size", str(cfg.get("val_size", 0.2))]
        if cfg.get("seeds"):
            cmd += ["--seeds", *[str(s) for s in cfg["seeds"]]]
    return cmd


def copy_studies(target: dict, donor: dict, combo: tuple[str, ...], apply: bool) -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    to_db = f"sqlite:///{(target['dir'] / 'optuna_studies.db').as_posix()}"
    from_db = f"sqlite:///{(donor['dir'] / 'optuna_studies.db').as_posix()}"
    if not (donor["dir"] / "optuna_studies.db").exists():
        print(f"      donor {donor['dir'].name} has no optuna_studies.db -- "
              f"combo will be tuned from scratch")
        return 0

    suffix = "_combo" + "+".join(combo)
    have = set(optuna.get_all_study_names(storage=to_db)) if (target["dir"] / "optuna_studies.db").exists() else set()
    copied = 0
    for s in optuna.get_all_study_names(storage=from_db):
        if not s.endswith(suffix) or s in have:
            continue
        if apply:
            optuna.copy_study(from_study_name=s, from_storage=from_db,
                               to_storage=to_db, to_study_name=s)
        copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/ensemble_logs")
    ap.add_argument("--apply", action="store_true",
                     help="actually copy the studies and relaunch; without it, only print the plan")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these run names")
    args = ap.parse_args()

    base = _root / args.out_dir
    runs = load_runs(base)
    if not runs:
        print(f"no runs under {base}")
        return

    for name, run in runs.items():
        if args.only and name not in args.only:
            continue
        source_names = list(run["sources"])
        wanted = full_combo_set(source_names)
        missing = [c for c in wanted if "+".join(c) not in run["combos_done"]]
        if not missing:
            continue

        print(f"\n=== {name} ===")
        if run["idle_min"] < RECENT_WRITE_MINUTES:
            print(f"  SKIP: written {run['idle_min']:.0f} min ago -- looks like it is still running")
            continue
        if not run["combos_done"]:
            print("  SKIP: no metrics.csv -- nothing computed yet, backfill is meaningless")
            continue
        print(f"  missing: {', '.join('+'.join(c) for c in missing)}")

        total_copied, from_scratch = 0, []
        for combo in missing:
            donors = []
            for dname, donor in runs.items():
                if dname == name or "+".join(combo) not in donor["combos_done"]:
                    continue
                why = compatible(run, donor, combo)
                if why is None:
                    donors.append(dname)
                else:
                    print(f"    {'+'.join(combo):<14} donor {dname}: NO ({why})")
            if not donors:
                print(f"    {'+'.join(combo):<14} -> no compatible donor, will be tuned from scratch")
                from_scratch.append(combo)
                continue
            donor_name = donors[0]
            n = copy_studies(run, runs[donor_name], combo, args.apply)
            total_copied += n
            budget = runs[donor_name]["cfg"].get("n_trials")
            print(f"    {'+'.join(combo):<14} <- {donor_name}: {n} studies"
                  f"{'' if args.apply else ' (dry run)'}, donor budget {budget} trials")

        if from_scratch:
            n_rep = len(run["coverage"]) or 5
            trials = run["cfg"].get("n_trials", 0) if run["cfg"].get("tune_boost") else 0
            print(f"  COST: {len(from_scratch)} combo(s) without a donor -> "
                  f"~{len(from_scratch) * trials * n_rep} optuna trials "
                  f"({len(from_scratch)} x {trials} x {n_rep} repeats)")

        combos_arg = " ".join(combo_digits(c, source_names) for c in wanted)
        cmd = rebuild_cli(run["cfg"], combos_arg)
        print("  relaunch: " + " ".join(shlex.quote(a) for a in cmd[1:]))
        if args.apply:
            print(f"  ---- running ({total_copied} studies imported) ----", flush=True)
            rc = subprocess.run(cmd, cwd=_root).returncode
            print(f"  ---- exit {rc} ----")

    if not args.apply:
        print("\n(dry run -- rerun with --apply to copy the studies and relaunch)")


if __name__ == "__main__":
    main()
