"""Inventory of ensemble runs, and whether each one's ensembling is stale.

Read-only. Nothing here launches, copies or rewrites anything.

A run's `metrics.csv` records, for every repeat, the weight each combo got in
the fitted ensemble (the `weights` column on the `ensemble[...]` rows). That
dict is an exact record of which combos the ensemble was actually fit over.
Meanwhile the run folder accumulates per-combo artifacts: a booster
(`checkpoints/repeat_{R}/boost_{combo}.json`) for every combo that has been
fit, and an optuna study (`repeat{R}_combo{combo}`) for every combo whose
head has been tuned -- including combos imported from a sibling run after
the fact.

So the question "is this run's ensembling out of date?" is just: does any
repeat have artifacts for a combo the ensemble never saw? If yes, its
`ensemble[simplex]` / `ensemble[logreg]` rows were fit over a strictly
smaller candidate set than what is now available, and only the final
ensembling stage needs redoing -- the per-combo heads themselves are done.

Completeness is judged the same way, from artifacts rather than from
timestamps: a run is INCOMPLETE when some repeat is missing from metrics.csv
or lacks a booster for a combo the others have. A run being actively written
to is not something this script guesses at.

Usage
-----
uv run python scripts/modeling/eval/ensemble_run_status.py
uv run python scripts/modeling/eval/ensemble_run_status.py --verbose
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

ENTITY_TYPES = {"esm", "gin"}
ENCODER_TAGS = ("chemberta", "gin", "esm1b", "esm2")


def source_label(src: str) -> str:
    """"mol=gin:.../chemberta_77m_m2or.npz:chemberta_77m" -> "mol=chemberta";
    "cls=gnn_signed:<prot>:<mol>" -> "cls=gnn_signed(chemberta)". The CLI type
    of an entity source says nothing about the actual encoder (`gin` is just
    the inchikey-keyed npz lookup and is routinely handed ChemBERTa), and for
    a pair-level source the first path is the *protein* one -- both are easy
    to misread straight off the config."""
    name, _, rest = src.partition("=")
    parts = rest.split(":")
    type_ = parts[0]

    def tag(path: str) -> str:
        stem = pathlib.Path(path).stem.lower()
        return next((t for t in ENCODER_TAGS if t in stem), stem or "?")

    if type_ in ENTITY_TYPES:
        return f"{name}={tag(parts[1]) if len(parts) > 1 and parts[1] else type_}"
    if len(parts) > 2 and parts[2]:                 # pair-level: prot path, then mol path
        return f"{name}={type_}({tag(parts[2])})"
    return f"{name}={type_}"


def scan(d: pathlib.Path) -> dict | None:
    cfg_path = d / "config.json"
    if not d.is_dir() or not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # combos the ensemble was actually fit over, per repeat, straight from the
    # weight dicts; and the combos that have a scored row at all.
    ens_combos: dict[int, set[str]] = {}
    combo_rows: dict[int, set[str]] = {}
    met = d / "metrics.csv"
    if met.exists():
        m = pd.read_csv(met)
        for _, r in m.iterrows():
            rep = int(r["repeat"])
            if r["kind"] == "combo":
                combo_rows.setdefault(rep, set()).add(r["name"])
            elif isinstance(r.get("weights"), str):
                ens_combos.setdefault(rep, set()).update(json.loads(r["weights"]))

    boosters: dict[int, set[str]] = {}
    for rd in sorted((d / "checkpoints").glob("repeat_*")) if (d / "checkpoints").exists() else []:
        rep = int(rd.name.split("_")[1])
        boosters[rep] = {p.stem[len("boost_"):] for p in rd.glob("boost_*.json")}

    studies: dict[int, set[str]] = {}
    db = d / "optuna_studies.db"
    if db.exists():
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            for s in optuna.get_all_study_names(storage=f"sqlite:///{db.as_posix()}"):
                if "_combo" not in s:
                    continue
                head, _, combo = s.partition("_combo")
                if head.startswith("repeat") and head[len("repeat"):].isdigit():
                    studies.setdefault(int(head[len("repeat"):]), set()).add(combo)
        except Exception as exc:                     # a locked/partial db shouldn't kill the report
            studies = {}
            print(f"  ! could not read {db.name}: {exc}", file=sys.stderr)

    return {"dir": d, "cfg": cfg, "ens_combos": ens_combos, "combo_rows": combo_rows,
            "boosters": boosters, "studies": studies}


def verdict(run: dict) -> tuple[str, list[str]]:
    """(status, notes). OUTDATED wins over INCOMPLETE: it is the actionable one."""
    notes = []
    repeats = sorted(set(run["combo_rows"]) | set(run["boosters"]) | set(run["studies"]))
    if not repeats:
        return "EMPTY", ["no metrics.csv, no checkpoints, no studies"]

    stale = {}
    for rep in repeats:
        available = run["boosters"].get(rep, set()) | run["studies"].get(rep, set())
        used = run["ens_combos"].get(rep, set())
        extra = available - used
        if extra and used:
            stale[rep] = extra
    if stale:
        every = sorted(set().union(*stale.values()))
        notes.append(f"ensemble fit without: {', '.join(every)} "
                     f"(repeats {', '.join(str(r) for r in sorted(stale))})")

    scored = set(run["combo_rows"])
    expected_reps = set(run["cfg"].get("repeats") or []) or scored | set(run["boosters"])
    missing_reps = sorted(expected_reps - scored)
    if missing_reps:
        notes.append(f"no scored rows for repeat(s) {', '.join(str(r) for r in missing_reps)}")

    if scored:
        union = set().union(*run["combo_rows"].values())
        ragged = {r: sorted(union - c) for r, c in run["combo_rows"].items() if union - c}
        if ragged:
            notes.append("combo rows uneven across repeats: "
                         + "; ".join(f"repeat {r} lacks {', '.join(v)}" for r, v in ragged.items()))

    if stale:
        return "OUTDATED", notes
    if notes:
        return "INCOMPLETE", notes
    return "OK", notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/ensemble_logs")
    ap.add_argument("--verbose", action="store_true", help="also print per-repeat combo sets")
    args = ap.parse_args()

    base = _root / args.out_dir
    rows = []
    for d in sorted(base.iterdir()):
        run = scan(d)
        if run is None:
            continue
        status, notes = verdict(run)
        cfg = run["cfg"]
        mode = cfg.get("full_full_mode") if cfg.get("regime") == "full_full" else cfg.get("split")
        scored = sorted(set().union(*run["combo_rows"].values())) if run["combo_rows"] else []
        rows.append({
            "run": d.name,
            "mode": mode,
            "sources": " ".join(source_label(s) for s in cfg.get("sources", [])),
            "trials": cfg.get("n_trials") if cfg.get("tune_boost") else "fixed",
            "repeats": len(run["combo_rows"]),
            "combos": len(scored),
            "status": status,
        })
        if notes or args.verbose:
            print(f"\n{d.name}  [{status}]")
            for n in notes:
                print(f"    - {n}")
            if args.verbose:
                for rep in sorted(set(run["combo_rows"]) | set(run["boosters"]) | set(run["studies"])):
                    print(f"    repeat {rep}: scored={sorted(run['combo_rows'].get(rep, []))}")
                    print(f"              ensemble={sorted(run['ens_combos'].get(rep, []))}")
                    print(f"              boosters={sorted(run['boosters'].get(rep, []))}")
                    print(f"              studies={sorted(run['studies'].get(rep, []))}")

    print()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nOUTDATED = artifacts exist for combos the recorded ensemble was not fit over;"
          "\n           only the ensembling stage needs redoing, the heads are already trained.")


if __name__ == "__main__":
    main()
