"""One line per ensemble run: when it was written, by which env, with what.

Built for one question that keeps coming back — *which runs were produced by a
given environment in a given window* — after a broken dependency in two of the
five venvs silently changed the boosting head (2026-09-20, see
orbind/docs/gotchas.md). `config.json` records the interpreter now; for everything
older, the env is inferred from the run's pair-level source, which is what decides
it in practice.

Deliberately narrow output: date, env, method, repeats, path. One line each, no
wrapping, sortable by eye. `--by method` groups instead of sorting by time.

    python scripts/analysis/run_dates.py
    python scripts/analysis/run_dates.py --since 2026-09-01 --env .venv-controls
    python scripts/analysis/run_dates.py --by method --root results/ensemble_logs_esm3
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

_repo = pathlib.Path(__file__).resolve()
while not (_repo / "pyproject.toml").exists():
    _repo = _repo.parent
sys.path.insert(0, str(_repo))

#: Pair-level source -> the env that can import it. Same table as the relauncher's;
#: kept here too so this script stays readable on its own.
ENV_FOR_SOURCE = {
    "molor": ".venv-molor",
    "lorax": ".venv-controls",
    "prosmith": ".venv-controls",
    "hladis": ".venv",
    "gnn_signed": ".venv",
}
SHORT_ENV = {".venv": "proj", ".venv-controls": "ctl", ".venv-molor": "molor"}


def method_of(config: dict) -> str:
    """The pair-level source that names the run, or 'boost' when there is none."""
    for spec in config.get("sources") or []:
        kind = str(spec).partition("=")[2].split(":")[0]
        if kind in ENV_FOR_SOURCE:
            return kind
    return "boost"


def env_of(config: dict) -> str:
    """Recorded interpreter if the run wrote one, else inferred from its method."""
    recorded = config.get("python")
    if recorded:
        stem = pathlib.PurePosixPath(recorded).parts
        for part in stem:
            if part.startswith(".venv"):
                return part
    return ENV_FOR_SOURCE.get(method_of(config), "?")


def n_repeats(run_dir: pathlib.Path) -> int:
    path = run_dir / "metrics.csv"
    if not path.exists() or path.stat().st_size == 0:
        return 0
    import pandas as pd
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    return df["repeat"].nunique() if "repeat" in df.columns else 0


def when_of(run_dir: pathlib.Path) -> datetime.datetime | None:
    """When the run FINISHED: metrics.csv's mtime, falling back to the log's.

    config.json's mtime is start-up, which is the wrong end for asking what an
    environment produced -- a run that began before an upgrade can finish after it.
    """
    for name in ("metrics.csv", "log.txt", "config.json"):
        p = run_dir / name
        if p.exists():
            return datetime.datetime.fromtimestamp(p.stat().st_mtime)
    return None


def collect(root: pathlib.Path) -> list[dict]:
    out = []
    if not root.is_dir():
        return out
    for cfg in root.glob("*/*/config.json"):
        try:
            config = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_dir = cfg.parent
        out.append({
            "when": when_of(run_dir),
            "env": env_of(config),
            "method": method_of(config),
            "repeats": n_repeats(run_dir),
            "xgboost": config.get("xgboost", ""),
            "path": run_dir,
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", nargs="+", default=None,
                    help="ensemble_logs roots (default: every results/ensemble_logs*)")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD, drop anything older")
    ap.add_argument("--env", default=None, help="only runs from this env, e.g. .venv-controls")
    ap.add_argument("--method", default=None, help="only this pair-level source")
    ap.add_argument("--by", default="date", choices=["date", "method", "env"])
    args = ap.parse_args(argv)

    roots = ([pathlib.Path(r) if pathlib.Path(r).is_absolute() else _repo / r
              for r in args.root] if args.root
             else sorted((_repo / "results").glob("ensemble_logs*")))

    rows = [r for root in roots for r in collect(root)]
    if args.since:
        cutoff = datetime.datetime.fromisoformat(args.since)
        rows = [r for r in rows if r["when"] and r["when"] >= cutoff]
    if args.env:
        rows = [r for r in rows if r["env"] == args.env]
    if args.method:
        rows = [r for r in rows if r["method"] == args.method]

    if not rows:
        print("no runs matched")
        return 0

    key = {"date": lambda r: (r["when"] or datetime.datetime.min,),
           "method": lambda r: (r["method"], r["when"] or datetime.datetime.min),
           "env": lambda r: (r["env"], r["when"] or datetime.datetime.min)}[args.by]
    rows.sort(key=key)

    for r in rows:
        when = r["when"].strftime("%m-%d %H:%M") if r["when"] else "  --  --:--"
        # results/ensemble_logs_esm3/pool/run -> esm3/pool/run; the common prefix
        # is noise when every line carries it.
        # .../results/ensemble_logs_esm3/pool/run -> esm3/pool/run. The common
        # prefix is noise when every line carries it; a root outside results/
        # (a test tree, a copy) keeps its last three components instead.
        try:
            rel = r["path"].relative_to(_repo / "results").as_posix()
        except ValueError:
            rel = "/".join(r["path"].parts[-3:])
        rel = rel.replace("ensemble_logs_", "").replace("ensemble_logs", "main", 1)
        print("%s  %-5s %-8s %d/5 %-7s %s" % (
            when, SHORT_ENV.get(r["env"], r["env"]), r["method"], r["repeats"],
            r["xgboost"] or "?", rel))

    print(f"\n{len(rows)} run(s). env column: recorded interpreter where config.json "
          f"has one, else inferred from the method. '?' in the xgboost column means "
          f"the run predates version recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
