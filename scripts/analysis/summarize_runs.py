"""One-line summary of every ensemble run on disk.

    python scripts/analysis/summarize_runs.py
    python scripts/analysis/summarize_runs.py --pool cc-ourind          # substring filter
    python scripts/analysis/summarize_runs.py --pool m2or-trans --folds # per-fold values

Walks `results/ensemble_logs/<pool>/<run>/metrics.csv` and prints, per pool, one
row per (run, combo): how many folds it has and each metric as mean +- 95% t-CI.

Deliberate choices
------------------
* **Every combo is its own row.** A run with `--combos "1 2 12"` produced three
  models; collapsing them to a "best" would hide which one the number came from,
  and picking a favourite per method is exactly the bookkeeping that went wrong
  before. Most runs have one combo, so this stays short.
* **The task decides the columns**, read from config.json (`ofm` defaults to
  regression). Mixing AUROC and R2 in one table is meaningless, so classification
  and regression pools simply print different headers.
* **`naive` is printed for regression whenever the run recorded it**, because R2
  there is measured against the TEST mean while the naive predictor uses the
  TRAIN mean -- on a cold-molecule split that gap is the whole story (cc/scaf
  fold 1: naive R2 = -4.92). Runs that predate the naive row show "--".
* Nothing is filtered by fold count: a 1/5 run is shown as 1/5 rather than
  dropped, since a half-finished run is the thing you most need to notice.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as _t

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

BASE = _root / "results" / "ensemble_logs"
COLS = {"classification": ["AUROC", "AUPRC", "MCC", "F1"],
        "regression": ["R2", "RMSE", "MAE", "Pearson", "Spearman"]}
SORT_BY = {"classification": "AUROC", "regression": "R2"}


def cell(vals) -> str:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "--".rjust(15)
    if len(v) == 1:
        return f"{v[0]:.3f}".rjust(15)
    h = _t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.3f}±{h:.3f}".rjust(15)


def task_of(run: pathlib.Path) -> tuple[str, str]:
    cfg = run / "config.json"
    c = json.loads(cfg.read_text()) if cfg.exists() else {}
    regime = c.get("regime", "?")
    task = c.get("task") or ("regression" if regime == "ofm" else "classification")
    if regime == "ofm":
        scope = f"{c.get('dataset', '?')}/{c.get('split_family', '?')}"
    elif regime == "full_full":
        scope = c.get("full_full_mode", "?")
    else:
        scope = c.get("split", "?")
    return task, scope


def summarize(pool: pathlib.Path, per_fold: bool, drop: set[str]) -> None:
    runs = sorted(r for r in pool.iterdir() if r.is_dir())
    rows, naive_vals, task, scope = [], None, None, None
    for r in runs:
        m = r / "metrics.csv"
        if not m.exists():
            rows.append({"run": r.name, "combo": "—", "folds": "нет metrics.csv"})
            continue
        t, sc = task_of(r)
        task = task or t
        scope = scope or sc
        d = pd.read_csv(m)
        cols = [c for c in COLS[t] if c in d.columns]
        nv = d[d["kind"] == "naive"] if "naive" in set(d["kind"]) else None
        if nv is not None and len(nv) and naive_vals is None:
            naive_vals = {c: nv[c].values for c in cols}
        combo = d[d["kind"] == "combo"]
        for spec in drop:
            run_pat, _, name = spec.rpartition(":")
            if run_pat and run_pat not in r.name:
                continue                  # scoped to another run
            combo = combo[combo["name"] != name]
        for name, g in combo.groupby("name", sort=True):
            g = g.sort_values("repeat")
            row = {"run": r.name, "combo": name, "folds": f"{g['repeat'].nunique()}/5"}
            row.update({c: g[c].values for c in cols})
            row["_per_fold"] = dict(zip(g["repeat"], g[SORT_BY[t]])) if SORT_BY[t] in g else {}
            rows.append(row)

    if task is None:
        print(f"\n{pool.name}: нечего показывать")
        return
    cols = COLS[task]
    key = SORT_BY[task]
    scored = [r for r in rows if key in r]
    scored.sort(key=lambda r: -np.nanmean(np.asarray(r[key], float)))
    other = [r for r in rows if key not in r]

    width = max([len(r["run"]) for r in rows] + [20]) + 2
    print("=" * (width + 12 + 15 * len(cols)))
    print(f"{pool.name}   [{scope}, {task}]")
    print("=" * (width + 12 + 15 * len(cols)))
    print("прогон".ljust(width) + "комбо".ljust(12) + "".join(c.rjust(15) for c in cols) + "  фолдов")
    for r in scored + other:
        line = r["run"].ljust(width) + str(r["combo"]).ljust(12)
        line += "".join(cell(r[c]) if c in r else "--".rjust(15) for c in cols)
        print(line + f"  {r['folds']}")
    if task == "regression":
        line = "naive (среднее train)".ljust(width) + "—".ljust(12)
        line += "".join(cell(naive_vals[c]) if naive_vals and c in naive_vals
                        else "--".rjust(15) for c in cols)
        print("-" * (width + 12 + 15 * len(cols)))
        print(line + ("  5/5" if naive_vals else "  (не записан)"))

    if per_fold:
        print(f"\n  {key} по фолдам:")
        folds = sorted({f for r in scored for f in r.get("_per_fold", {})})
        print("  " + "прогон/комбо".ljust(width + 12) + "".join(f"f{f}".rjust(9) for f in folds))
        for r in scored:
            pf = r.get("_per_fold", {})
            print("  " + f"{r['run']} [{r['combo']}]".ljust(width + 12)
                  + "".join((f"{pf[f]:.3f}" if f in pf else "—").rjust(9) for f in folds))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=None, help="substring filter over pool folder names")
    ap.add_argument("--folds", action="store_true",
                    help="also print the primary metric fold by fold (a fold on the ofm "
                         "datasets is a different set of molecules, not a reseed, so its "
                         "spread is structural)")
    ap.add_argument("--drop-combo", nargs="*", default=[],
                    help="rows to hide, as `combo` or `run_substring:combo`. SCOPE IT: a bare "
                         "`cls+mol` also hides every GNN run, whose only combo is cls+mol. To "
                         "drop hladis's secondary row (a wash against its cls: 0.729 vs 0.734 "
                         "AUPRC transductive) use `--drop-combo hladis:cls+mol`. Hides the row "
                         "only; metrics.csv is untouched.")
    ap.add_argument("--base", default=str(BASE))
    args = ap.parse_args()

    base = pathlib.Path(args.base)
    if not base.exists():
        sys.exit(f"{base} не найдено")
    pools = [p for p in sorted(base.iterdir()) if p.is_dir()
             and (args.pool is None or args.pool in p.name)]
    if not pools:
        sys.exit(f"под фильтр --pool {args.pool!r} ничего не попало")
    for p in pools:
        summarize(p, args.folds, set(args.drop_combo))


if __name__ == "__main__":
    main()
