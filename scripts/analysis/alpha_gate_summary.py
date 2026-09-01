#!/usr/bin/env python
"""Read a v8 alpha-gate sweep: does the dial move the geometry, and does it pay.

    python scripts/analysis/alpha_gate_summary.py            # every csv found
    python scripts/analysis/alpha_gate_summary.py -c         # dense, for pasting
    python scripts/analysis/alpha_gate_summary.py hc_rand --metric Spearman

Three blocks per (dataset, regime):

  PREDICTION  every arm's test metric, mean over folds, and the two comparisons that
              matter -- against `boost_full` ([raw ESM || molecule], what a graph has
              to beat) and against `graph_legacy` (the pre-v8 model). Both are PAIRED
              by fold: the arms share folds, so the per-fold difference is far
              stronger evidence than two means, and `won` counts the folds in favour.
  GEOMETRY    the receptor cloud's alignment with raw ESM and with the train response
              profile, as z against each cell's own permutation null. This is the
              dial itself. `alpha=0` should sit high against ESM and at the null
              against the profile; `alpha=1` is the historical graph, which was
              measured AT the null against ESM.
  VERDICT     the best alpha, whether it beats both references, and -- separately --
              whether the geometry actually travelled. A dial that pays without
              moving is a different (and more suspicious) result than one that moves.

R2 is against the TEST mean, so `naive` is not always 0: read it as the honest zero
for that fold, and a model below it lost to predicting a constant.
"""
from __future__ import annotations

import argparse
import fnmatch
import pathlib

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

# Both task families; whichever columns a file actually carries are the ones printed,
# so a regression sweep and a classification one can sit in the same directory.
METRIC_SETS = {"regression": ["R2", "RMSE", "MAE", "Pearson", "Spearman"],
               "classification": ["AUROC", "AUPRC", "MCC", "F1"]}
METRICS = METRIC_SETS["regression"]
GEOMS = ["rsa", "cca", "procrustes"]
GL = {"rsa": "RSA", "cca": "CCA", "procrustes": "Proc"}
REFS = ["esm", "fun"]
REF_LABEL = {"esm": "vs raw ESM", "fun": "vs response profile"}
BASE, LEGACY = "boost_full", "graph_legacy"


def arm_label(r):
    return f"gate a={r['alpha']:.2f}" if r["arm"] == "gate" else r["arm"]


def order_arms(df):
    """naive, boost_full, graph_legacy, then the gate by increasing alpha."""
    fixed = [a for a in ("naive", BASE, LEGACY) if a in set(df["arm"])]
    gates = sorted({float(a) for a in df.loc[df.arm == "gate", "alpha"].dropna()})
    return [(a, None) for a in fixed] + [("gate", a) for a in gates]


def pick(df, arm, alpha):
    q = df[df.arm == arm]
    return q if alpha is None else q[np.isclose(q["alpha"].astype(float), alpha)]


def paired(df, arm, alpha, ref_arm, metric):
    """Per-fold difference against a reference arm run on the SAME folds."""
    a = pick(df, arm, alpha).set_index("fold")[metric]
    b = pick(df, ref_arm, None).set_index("fold")[metric]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).astype(float).dropna()
    if d.empty:
        return np.nan, np.nan, 0, 0
    sem = d.std() / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return float(d.mean()), float(sem), int((d > 0).sum()), int(len(d))


def mean_sem(df, arm, alpha, col):
    v = pd.to_numeric(pick(df, arm, alpha).get(col, pd.Series(dtype=float)),
                      errors="coerce").dropna()
    if v.empty:
        return np.nan, np.nan, 0
    return float(v.mean()), (float(v.std() / np.sqrt(len(v))) if len(v) > 1 else np.nan), len(v)


def fmt(v, nd=3, w=0):
    return f"{'':>{w}}" if v is None or not np.isfinite(v) else f"{v:{w}.{nd}f}"


def readable(tag, df, args):
    arms = order_arms(df)
    print("=" * 84)
    print(f"=== {tag}   folds {sorted(set(df['fold']))}   metric of record: {args.metric}")
    print("=" * 84)
    bad = df[df["status"].astype(str).str.startswith("failed")] if "status" in df else df.iloc[:0]
    if len(bad):
        print(f"  !! {len(bad)} failed cells: "
              + "; ".join(f"{arm_label(r)} fold {r['fold']}: {r['status'][:60]}"
                          for _, r in bad.head(4).iterrows()))

    print("\n  PREDICTION")
    print("  " + "-" * 12)
    head = f"    {'arm':<14}{'n':>3}" + "".join(f"{m:>10}" for m in METRICS)
    print(head + f"{'d vs boost':>12}{'won':>6}{'d vs legacy':>13}{'won':>6}")
    for arm, alpha in arms:
        m, s, n = mean_sem(df, arm, alpha, args.metric)
        cells = "".join(f"{fmt(mean_sem(df, arm, alpha, k)[0]):>10}" for k in METRICS)
        line = f"    {(arm_label({'arm': arm, 'alpha': alpha})):<14}{n:>3}{cells}"
        for ref in (BASE, LEGACY):
            if arm == ref or ref not in set(df["arm"]):
                line += f"{'':>12}{'':>6}" if ref == BASE else f"{'':>13}{'':>6}"
                continue
            d, ds, w, tot = paired(df, arm, alpha, ref, args.metric)
            wide = 12 if ref == BASE else 13
            line += (f"{fmt(d, 3):>{wide}}{f'{w}/{tot}':>6}" if np.isfinite(d)
                     else f"{'':>{wide}}{'':>6}")
        print(line)
    print(f"    (deltas are per-fold differences in {args.metric}, paired; "
          f"'won' = folds in favour)")

    zcols = [c for c in df.columns if c.endswith("_z")]
    if zcols:
        print("\n  GEOMETRY -- receptor cloud, z vs its own permutation null")
        print("  " + "-" * 12)
        print(f"    {'arm':<14}" + "".join(
            f"{GL[g] + '/' + r[:3]:>11}" for r in REFS for g in GEOMS))
        for arm, alpha in arms:
            if arm in ("naive", BASE):
                continue
            vals = "".join(f"{fmt(mean_sem(df, arm, alpha, f'{g}_{r}_z')[0], 1):>11}"
                           for r in REFS for g in GEOMS)
            print(f"    {(arm_label({'arm': arm, 'alpha': alpha})):<14}{vals}")
        print("    (esm = how structural the cloud came out, fun = how functional)")

    print("\n  VERDICT")
    print("  " + "-" * 12)
    gates = [(a, alpha) for a, alpha in arms if a == "gate"]
    if not gates:
        print("    no gate arm in this file")
        return
    scored = [(mean_sem(df, "gate", al, args.metric)[0], al) for _, al in gates]
    scored = [(v, al) for v, al in scored if np.isfinite(v)]
    if not scored:
        print("    every gate cell is NaN")
        return
    best_v, best_a = max(scored, key=lambda t: t[0])
    print(f"    best alpha = {best_a:.2f}  ({args.metric} {best_v:+.3f})"
          f"{'  <- an interior alpha' if 0 < best_a < 1 else '  <- an endpoint'}")
    for ref, name in ((BASE, "boost [ESM||mol]"), (LEGACY, "the pre-v8 graph")):
        if ref not in set(df["arm"]):
            continue
        d, ds, w, tot = paired(df, "gate", best_a, ref, args.metric)
        if not np.isfinite(d):
            continue
        se = "" if not np.isfinite(ds) else f" +/-{ds:.3f}"
        call = "beats" if (w > tot / 2 and d > 0) else ("ties" if abs(d) < 1e-9 else "loses to")
        print(f"    {call} {name}: {d:+.3f}{se} over {tot} folds, {w} in favour")
    span = [mean_sem(df, "gate", al, "rsa_esm_z")[0] for _, al in gates]
    span = [v for v in span if np.isfinite(v)]
    if len(span) > 1:
        print(f"    dial travel (RSA vs ESM, z): {min(span):+.1f} .. {max(span):+.1f}"
              + ("   -- the geometry did move" if max(span) - min(span) > 3
                 else "   -- the geometry BARELY MOVED; the knob is not doing its job"))


def compact(tag, df, args):
    arms = order_arms(df)
    folds = sorted(set(df["fold"]))
    print(f"#AGATE1 {tag} folds={','.join(map(str, folds))} metric={args.metric} "
          f"arms={len(arms)}")
    print("#cols arm n " + " ".join(METRICS) + " dBoost won dLegacy won "
          + " ".join(f"{g[:4]}/{r}" for r in REFS for g in GEOMS))
    for arm, alpha in arms:
        lbl = ("g" + f"{alpha:.2f}") if arm == "gate" else {"naive": "naive",
                                                            BASE: "boost", LEGACY: "legacy"}[arm]
        n = mean_sem(df, arm, alpha, args.metric)[2]
        cells = " ".join(fmt(mean_sem(df, arm, alpha, k)[0]) or "." for k in METRICS)
        out = f"{lbl:<7} {n} {cells}"
        for ref in (BASE, LEGACY):
            if arm == ref or ref not in set(df["arm"]):
                out += " . ."
                continue
            d, _, w, tot = paired(df, arm, alpha, ref, args.metric)
            out += f" {fmt(d) or '.'} {w}/{tot}"
        out += " " + " ".join(fmt(mean_sem(df, arm, alpha, f"{g}_{r}_z")[0], 1) or "."
                              for r in REFS for g in GEOMS)
        print(out)
    print("#end")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*",
                    help="which runs to print (default: all). A stem (hc_rand), a glob "
                         "(*_onehot) or any substring of one (onehot) all work")
    ap.add_argument("-x", "--exclude", nargs="+", default=None,
                    help="drop runs matching these, same syntax. Handy for the archived "
                         "series: -x '*__foldseed' '*__rmsnorm*'")
    ap.add_argument("--root", default="results/graph/v8_alpha_gate")
    ap.add_argument("--metric", default=None,
                    choices=METRIC_SETS["regression"] + METRIC_SETS["classification"],
                    help="metric the deltas and the best-alpha verdict are read on. "
                         "Default: R2 for a regression sweep, AUROC for a classification "
                         "one -- and a metric the file does not carry falls back to that")
    ap.add_argument("-c", "--compact", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_absolute() and not root.exists():
        root = _root / root
    found = sorted(root.glob("metrics_*.csv")) if root.exists() else []
    names = {p: p.stem.replace("metrics_", "").lower() for p in found}

    def hit(name, pat):
        pat = pat.lower()
        return name == pat or fnmatch.fnmatch(name, pat) or pat in name

    if args.runs:
        found = [p for p in found if any(hit(names[p], r) for r in args.runs)]
    if args.exclude:
        found = [p for p in found if not any(hit(names[p], x) for x in args.exclude)]
    if not found:
        raise SystemExit(f"nothing matched under {root}"
                         + (f"; present: {sorted(names.values())}" if names else ""))
    for p in found:
        df = pd.read_csv(p)
        # pick the task family this file was written with, and the metric to rank by
        global METRICS
        have = [k for k, cols in METRIC_SETS.items()
                if any(c in df.columns and df[c].notna().any() for c in cols)]
        METRICS = METRIC_SETS[have[0]] if have else METRIC_SETS["regression"]
        args = argparse.Namespace(**vars(args))
        if args.metric not in METRICS:
            args.metric = METRICS[0]
        if "status" in df.columns:
            df = df[~df["status"].astype(str).str.startswith("failed") | True]
        (compact if args.compact else readable)(p.stem.replace("metrics_", ""), df, args)


if __name__ == "__main__":
    main()
