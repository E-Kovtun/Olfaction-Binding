#!/usr/bin/env python
"""Read a mechanism-holdout run: the geometry table, the paired graph comparison, and the
sparse-matrix design check.

    .venv/bin/python scripts/analysis/mechanism_summary.py            # every run found
    .venv/bin/python scripts/analysis/mechanism_summary.py m2or__q99greedy

Every measure is printed in BOTH units on purpose. `z` says whether a number is
distinguishable from the permutation null; `value` says whether it is large. On M2OR they
part company badly -- the null's spread there is ~0.0015, so a Procrustes gap of 0.009 on a
0-1 scale reads as z = 6. Quoting the z alone would call that a big effect; it is not.

Three blocks per run:

  GEOMETRY   per model, the trust-weighted mean over classes, raw and as z.
  PAIRED     GNN+ESM minus GNN one-hot on the SAME (class, seed) cells -- the two share
             classes and seeds, so pairing is far more powerful than comparing two means.
             The share of cells in favour matters more than the interval: with five seeds
             the interval is wide almost by construction, while "25 of 30 cells" is hard to
             explain away as initialisation.
  DESIGN     sparse matrices only. `tested mask` is `retained profile` with the responses
             deleted, so it scores assay design alone; the percentage is how much of the
             profile's whole gain over the null the design already accounts for.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

GEOM = ["rsa", "cca", "procrustes"]
PROFILE, MASK = "retained profile", "tested mask"


def load(d):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    res = pd.read_csv(d / "readouts.csv")
    res["seed"] = res["seed"].astype(str)
    nul = pd.read_csv(d / "nulls.csv").set_index("cls")
    mn = pd.read_csv(d / "model_nulls.csv") if (d / "model_nulls.csv").exists() \
        else pd.DataFrame()
    return meta, res, nul, mn


def report(d, args):
    meta, res, nul, mn = load(d)
    classes = [c for c in meta["classes"] if c in set(res["cls"])]
    if "trust" in nul.columns and not args.equal:
        w = np.array([float(nul.loc[c, "trust"]) for c in classes])
    else:
        w = np.ones(len(classes))
    w = w / w.sum()

    def null(c, m, g, col=""):
        if len(mn):
            r = mn[(mn.cls == c) & (mn.model == m)]
            if len(r) and f"{g}_null{col}" in r:
                return float(r[f"{g}_null{col}"].iloc[0])
        key = f"{g}_null{col}"
        return float(nul.loc[c, key]) if key in nul.columns else np.nan

    def cell(c, m, g, seed=None):
        r = res[(res.cls == c) & (res.model == m)]
        if seed is not None:
            r = r[r.seed == seed]
        return float(r[g].mean()) if len(r) else np.nan

    def wmean(v):
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan

    def z_of(c, m, g, seed=None):
        sd = null(c, m, g, "_sd")
        if not np.isfinite(sd) or sd <= 1e-12:
            return np.nan
        return (cell(c, m, g, seed) - null(c, m, g)) / sd

    q = f"q={float(meta.get('variant', {}).get('q', 0.0)):g}"
    A, B = f"GNN+ESM full ({q})", f"GNN one-hot full ({q})"
    order = [m for m in ["raw ESM", B, A, "GNN + PCA128(ESM)", PROFILE, MASK]
             if m in set(res["model"])]
    seeds = sorted(set(res[res.model == A]["seed"]) & set(res[res.model == B]["seed"]))

    print(f"\n{'=' * 78}\n{d.name}   variant {meta.get('variant_name', 'q0cov')} "
          f"{meta.get('variant')}\n  {len(classes)} classes x {len(seeds)} seeds, "
          f"{meta['epochs']} epochs   weights: "
          f"{'trust' if 'trust' in nul.columns and not args.equal else 'equal'}")
    print("  " + "  ".join(f"{c}={float(nul.loc[c, 'trust']):.2f}" for c in classes
                           if "trust" in nul.columns))

    for g in GEOM:
        print(f"\n  --- {g}{'':<12} value      null       z")
        for m in order:
            v = wmean([cell(c, m, g) for c in classes])
            n0 = wmean([null(c, m, g) for c in classes])
            z = wmean([z_of(c, m, g) for c in classes])
            print(f"  {m:26} {v:+8.4f}  {n0:8.4f}  {z:+7.2f}")

    if len(seeds) and {A, B} <= set(res["model"]):
        from scipy.stats import t as _t
        print(f"\n  --- PAIRED  GNN+ESM minus GNN one-hot, on the same (class, seed)")
        for g in GEOM:
            cells = np.array([[z_of(c, A, g, s) - z_of(c, B, g, s) for c in classes]
                              for s in seeds])
            raw = np.array([[cell(c, A, g, s) - cell(c, B, g, s) for c in classes]
                            for s in seeds])
            per = np.array([wmean(row) for row in cells])
            hw = (_t.ppf(.975, len(per) - 1) * per.std(ddof=1) / len(per) ** .5
                  if len(per) > 1 else 0.0)
            pos = np.isfinite(cells) & (cells > 0)
            print(f"  {g:14} dz {per.mean():+6.2f} +- {hw:4.2f}   "
                  f"dvalue {wmean([np.nanmean(raw[:, j]) for j in range(len(classes))]):+8.4f}"
                  f"   for GNN+ESM: {pos.sum():3}/{np.isfinite(cells).sum()}")

    if MASK in set(res["model"]) and PROFILE in set(res["model"]):
        print(f"\n  --- DESIGN CHECK  how much of `{PROFILE}` is assay design, not binding")
        for g in GEOM:
            pv = wmean([cell(c, PROFILE, g) - null(c, PROFILE, g) for c in classes])
            mv = wmean([cell(c, MASK, g) - null(c, MASK, g) for c in classes])
            share = 100 * mv / pv if np.isfinite(pv) and abs(pv) > 1e-9 else np.nan
            print(f"  {g:14} profile over null {pv:+7.4f}   mask over null {mv:+7.4f}"
                  f"   design accounts for {share:5.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directory names (default: all found)")
    ap.add_argument("--root", default="results/mechanism_holdout")
    ap.add_argument("--equal", action="store_true",
                    help="equal class weights instead of trust -- print both and say so if "
                         "they disagree")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    if not root.is_absolute():
        root = _root / root
    dirs = [root / r for r in args.runs] if args.runs else \
        sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        if (d / "readouts.csv").exists() and (d / "meta.json").exists():
            report(d, args)
        else:
            print(f"\n{d.name}: no finished run here")


if __name__ == "__main__":
    main()
