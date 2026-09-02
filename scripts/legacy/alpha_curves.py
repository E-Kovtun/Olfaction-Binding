#!/usr/bin/env python
"""Turn the alpha sweeps into the long table the curve notebook draws, and audit it.

    python scripts/analysis/alpha_curves.py                    # write + summarise
    python scripts/analysis/alpha_curves.py -c                 # dense, for pasting
    python scripts/analysis/alpha_curves.py --nodes onehot --mol-source chemberta

The producer is `run_alpha_gate_sweep.py` with a dense `--alphas`; this is the second
layer, and the notebook is the third and computes nothing. One row here is one
(series, alpha, fold, seed, geometry, reference) measurement, in both the raw scale and
as z against that cell's own permutation null.

WHICH SCALE TO PLOT. Both are kept because they answer different questions and neither
alone is honest:

  raw   the geometry itself -- RSA and CCA are correlations and live in [0, 1], so the
        rise and fall are directly readable and comparable across datasets. Procrustes
        is a residual on its own scale and is only comparable within a panel.
  z     the same number against a permutation null. It says whether an alignment is
        distinguishable from chance at all, which the raw value cannot -- but its size
        is driven by the null's spread, so it inflates with the receptor count (1237 on
        M2OR gives z past 1000 where 24 on HC gives z of 20). Never read z ACROSS
        panels as an effect size.

READ THE DIAL, NOT THE SCORE. `esm` should fall with alpha and `fun` should rise; the
crossing is where the receptor cloud stops being structural and starts being
functional. `mono` below is the Spearman of the curve against alpha and is the
manipulation check -- if it is not near -1 for esm and +1 for fun, the knob is not
doing what its name says and nothing downstream of it means anything.

ONE SEED IS ENOUGH HERE, unlike the headline table. The geometry columns are a property
of the map alpha builds, not of the draw that trained it: at alpha=0 the emitted cloud
is the frozen branch exactly, and every run on record reproduced it to the byte. The
PREDICTION column on the same rows is NOT seed-stable, so treat it as a shape, not a
scoreboard -- headline_table.py is the scoreboard.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

GEOMS = ["rsa", "cca", "procrustes"]
REFS = ["esm", "fun"]
OF_RECORD = {"regression": "R2", "classification": "AUROC"}
# Column names come from the module that emits them; see orbind.dataset.
from orbind.dataset import METRIC_NAMES                          # noqa: E402
CLS, REG = METRIC_NAMES["classification"], METRIC_NAMES["regression"]
SERIES = ["dataset", "regime", "mol_source", "variant_tag", "nodes"]
KEY = SERIES + ["series"]


def _reader():
    """Reuse the headline table's filename parser -- one definition, one place to fix."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_ht", _root / "scripts/analysis/headline_table.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def label(r):
    """Series identity as one string -- the notebook's grouping key."""
    v = f"/{r.variant_tag}" if getattr(r, "variant_tag", "") else ""
    return f"{r.dataset}/{r.regime[:5]}{v}/{r.mol_source}/{r.nodes}"


def load(root, args, ht):
    ns = argparse.Namespace(dataset=args.dataset, mol_source=args.mol_source,
                            regime=args.regime)
    df = ht.load(pathlib.Path(root), ns)
    if args.nodes:
        df = df[df.nodes.isin(args.nodes)]
    if args.variant:
        df = df[df.variant_tag.isin(args.variant)]
    # `df` here is a filtered view, so assign rather than mutate in place
    df = df.assign(variant_tag=df["variant_tag"].fillna("").astype(str))
    # One string key per series. Selecting on this instead of comparing five columns
    # is what keeps a blank variant from silently dropping a whole dataset.
    return df.assign(series=[label(r) for r in df.itertuples()])


def melt(df):
    """Wide sweep rows -> one row per (series, alpha, fold, seed, geom, ref).

    Only the GATE arm has an alpha, so only it can be a curve. `graph_legacy` is the
    same model as alpha=1 up to a global scalar and `boost_full` has no receptor cloud
    at all; both are carried separately as reference LEVELS, not curve points."""
    gate = df[df.arm == "gate"].copy()
    out = []
    for g in GEOMS:
        for r in REFS:
            raw, z = f"{g}_{r}", f"{g}_{r}_z"
            if raw not in gate.columns:
                continue
            block = gate[KEY + ["alpha", "fold", "seed"]].copy()
            block["geom"], block["ref"] = g, r
            block["value"] = pd.to_numeric(gate[raw], errors="coerce")
            block["z"] = (pd.to_numeric(gate[z], errors="coerce")
                          if z in gate.columns else np.nan)
            met = [c for c in CLS + REG if c in gate.columns]
            for c in met:
                block[c] = pd.to_numeric(gate[c], errors="coerce")
            out.append(block)
    if not out:
        raise SystemExit("no geometry columns in any gate row -- was the sweep run "
                         "with --n-perm 0, or with --no-gate?")
    return pd.concat(out, ignore_index=True).dropna(subset=["value"])


def levels(df):
    """The two horizontal references a curve is read against, per series."""
    rows = []
    for arm, label in (("boost_full", "boost"), ("graph_legacy", "legacy")):
        q = df[df.arm == arm]
        if q.empty:
            continue
        met = [c for c in CLS + REG if c in q.columns and q[c].notna().any()]
        agg = q.groupby(KEY, dropna=False)[met].mean().reset_index()
        agg["level"] = label
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def audit(long, args):
    """Per (series, geom, ref): does the curve travel, and does it travel the right
    way. This is the manipulation check, and it comes before any reading of the plot."""
    from scipy.stats import spearmanr
    rows = []
    for k, g in long.groupby(SERIES + ["geom", "ref"], dropna=False):
        m = g.groupby("alpha")["value"].mean().sort_index()
        if len(m) < 3:
            continue
        rho = spearmanr(m.index.to_numpy(), m.to_numpy()).statistic
        rows.append(dict(zip(SERIES + ["geom", "ref"], k))
                    | dict(n_alpha=len(m), lo=float(m.min()), hi=float(m.max()),
                           travel=float(m.max() - m.min()), mono=float(rho),
                           at0=float(m.iloc[0]), at1=float(m.iloc[-1])))
    return pd.DataFrame(rows)


def crossings(long):
    """Where the cloud stops being structural and starts being functional.

    Each reference curve is min-max scaled WITHIN its own (series, geom) panel first,
    because raw RSA against ESM and raw RSA against a response matrix are not on a
    common scale and their literal intersection would be an artefact of that. After
    scaling, both run 0..1 and the crossing is the alpha where the cloud is equally far
    along both dials."""
    rows = []
    for k, g in long.groupby(SERIES + ["geom"], dropna=False):
        m = g.groupby(["ref", "alpha"])["value"].mean().unstack(0)
        if not {"esm", "fun"}.issubset(m.columns) or len(m) < 3:
            continue
        n = (m - m.min()) / (m.max() - m.min()).replace(0, np.nan)
        d = (n["esm"] - n["fun"]).dropna()
        if d.empty or d.iloc[0] * d.iloc[-1] > 0:
            rows.append(dict(zip(SERIES + ["geom"], k)) | dict(alpha_cross=np.nan))
            continue
        a = d.index.to_numpy(float)
        i = int(np.argmax(np.sign(d.to_numpy()) != np.sign(d.iloc[0])))
        x0, x1, y0, y1 = a[i - 1], a[i], d.iloc[i - 1], d.iloc[i]
        rows.append(dict(zip(SERIES + ["geom"], k))
                    | dict(alpha_cross=float(x0 - y0 * (x1 - x0) / (y1 - y0))))
    return pd.DataFrame(rows)


def _splits(long):
    """Splits per series, not pooled: m2or/inductive's repeats are the cold-molecule
    seeds 42-46 while every other series uses folds 1-5, so a pooled nunique reads as
    ten when every series in fact rests on five."""
    per = long.groupby("series")["fold"].nunique()
    return per.min() if per.min() == per.max() else f"{per.min()}-{per.max()}"


def readable(long, aud, cross, lev, args):
    print("=" * 92)
    print("=== ALPHA CURVES -- does the dial travel, and which way")
    print("=" * 92)
    al = sorted(long.alpha.unique())
    print(f"  {len(al)} alphas: {', '.join(f'{a:g}' for a in al)}")
    print(f"  {long.groupby(SERIES).ngroups} series, "
          f"{_splits(long)} splits each, {long.seed.nunique()} model seed(s)\n")
    # the series label carries dataset/regime/variant/source/nodes, so its width is
    # data-dependent -- measure it rather than guess and weld two columns together
    W = max([len(label(r)) for r in aud.itertuples()] + [len("series")]) + 2
    print(f"  {'series':<{W}}{'geom':<12}{'vs ESM: a=0 -> a=1':>26}{'mono':>8}"
          f"{'  |':<3}{'vs PROFILE: a=0 -> a=1':>26}{'mono':>8}")
    print("  " + "-" * (W + 81))
    for k, g in aud.groupby(SERIES, dropna=False):
        for geom in GEOMS:
            e = g[(g.geom == geom) & (g.ref == "esm")]
            f = g[(g.geom == geom) & (g.ref == "fun")]
            if e.empty and f.empty:
                continue
            cell = lambda q: ("" if q.empty else  # noqa: E731
                              f"{q.iloc[0].at0:>11.3f} ->{q.iloc[0].at1:>9.3f}")
            mono = lambda q: "" if q.empty else f"{q.iloc[0].mono:>+8.2f}"  # noqa: E731
            name = label(e.iloc[0] if not e.empty else f.iloc[0])
            print(f"  {name if geom == GEOMS[0] else '':<{W}}{geom:<12}"
                  f"{cell(e):>26}{mono(e)}{'  |':<3}{cell(f):>26}{mono(f)}")
        c = cross[np.logical_and.reduce([cross[s] == v for s, v in zip(SERIES, k)])]
        if len(c):
            txt = "  ".join(f"{r.geom}={r.alpha_cross:.2f}"
                            if np.isfinite(r.alpha_cross) else f"{r.geom}=never"
                            for r in c.itertuples())
            print(f"  {'':<{W}}{'crossover':<12}{txt}")
        print()

    print("  VERDICT")
    print("  " + "-" * 8)
    bad = aud[((aud.ref == "esm") & (aud.mono > -0.7))
              | ((aud.ref == "fun") & (aud.mono < 0.7))]
    if len(bad):
        print(f"    {len(bad)} curve(s) do not travel monotonically in the direction "
              f"the gate defines:")
        for r in bad.head(8).itertuples():
            print(f"      {label(r):<40}{r.geom}/{r.ref}  mono {r.mono:+.2f}  "
                  f"travel {r.travel:.3f}")
    else:
        print("    every curve is monotone in the expected direction "
              "(esm falls, profile rises)")
    flat = aud[aud.travel < 1e-3]
    if len(flat):
        print(f"    {len(flat)} curve(s) are flat (travel < 0.001) -- the knob moved "
              f"nothing there")
    if long.seed.nunique() > 1:
        print(f"    NOTE {long.seed.nunique()} seeds are present; the geometry is "
              f"seed-stable but the prediction columns are not")


def compact(long, aud, cross, lev, args):
    print(f"#ACURVE1 alphas={len(long.alpha.unique())} "
          f"series={long.groupby(SERIES).ngroups} splits={_splits(long)} "
          f"seeds={long.seed.nunique()}")
    print("#cols series geom ref n_alpha at0 at1 travel mono")
    for r in aud.itertuples():
        print(f"{label(r)} {r.geom} {r.ref} {r.n_alpha} {r.at0:.3f} {r.at1:.3f} "
              f"{r.travel:.3f} {r.mono:+.2f}")
    print("#cols series geom alpha_cross")
    for r in cross.itertuples():
        v = f"{r.alpha_cross:.3f}" if np.isfinite(r.alpha_cross) else "."
        print(f"{label(r)} {r.geom} {v}")
    print("#end")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/graph/v8_alpha_gate")
    ap.add_argument("--dataset", nargs="+", default=None)
    ap.add_argument("--regime", nargs="+", default=None,
                    choices=["transductive", "inductive"])
    ap.add_argument("--mol-source", nargs="+", default=None)
    ap.add_argument("--variant", nargs="+", default=None)
    ap.add_argument("--nodes", nargs="+", default=None, choices=["esm", "onehot"],
                    help="default: whatever is on disk. The one-hot series is the only "
                         "one where alpha is an honest fraction of structure -- with ESM "
                         "node features it still reaches the cloud through message "
                         "passing at alpha=1")
    ap.add_argument("--out", default=None,
                    help="long CSV for the notebook (default: <root>/curves_long.csv)")
    ap.add_argument("-c", "--compact", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_absolute() and not root.exists():
        root = _root / root
    ht = _reader()
    df = load(root, args, ht)
    long = melt(df)
    lev = levels(df)
    aud, cross = audit(long, args), crossings(long)

    out = pathlib.Path(args.out) if args.out else root / "curves_long.csv"
    long.to_csv(out, index=False)
    lev.to_csv(out.with_name(out.stem + "_levels.csv"), index=False)
    (compact if args.compact else readable)(long, aud, cross, lev, args)
    print(f"\nwrote {out}  ({len(long)} rows)"
          f"\n      {out.with_name(out.stem + '_levels.csv')}  ({len(lev)} rows)"
          f"\nnotebook: notebooks/graph/mechanism_holdout/alpha_gate_curves.ipynb")


if __name__ == "__main__":
    main()
