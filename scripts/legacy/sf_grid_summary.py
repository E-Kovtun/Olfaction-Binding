#!/usr/bin/env python
"""Read a structure x function grid run and print it compactly enough to paste.

    .venv/bin/python scripts/analysis/sf_grid_summary.py              # every dataset found
    .venv/bin/python scripts/analysis/sf_grid_summary.py hc --per-class
    .venv/bin/python scripts/analysis/sf_grid_summary.py -c           # ~21 lines, for pasting

Same numbers as notebooks/graph/mechanism_holdout/structure_function_grid.ipynb, as text.
The notebook is for looking; this is for quoting.

`-c` / `--compact` drops the prose and prints one row per (geometry, reference) with the
values in the fixed cell order announced once by the `#cells` header, `.` for a cell that
was never run:

    #SFGRID1 ds= unit= cells= cls= seeds= ep= crit= q= nperm=
    #cells    the cell order every row below is in, as k|phi ("all" = every component)
    n         trainings behind each cell (classes x seeds)
    <g>/esm   alignment with raw ESM         "how structural did this come out"
    <g>/fun   alignment with the profile     "how functional did this come out"
    <g>/tar   alignment with the class       the result
    <g>/pos   fun - esm, one decimal         where the cell actually landed
    V <g>     best cell, its n, and the two PURE sources (pureS = k=all,phi=0;
              pureF = k=0,phi=1) plus lab00 = (k=0,phi=0), so a mix that beats both
              can be read off directly
    #edges    phi:pos+neg -- what the edge knob actually left behind
    #gaps     cells never run, on the observed axes

Five blocks per dataset:

  STATUS    what exists -- cells, classes, seeds, and which of the (k, phi) combinations on
            the observed axes were never run. The grid is filled one trajectory at a time,
            so a partial surface is the normal state and every table below is drawn on
            whatever is there.
  CHECK     the manipulation check, as the two arms: does k buy alignment with ESM and phi
            alignment with the response profile. If these do not move, nothing else is
            interpretable -- the axes would not be the axes they are named for.
  TARGET    the (k, phi) surface against the held-out class, per geometry. The result.
  POSITION  sim(func) - sim(esm) per cell: where each model actually landed, which is not
            the same as where its knobs were set.
  VERDICT   the best cell against the class versus the two PURE sources by name --
            (k=all, phi=0) and (k=0, phi=1). The claim the grid tests is that a mix beats
            both, so a mixed cell winning is the positive answer and a pure corner winning
            is the equally clean negative one. `(k=0, phi=0)` is called out separately: it
            has neither node features nor messages but the decoder still supervises every
            pair, so it is a third baseline ("labels without message passing"), not a floor.

Everything is printed as z against each cell's own permutation null, with `--raw` for the
geometries' own units. z is comparable within a panel and NOT across datasets -- the null's
spread shrinks as the receptor panel grows, so a z of 6 on HC (24 receptors) and one on CC
(50) are different claims.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

GEOMS = ["rsa", "cca", "procrustes"]
REFS = ["esm", "func", "target"]
KEY = ["k", "phi", "cls", "seed"]
GL = {"rsa": "RSA", "cca": "CCA", "procrustes": "Proc"}
PURE = {(-1.0, 0.0): "pure structure (k=all,phi=0)", (0.0, 1.0): "pure function (k=0,phi=1)"}
LABELS = {(0.0, 0.0): "label factorisation (k=0,phi=0)"}
# The arms the grid script walks, named rather than inferred so a half-finished run still
# gets its arms printed instead of a guess.
ARMS = [("k-arm  (phi=1)", "k", lambda k, p: p == 1.0),
        ("phi-arm  (k=all)", "phi", lambda k, p: k < 0),
        ("diagonal", "phi",
         lambda k, p: (k, p) in {(-1, 0.0), (32, 0.05), (8, 0.25), (2, 0.5), (0, 1.0)})]


def klab(k):
    return "all" if k < 0 else str(int(k))


def kord(vals):
    return sorted(set(vals), key=lambda k: (k < 0, k))


def load(d):
    """grid.csv + nulls.csv -> one row per (cell, seed, geometry, reference)."""
    grid = pd.read_csv(d / "grid.csv").drop_duplicates(subset=KEY, keep="last")
    extra = [c for c in ("n_pos", "n_neg", "k_eff", "dim_z", "seconds") if c in grid.columns]
    parts = []
    for g in GEOMS:
        for r in REFS:
            col = f"{g}__{r}"
            if col not in grid.columns:
                continue
            s = grid[KEY + extra].copy()
            s["geom"], s["ref"], s["value"] = g, r, grid[col].to_numpy()
            parts.append(s)
    if not parts:
        raise SystemExit(f"{d/'grid.csv'} carries no <geometry>__<reference> columns")
    long = pd.concat(parts, ignore_index=True)
    npth = d / "nulls.csv"
    if npth.exists():
        nul = pd.read_csv(npth).drop_duplicates(subset=["k", "phi", "cls", "ref"], keep="last")
        nparts = []
        for g in GEOMS:
            a, b = f"{g}_null", f"{g}_null_sd"
            if a in nul.columns:
                s = nul[["k", "phi", "cls", "ref", a, b]].rename(
                    columns={a: "null", b: "null_sd"})
                s["geom"] = g
                nparts.append(s)
        if nparts:
            # a null is computed once per cell, on the first seed; it attaches to every seed
            long = long.merge(pd.concat(nparts, ignore_index=True),
                              on=["k", "phi", "cls", "ref", "geom"], how="left")
    for c in ("null", "null_sd"):
        if c not in long.columns:
            long[c] = np.nan
    long["z"] = (long["value"] - long["null"]) / long["null_sd"].where(long["null_sd"] > 1e-12)
    mp = d / "meta.json"
    return long, (json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {})


def fmt(v, w=6, nd=2):
    return " " * w if not np.isfinite(v) else f"{v:+{w}.{nd}f}"


def pivot(sub, col, ks, phis):
    return (sub.pivot_table(index="k", columns="phi", values=col, aggfunc="mean")
            .reindex(index=ks, columns=phis))


def table(m, ks, phis, nd=2):
    """An already-pivoted (k, phi) matrix as text, blank where a cell was never run."""
    out = ["    k\\phi " + "".join(f"{p:>7g}" for p in phis)]
    for k in ks:
        out.append(f"    {klab(k):>5} " + "".join(fmt(m.loc[k, p], 7, nd) for p in phis))
    return "\n".join(out)


def block(name):
    return f"\n  {name}\n  " + "-" * (len(name) + 2)


def compact(ds, long, meta, col, unit, cells, args):
    """The same numbers with the prose removed: ~15 lines per dataset, meant to be pasted.

    One row per (geometry, reference), values in the fixed cell order of the `#cells` line,
    `.` for a cell that was never run. Positional rather than labelled because the labels
    are what makes the readable format long, and the header pins them once.
    """
    classes, seeds = sorted(set(long["cls"])), sorted(set(long["seed"]))
    v = meta.get("variant", {}) if meta else {}
    print(f"#SFGRID1 ds={ds} unit={unit} cells={len(cells)} cls={','.join(classes)} "
          f"seeds={','.join(str(s) for s in seeds)} ep={meta.get('epochs', '?')} "
          f"crit={v.get('criterion', '?')} q={v.get('q', '?')} nperm={meta.get('n_perm', '?')}")
    print("#cells " + " ".join(f"{klab(k)}|{p:g}" for k, p in cells))

    def line(tag, vals, nd=2):
        print(f"{tag:<9}" + " ".join("." if not np.isfinite(x) else f"{x:+.{nd}f}"
                                     for x in vals))

    def series(part, g, r):
        m = part[(part.geom == g) & (part.ref == r)].groupby(["k", "phi"])[col].mean()
        return [m.get((k, p), np.nan) for k, p in cells]

    for gname, part in ([(c, long[long.cls == c]) for c in classes]
                        if args.per_class and len(classes) > 1 else [(None, long)]):
        if gname:
            print(f"#cls {gname}")
        n = part[(part.geom == GEOMS[0]) & (part.ref == REFS[0])].groupby(["k", "phi"]).size()
        print("n        " + " ".join(str(int(n.get(c, 0))) for c in cells))
        for g in GEOMS:
            for r in REFS:
                line(f"{g[:4]}/{r[:3]}", series(part, g, r))
            line(f"{g[:4]}/pos", [f - e for f, e in zip(series(part, g, "func"),
                                                        series(part, g, "esm"))], nd=1)
        for g in GEOMS:
            q = (part[(part.geom == g) & (part.ref == "target")]
                 .groupby(["k", "phi"])[col].agg(["mean", "count"]).reset_index()
                 .sort_values("mean", ascending=False))
            if q.empty:
                continue
            t = q.iloc[0]
            got = {}
            for c, nm in list(PURE.items()) + list(LABELS.items()):
                hit = q[(q.k == c[0]) & (q.phi == c[1])]
                got[nm.split(" (")[0]] = float(hit["mean"].iloc[0]) if not hit.empty else np.nan
            f = {k: ("." if not np.isfinite(x) else f"{x:+.2f}") for k, x in got.items()}
            print(f"V {g[:4]:<5} best={klab(t['k'])}|{t['phi']:g} {t['mean']:+.2f} "
                  f"n={int(t['count'])} pureS={f['pure structure']} "
                  f"pureF={f['pure function']} lab00={f['label factorisation']}")
    if "n_pos" in long.columns:
        e = long.drop_duplicates(subset=KEY).groupby("phi")[["n_pos", "n_neg"]].mean()
        print("#edges " + " ".join(f"{p:g}:{int(r.n_pos)}+{int(r.n_neg)}"
                                   for p, r in e.iterrows()))
    gaps = [c for c in ((k, p) for k in kord(long["k"]) for p in sorted(set(long["phi"])))
            if c not in set(cells)]
    print(f"#gaps {len(gaps)}" + ("" if not gaps else " " + " ".join(
        f"{klab(k)}|{p:g}" for k, p in gaps)))
    print("#end")


def summarise(ds, d, args):
    long, meta = load(d)
    if args.classes:
        long = long[long["cls"].isin(args.classes)]
        if long.empty:
            raise SystemExit(f"no rows for classes {args.classes} in {ds}")
    col = "value" if args.raw else "z"
    unit = "raw" if args.raw else "z"
    if col == "z" and not long["z"].notna().any():
        col, unit = "value", "raw (no nulls found)"
    ks, phis = kord(long["k"]), sorted(set(long["phi"]))
    cells = sorted({(k, p) for k, p in zip(long["k"], long["phi"])}, key=lambda c: (c[0] < 0, c))
    if args.compact:
        return compact(ds, long, meta, col, unit.split()[0], cells, args)

    print("=" * 78)
    print(f"=== {ds.upper()}   {len(cells)} cells   unit: {unit}")
    print("=" * 78)

    print(block("STATUS"))
    seeds, classes = sorted(set(long["seed"])), sorted(set(long["cls"]))
    print(f"    trainings {long.groupby(KEY).ngroups}  classes {classes}  seeds {seeds}")
    print(f"    k    {[klab(k) for k in ks]}")
    print(f"    phi  {[f'{p:g}' for p in phis]}")
    if meta:
        v = meta.get("variant", {})
        print(f"    meta {meta.get('epochs')} epochs, {v.get('criterion', '?')} "
              f"q={v.get('q', '?')}, {meta.get('n_perm')} perms/null")
    if "seconds" in long.columns:
        per = long.drop_duplicates(subset=KEY)["seconds"]
        print(f"    time {per.sum() / 3600:.1f} h total, {per.median():.0f} s median/cell")
    gaps = [c for c in ((k, p) for k in ks for p in phis) if c not in set(cells)]
    print(f"    unvisited {len(gaps)} of {len(ks) * len(phis)}: "
          + (", ".join(f"{klab(k)}|{p:g}" for k, p in gaps[:20]) + (" ..." if len(gaps) > 20 else "")
             if gaps else "none"))
    nn = int(long[col].isna().sum())
    if nn:
        print(f"    !! {nn} of {len(long)} readouts are NaN (blank below)")

    print(block("CHECK -- the arms  (rows: geometry x reference)"))
    for arm, xax, sel in ARMS:
        sub = long[[bool(sel(k, p)) for k, p in zip(long["k"], long["phi"])]]
        if sub.empty:
            print(f"    {arm}: not run")
            continue
        order = kord(sub[xax]) if xax == "k" else sorted(set(sub[xax]))
        if len(order) < 2:
            print(f"    {arm}: only one point")
            continue
        print(f"    {arm}")
        print("      geom/ref  " + "".join(
            f"{(klab(v) if xax == 'k' else f'{v:g}'):>8}" for v in order))
        for g in GEOMS:
            for r in REFS:
                m = (sub[(sub.geom == g) & (sub.ref == r)]
                     .groupby(xax)[col].mean().reindex(order))
                print(f"      {GL[g]:<4} {r:<5}" + "".join(fmt(x, 8) for x in m))

    print(block("TARGET -- vs the held-out class  (the result)"))
    for g in GEOMS:
        print(f"    {GL[g]}")
        print(table(pivot(long[(long.geom == g) & (long.ref == "target")], col, ks, phis),
                    ks, phis))

    print(block("POSITION -- vs profile MINUS vs ESM  (+ functional, - structural)"))
    for g in GEOMS:
        row = long[long.geom == g]
        p = {r: pivot(row[row.ref == r], col, ks, phis) for r in ("func", "esm")}
        print(f"    {GL[g]}")
        print(table(p["func"] - p["esm"], ks, phis, nd=1))

    print(block("VERDICT -- best cell vs the two PURE sources"))
    groups = ([(c, long[long.cls == c]) for c in classes] if args.per_class else []) + \
             [("ALL CLASSES" if len(classes) > 1 else classes[0], long)]
    for gname, part in groups:
        print(f"    [{gname}]")
        for g in GEOMS:
            q = (part[(part.geom == g) & (part.ref == "target")]
                 .groupby(["k", "phi"])[col].agg(["mean", "std", "count"]).reset_index()
                 .sort_values("mean", ascending=False))
            if q.empty:
                continue
            t = q.iloc[0]
            err = t["std"] / np.sqrt(t["count"]) if t["count"] > 1 else np.nan
            have = {}
            for c, nm in PURE.items():
                hit = q[(q.k == c[0]) & (q.phi == c[1])]
                if not hit.empty:
                    have[nm.split(" (")[0]] = float(hit["mean"].iloc[0])
            head = (f"      {GL[g]:<4} best {klab(t['k'])}|{t['phi']:g} = {t['mean']:+.2f}"
                    f"{'' if not np.isfinite(err) else f' +/-{err:.2f}'} (n={int(t['count'])})")
            cell = (float(t["k"]), float(t["phi"]))
            if col == "z" and t["mean"] < 2:
                print(f"{head}  -> below its own null; no shape")
            elif len(have) < 2:
                print(f"{head}  -> {2 - len(have)} pure corner(s) not run; cannot compare")
            elif cell in PURE:
                print(f"{head}  -> {PURE[cell]} wins")
            elif cell in LABELS:
                print(f"{head}  -> {LABELS[cell]} wins")
            else:
                det = ", ".join(f"{n} {v:+.2f}" for n, v in have.items())
                print(f"{head}  -> MIX by {t['mean'] - max(have.values()):+.2f}  [{det}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="cc / hc (default: every one found)")
    ap.add_argument("--root", default="results/sf_grid")
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--per-class", action="store_true",
                    help="a verdict per held-out class as well as pooled")
    ap.add_argument("-c", "--compact", action="store_true",
                    help="dense positional format, ~15 lines per dataset, for pasting")
    ap.add_argument("--raw", action="store_true", help="the geometries' own units, not z")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_absolute() and not root.exists():
        root = _root / root      # so the default works from anywhere inside the repo
    found = sorted(p for p in root.glob("*") if (p / "grid.csv").exists()) if root.exists() else []
    if args.datasets:
        want = {d.lower() for d in args.datasets}
        picked = [p for p in found if p.name.lower() in want]
        missing = want - {p.name.lower() for p in picked}
        if missing:
            raise SystemExit(f"no grid.csv for {sorted(missing)} under {root}; "
                             f"found {[p.name for p in found]}")
        found = picked
    if not found:
        raise SystemExit(f"nothing with a grid.csv under {root}")
    for d in found:
        summarise(d.name, d, args)


if __name__ == "__main__":
    main()
