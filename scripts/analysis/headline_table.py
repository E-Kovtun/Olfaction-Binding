#!/usr/bin/env python
"""The headline table: boosting vs the old graph vs the ESM-free graph, with error bars.

    python scripts/analysis/headline_table.py                 # every run on disk
    python scripts/analysis/headline_table.py -c              # dense, for pasting
    python scripts/analysis/headline_table.py --mol-source chemberta --all-metrics

THE COLUMNS, one row per (dataset, regime, molecule source, edge variant):

Receptor nodes are ONE-HOT throughout the v8 grid, so the protein embedding reaches the
model only through the gate's frozen structural branch at weight (1 - alpha). That makes
alpha itself the protein-embedding axis, and the table the two ends of one dial rather
than a comparison of two different graphs:

  boost      XGBoost on [ raw ESM || molecule ]. No graph at all. The number a graph
             has to beat, and on some cells still the ceiling.
  alpha=0    the receptor vector IS the frozen rank-k rotation of ESM; the graph is
             multiplied by zero. Structure alone. It reads the same information as
             `boost` through a different reader, so the two should nearly agree -- the
             CHECKS block turns that into an automatic test of the anchor.
  alpha=1    the graph alone over one-hot receptors. No protein embedding enters the
             model anywhere. Function alone.

`--at-alpha` adds columns in between; `--legacy` adds the pre-v8 arm (alpha=None), which
is alpha=1 up to one global scalar inside the forward pass -- invisible to every
geometry readout and to the tree head, but not to the gradient.

ERROR BARS ARE OVER THE SPLITS. Each row rests on the dataset's own five splits -- folds
1-5 upstream, or the cold-molecule seeds 42-46 on m2or/inductive -- at a fixed model
seed, which is the convention every reported number was produced with. The model draw
(graph init, the bag, the head's subsample/colsample) is deliberately NOT an axis here:
pooling it would widen the bar honestly but make a 25-cell row incomparable, line for
line, with the five-cell series already in the paper. `--all-seeds` opts into the wider
bar when that is what is wanted.

The deltas are PAIRED on (fold, seed): the arms ran on the same split with the same
draw, so their difference removes both nuisances at once and is far stronger evidence
than two overlapping intervals. `won` counts the cells in favour.
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

# Column names come from the module that emits them, so a reader can never drift
# from the writer. See `orbind.dataset.regression_metrics_full`.
from orbind.dataset import METRIC_NAMES                         # noqa: E402
METRIC_SETS = METRIC_NAMES
OF_RECORD = {"regression": "R2", "classification": "AUROC"}
# Filename suffixes, peeled from the right in the order `out_path` appends them:
# metrics_{ds}_{family}[_{variant}][_{molsource}][_onehot].csv
# `nodedial` is the v9 run: its receptor node features ARE the dial, so it belongs to
# the same filename slot as `onehot`. Leaving it out is not a cosmetic miss -- an
# unrecognised tag makes the family unparseable and the whole file is dropped with one
# counted line, which is how a finished sweep reads as an empty directory.
KNOWN_NODES = {"onehot", "nodedial"}
KNOWN_MOL = {"chemberta", "gin", "ecfp"}
KNOWN_VARIANT = {"q99greedy", "q0cov"}
REGIME_OF = {"rand": "transductive", "transductive": "transductive",
             "our_inductive": "inductive", "inductive_molecule_v5": "inductive"}
# The edge variant each dataset's reported numbers stand on. m2or has two live variants
# and only q99greedy is of record, so the other is hidden unless asked for -- hidden,
# not deleted: q0cov is the evidence that what breaks m2or transductive is the EDGE SET
# and not the protein embedding.
CANONICAL_VARIANT = {"cc": "q0cov", "hc": "q0cov", "m2or": "q99greedy"}
# Every reported number was produced at seed 42, the ensembler's own default under
# ofm/full_full. Extra seeds widen the interval honestly, but a 25-cell row is not
# comparable line for line with a 5-cell one from an older series, and mixing the two
# in one table is what made it unreadable. Default to the convention; --all-seeds
# opts into the wider bar.
DEFAULT_SEED = 42
# The three models, as (label, arm, alpha). Under the fully separated design every run
# has ONE-HOT receptor nodes, so ESM reaches the model only through the gate's frozen
# branch at weight (1 - alpha) -- which makes alpha itself the protein-embedding axis
# and turns the table into the two ends of one dial rather than two different graphs:
#
#   alpha=0   the receptor vector IS the frozen ESM projection, the graph is multiplied
#             by zero. Structure alone. It should land on `boost`, which is the same
#             information through a different reader -- and that agreement is the
#             table's built-in sanity check.
#   alpha=1   the graph alone, over one-hot receptors. No protein embedding enters the
#             model anywhere. Function alone.
MODELS = [("boost", "boost_full", None),
          ("alpha=0", "gate", 0.0),
          ("alpha=1", "gate", 1.0)]
LEGACY_MODEL = ("legacy", "graph_legacy", None)


def parse_name(stem):
    """(dataset, family, nodes, mol_source, variant) from a metrics filename. Absent
    tags mean the defaults the sweep used before that axis existed: ESM nodes, GIN
    molecules, and the dataset's own canonical edge variant."""
    parts = stem.replace("metrics_", "").split("_")
    ds, rest = parts[0], parts[1:]
    nodes, mol, variant = "esm", "gin", ""
    if rest and rest[-1] in KNOWN_NODES:
        nodes = rest.pop()
    if rest and rest[-1] in KNOWN_MOL:
        mol = rest.pop()
    if rest and rest[-1] in KNOWN_VARIANT:
        variant = rest.pop()
    return ds, "_".join(rest), nodes, mol, variant


def load(root, args):
    """Every metrics CSV as one long frame, tagged with what its filename says."""
    frames, skipped = [], []
    for p in sorted(pathlib.Path(root).glob("metrics_*.csv")):
        ds, family, nodes, mol, variant = parse_name(p.stem)
        if family not in REGIME_OF:
            skipped.append(p.name)
            continue
        df = pd.read_csv(p)
        if "seed" not in df.columns:          # written before seeds existed
            df["seed"] = 42
        vtag = variant or (df["variant"].dropna().iloc[0]
                           if "variant" in df.columns and df["variant"].notna().any()
                           else "")
        df = df.assign(seed=df["seed"].fillna(42).astype(int), dataset=ds,
                       regime=REGIME_OF[family], nodes=nodes, mol_source=mol,
                       variant_tag=vtag, file=p.name)
        frames.append(df)
    if skipped:
        print(f"  ({len(skipped)} file(s) from no recognised split family, ignored)")
    if not frames:
        raise SystemExit(f"no metrics_*.csv under {root}")
    df = pd.concat(frames, ignore_index=True)
    df.loc[:, "variant_tag"] = df["variant_tag"].fillna("").astype(str)
    # A blank variant means "written before the column existed", which is always the
    # dataset's own default -- not a second, nameless variant.
    df.loc[:, "variant_tag"] = [v or CANONICAL_VARIANT.get(d, "")
                                for d, v in zip(df.dataset, df.variant_tag)]
    if args.dataset:
        df = df[df.dataset.isin(args.dataset)]
    if args.mol_source:
        df = df[df.mol_source.isin(args.mol_source)]
    if args.regime:
        df = df[df.regime.isin(args.regime)]
    if getattr(args, "nodes", None):
        df = df[df.nodes.isin(args.nodes)]
    if getattr(args, "variant", None):
        df = df[df.variant_tag.isin(args.variant)]
    elif not getattr(args, "all_variants", False):
        df = df[[v == CANONICAL_VARIANT.get(d, v)
                 for d, v in zip(df.dataset, df.variant_tag)]]
    if not getattr(args, "all_seeds", False):
        seed = int(getattr(args, "seed", DEFAULT_SEED) or DEFAULT_SEED)
        df = df[df.seed == seed]
        if df.empty:
            raise SystemExit(f"nothing left at seed {seed} -- pass --all-seeds, or "
                             f"--seed with one that is present")
    return df


def cells(df, arm, alpha, nodes=None):
    """One model's cells, indexed by (fold, seed) so arms can be paired."""
    q = df if nodes is None else df[df.nodes == nodes]
    q = q[q.arm == arm]
    q = q[q.alpha.isna()] if alpha is None else q[np.isclose(q.alpha.astype(float),
                                                             alpha, equal_nan=False)]
    if "status" in q.columns:
        q = q[~q["status"].astype(str).str.startswith("failed")]
    return q.set_index(["fold", "seed"])


def _t(n, level=0.95):
    """The two-sided critical value at n-1 degrees of freedom.

    NOT 1.96. That approximation was written when a row pooled 25 cells; under the
    fully separated grid a row rests on the dataset's five splits at a fixed model
    seed, where t is 2.776 and the normal interval is 29% too narrow. An error bar
    that narrow turns "indistinguishable" into "significant" on exactly the 0.01-scale
    differences this table exists to adjudicate."""
    from scipy.stats import t as _tdist
    return float(_tdist.ppf(0.5 + level / 2, max(n - 1, 1)))


def ci95(v):
    """mean, half-width of the 95% interval, n. Student-t -- see `_t`."""
    v = pd.to_numeric(v, errors="coerce").dropna()
    if v.empty:
        return np.nan, np.nan, 0
    if len(v) == 1:
        return float(v.iloc[0]), np.nan, 1
    return (float(v.mean()),
            float(_t(len(v)) * v.std(ddof=1) / np.sqrt(len(v))), len(v))


def paired(a, b, metric):
    """Per-(fold, seed) difference between two arms run on the same cells."""
    common = a.index.intersection(b.index)
    if not len(common):
        return np.nan, np.nan, 0, 0
    d = (pd.to_numeric(a.loc[common, metric], errors="coerce")
         - pd.to_numeric(b.loc[common, metric], errors="coerce")).dropna()
    if d.empty:
        return np.nan, np.nan, 0, 0
    hw = _t(len(d)) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return float(d.mean()), hw, int((d > 0).sum()), int(len(d))


def group_key(df):
    return ["dataset", "regime", "mol_source", "variant_tag"]


def models_for(args):
    """The columns of the table: the two ends of the dial, plus whatever the caller
    asked to see between them."""
    out = list(MODELS)
    for a in (getattr(args, "at_alpha", None) or []):
        if not any(np.isclose(float(a), x) for _, _, x in out if x is not None):
            out.insert(-1, (f"alpha={float(a):g}", "gate", float(a)))
    if getattr(args, "legacy", False):
        out.append(LEGACY_MODEL)
    return out


def build(df, args):
    """One record per (group, model): the mean, its interval, and the paired delta
    against boost measured on the cells the two actually share."""
    out, models = [], models_for(args)
    for gk, g in df.groupby(group_key(df), dropna=False):
        fam = ("classification"
               if any(c in g.columns and g[c].notna().any()
                      for c in METRIC_SETS["classification"]) else "regression")
        metric = args.metric if args.metric in METRIC_SETS[fam] else OF_RECORD[fam]
        base = cells(g, "boost_full", None)
        for name, arm, alpha in models:
            c = cells(g, arm, alpha)
            have = len(c) and metric in c.columns
            m, hw, n = ci95(c[metric]) if have else (np.nan, np.nan, 0)
            d, dhw, won, tot = ((np.nan, np.nan, 0, 0) if name == "boost" or not have
                                else paired(c, base, metric))
            out.append(dict(zip(group_key(df), gk)) | dict(
                model=name, metric=metric, mean=m, hw=hw, n=n,
                seeds=c.index.get_level_values("seed").nunique() if len(c) else 0,
                folds=c.index.get_level_values("fold").nunique() if len(c) else 0,
                delta=d, delta_hw=dhw, won=won, tot=tot, fallback=""))
    return pd.DataFrame(out)


def checks(df, tab, args=None):
    """Everything that would make a cell of the table a lie."""
    msgs = []
    bad = df[df.get("status", pd.Series("ok", index=df.index))
             .astype(str).str.startswith("failed")]
    for f, n in bad.groupby("file").size().items():
        msgs.append(f"FAILED  {n} cell(s) in {f} -- excluded from every mean")
    thin = tab[(tab.n > 0) & (tab.seeds < 2)]
    if len(thin) and getattr(args, "all_seeds", False):
        msgs.append(f"1 SEED  {len(thin)} row(s) rest on a single model draw, so their "
                    f"interval covers splits only: "
                    + ", ".join(sorted({f"{r.dataset}/{r.regime}/{r.mol_source}"
                                        for r in thin.itertuples()}))[:160])
    miss = tab[tab.n == 0]
    for r in miss.itertuples():
        msgs.append(f"ABSENT  {r.dataset}/{r.regime}/{r.mol_source}"
                    f"{'/' + r.variant_tag if r.variant_tag else ''}: no `{r.model}` run")
    # The invariant that replaces the old cross-node boost check. At alpha=0 the
    # receptor vector is a frozen rank-k rotation of the very ESM matrix `boost` reads
    # raw, so the two see the same information through different readers and must land
    # close. A wide gap means the anchor is not what it claims -- the wrong rank, a
    # centred projection, or one-hot vectors having reached it by mistake.
    for gk, g in tab.groupby(group_key(tab), dropna=False):
        b_ = g[g.model == "boost"]
        z_ = g[g.model == "alpha=0"]
        if not len(b_) or not len(z_) or not np.isfinite(z_.iloc[0].delta):
            continue
        d = abs(float(z_.iloc[0].delta))
        if d > 0.05:
            msgs.append(f"ANCHOR  {gk[0]}/{gk[1]}/{gk[2]}: alpha=0 is {d:.3f} from boost "
                        f"on {z_.iloc[0].metric} -- at alpha=0 the receptor vector is a "
                        f"frozen rotation of the same ESM, so they should nearly agree")
    return msgs


ORDER_REGIME = {"transductive": 0, "inductive": 1}
ORDER_MOL = {"chemberta": 0, "gin": 1}
ID_COLS = [("dataset", 8), ("regime", 14), ("mol_source", 11), ("variant_tag", 11)]
VW, DW = 15, 23          # width of a value cell and of a delta cell


def _val(m, hw, w=VW):
    """A model's score. `--` rather than a blank, so a missing arm reads as missing
    rather than as a formatting slip."""
    if not np.isfinite(m):
        return f"{'--':^{w}}"
    return (f"{m:.3f} +/-{hw:.3f}" if np.isfinite(hw) else f"{m:.3f}").rjust(w)


def _dlt(d, hw, won, tot, w=DW):
    """The paired difference against boost, and the cells in favour. Paired on
    (fold, seed): the arms ran on the same split with the same draw, so the difference
    removes both nuisances at once and is far sharper than two overlapping intervals."""
    if not np.isfinite(d):
        return " " * w
    body = f"{d:+.3f}" + (f" +/-{hw:.3f}" if np.isfinite(hw) else "")
    return f"{body:>{w - 6}}{f'{won}/{tot}':>6}"


def _shown(tab):
    """Which identity columns earn a column of their own.

    A column that never varies WITHIN a dataset -- m2or's edge variant, say -- is a
    property of that dataset rather than a dimension of the study, so printing it on
    every row is noise. It moves to a note under the title instead."""
    show, folded = [], []
    for c, w in ID_COLS:
        vals = tab[c].astype(str)
        if c == "dataset":                      # the row's identity, never folded
            show.append((c, w))
        elif vals.nunique() <= 1:
            if vals.iloc[0]:
                folded.append(f"{c.split('_')[0]} {vals.iloc[0]}")
        elif tab.groupby("dataset")[c].nunique().max() > 1:
            show.append((c, w))
        else:
            folded += [f"{d} {v}" for d, v in
                       sorted({(r.dataset, str(getattr(r, c))) for r in tab.itertuples()})
                       if v]
    return show, folded


def readable(tab, df, args):
    """One row per cell of the study, the three models side by side.

    The earlier shape -- three rows per cell, the identity columns blank on two of
    them, a blank line between every group -- made a fourteen-cell study read as
    fourteen separate tables. Everything a reader compares (boost against each graph,
    and the two graphs against each other) now sits on one line."""
    piv, keys = {}, []
    for r in tab.itertuples():
        k = tuple(getattr(r, c) for c, _ in ID_COLS)
        if k not in piv:
            piv[k], _ = {}, keys.append(k)
        piv[k][r.model] = r
    keys.sort(key=lambda k: (k[0], ORDER_REGIME.get(k[1], 9),
                             ORDER_MOL.get(k[2], 9), k[3]))
    show, folded = _shown(tab)
    idw = sum(w for _, w in show) + 6 + 4
    width = idw + VW + 2 + (len(models_for(args)) - 1) * (VW + DW + 2)

    seed_note = ("all seeds pooled" if getattr(args, "all_seeds", False)
                 else f"seed {getattr(args, 'seed', DEFAULT_SEED)} only")
    print("=" * width)
    print("=== HEADLINE -- what the receptor representation is actually worth")
    print("===   Receptor nodes are ONE-HOT throughout, so ESM reaches the model only")
    print("===   through the gate's frozen branch at weight (1 - alpha): alpha IS the")
    print("===   protein-embedding axis, and the columns are the two ends of one dial.")
    print("===     boost    XGBoost on [ raw ESM || molecule ]. No graph at all.")
    print("===     alpha=0  the frozen ESM projection, graph multiplied by zero.")
    print("===     alpha=1  the graph alone. No protein embedding anywhere.")
    print(f"===   {seed_note}; +/- is the 95% interval over cells; deltas paired per cell."
          + (f"  [{'; '.join(folded)}]" if folded else ""))
    print("=" * width)

    labels = [m for m, *_ in models_for(args)]
    print(" " * idw + f"  {labels[0]:^{VW}}"
          + "".join(f"  {lb:^{VW + DW}}" for lb in labels[1:]))
    hdr = ("  " + "".join(f"{c.split('_')[0]:<{w}}" for c, w in show)
           + f"{'metric':<6}{'n':>4}"
           + f"  {'value':>{VW}}"
           + "".join(f"  {'value':>{VW}}{'d vs boost':>{DW}}" for _ in labels[1:]))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    prev_ds = None
    for k in keys:
        g = piv[k]
        named = dict(zip([c for c, _ in ID_COLS], k))
        if prev_ds is not None and k[0] != prev_ds:
            print()
        prev_ds = k[0]
        rows = [g.get(m) for m, *_ in models_for(args)]
        met = next((r.metric for r in rows if r is not None), "")
        cnt = max((r.n for r in rows if r is not None), default=0)
        line = ("  " + "".join(f"{str(named[c]) or '-':<{w}}" for c, w in show)
                + f"{met:<6}{cnt:>4}")
        b = rows[0]
        line += "  " + (_val(b.mean, b.hw) if b is not None else " " * VW)
        for r in rows[1:]:
            line += "  " + (_val(r.mean, r.hw) + _dlt(r.delta, r.delta_hw, r.won, r.tot)
                            if r is not None else " " * (VW + DW))
        print(line)

    msgs = checks(df, tab, args)
    print("\n  CHECKS")
    print("  " + "-" * 8)
    for m in msgs or ["all clear: no failed cells, every model present, and alpha=0 "
                      "lands on boost as the frozen anchor requires"]:
        print(f"    {m}")


def compact(tab, df, args):
    print(f"#HEAD1 rows={len(tab)} groups={tab.groupby(group_key(tab)).ngroups}")
    print("#cols dataset regime mol variant model metric mean hw n folds seeds "
          "dBoost dhw won tot")
    for r in tab.itertuples():
        f = lambda v, nd=3: "." if not np.isfinite(v) else f"{v:.{nd}f}"  # noqa: E731
        print(f"{r.dataset} {r.regime[:5]} {r.mol_source} {r.variant_tag or '.'} "
              f"{r.model.replace(' ', '-')} {r.metric} {f(r.mean)} {f(r.hw)} {r.n} "
              f"{r.folds} {r.seeds} {f(r.delta)} {f(r.delta_hw)} {r.won} {r.tot}")
    for m in checks(df, tab, args):
        print(f"#chk {m}")
    print("#end")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/graph/v8_alpha_gate")
    ap.add_argument("--dataset", nargs="+", default=None)
    ap.add_argument("--regime", nargs="+", default=None,
                    choices=["transductive", "inductive"])
    ap.add_argument("--mol-source", nargs="+", default=None,
                    help="restrict to these molecule sources (chemberta, gin)")
    ap.add_argument("--at-alpha", type=float, nargs="+", default=None,
                    help="extra gate columns between the two ends, e.g. --at-alpha 0.5")
    ap.add_argument("--legacy", action="store_true",
                    help="also show `graph_legacy` (alpha=None). It is alpha=1 up to one "
                         "global scalar applied inside the forward pass -- invisible to "
                         "every geometry readout and to the tree head, but not to the "
                         "gradient, so the two are equal only up to training noise")
    ap.add_argument("--nodes", nargs="+", default=None, choices=["esm", "onehot"],
                    help="node features to include. The v8 grid is one-hot throughout; "
                         "pass `esm` only to read an archived pre-separation run")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="show only this seed. Default 42 -- the seed every reported "
                         "number was produced at, which keeps a five-seed series and a "
                         "one-seed one on the same footing (5 cells each)")
    ap.add_argument("--all-seeds", action="store_true",
                    help="pool every seed instead: wider evidence per row, but rows "
                         "with different seed counts stop being comparable line for line")
    ap.add_argument("--variant", nargs="+", default=None,
                    help="edge variants to show. Default: each dataset's canonical one "
                         "(q99greedy on m2or)")
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--metric", default=None,
                    help="metric to rank on; default R2 for regression runs, AUROC for "
                         "classification, per group")
    ap.add_argument("--all-metrics", action="store_true",
                    help="one table per metric of the group's family")
    ap.add_argument("-c", "--compact", action="store_true")
    ap.add_argument("--csv", default=None, help="also write the table here")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_absolute() and not root.exists():
        root = _root / root
    df = load(root, args)

    fams = sorted({m for f in METRIC_SETS.values() for m in f}) if args.all_metrics else [None]
    tabs = []
    for m in ([args.metric] if not args.all_metrics else fams):
        a = argparse.Namespace(**vars(args)); a.metric = m
        t = build(df, a)
        if args.all_metrics and m is not None:
            t = t[t.metric == m]
            if t.empty:
                continue
        tabs.append(t)
        (compact if args.compact else readable)(t, df, a)
    if args.csv:
        pd.concat(tabs).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
