#!/usr/bin/env python
"""The headline table: boosting vs the old graph vs the ESM-free graph, with error bars.

    python scripts/analysis/headline_table.py                 # every run on disk
    python scripts/analysis/headline_table.py -c              # dense, for pasting
    python scripts/analysis/headline_table.py --mol-source chemberta --all-metrics

THREE MODELS, one row each per (dataset, regime, molecule source, edge variant):

  boost      `boost_full` -- XGBoost on [ raw ESM || molecule ]. No graph at all. This
             is the number a graph has to beat, and on several cells it is still the
             ceiling.
  GNN old    `graph_legacy` from the ESM-node run -- the pre-v8 model, ESM in the node
             features, no frozen branch. This is what the paper's graph rows are.
  GNN new    the gate at alpha=1 from the ONE-HOT run -- receptor nodes are an
             identity, the frozen structural branch is at weight zero, so NO protein
             embedding enters the model anywhere. If this ties `GNN old`, the
             structural channel was decorative; if it also beats `boost`, the graph is
             winning without the representation the baseline is built on.

ERROR BARS ARE OVER FOLDS x SEEDS, and both axes are load-bearing. A fold changes which
rows are held out; a seed changes the model's own draw -- graph init, the bag, the
head's subsample/colsample -- and that was measured at +/-0.007 (graph, GPU scatter) to
~0.02 (head lottery), which is the size of the effects here. Five folds at one seed
bound the first source and say nothing about the second, so a row with `seeds=1` carries
an interval that is honest about splits and silent about everything else; the CHECKS
block calls those out by name.

The deltas are PAIRED on (fold, seed): the arms ran on the same split with the same
draw, so their difference removes both nuisances at once and is far stronger evidence
than two overlapping intervals. `won` counts the cells in favour.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

METRIC_SETS = {"regression": ["R2", "RMSE", "MAE", "Pearson", "Spearman"],
               "classification": ["AUROC", "AUPRC", "MCC", "F1"]}
OF_RECORD = {"regression": "R2", "classification": "AUROC"}
# Filename suffixes, peeled from the right in the order `out_path` appends them:
# metrics_{ds}_{family}[_{variant}][_{molsource}][_onehot].csv
KNOWN_NODES = {"onehot"}
KNOWN_MOL = {"chemberta", "gin"}
KNOWN_VARIANT = {"q99greedy", "q0cov"}
REGIME_OF = {"rand": "transductive", "transductive": "transductive",
             "our_inductive": "inductive", "inductive_molecule_v5": "inductive"}
# Which arm, in which node run, plays which model.
MODELS = [("boost", "esm", "boost_full", None),
          ("GNN old", "esm", "graph_legacy", None),
          ("GNN new", "onehot", "gate", 1.0)]


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
    frames = []
    for p in sorted(pathlib.Path(root).glob("metrics_*.csv")):
        ds, family, nodes, mol, variant = parse_name(p.stem)
        if family not in REGIME_OF:
            print(f"  (skipping {p.name}: unrecognised split family {family!r})")
            continue
        df = pd.read_csv(p)
        if "seed" not in df.columns:          # written before seeds existed
            df["seed"] = 42
        df = df.assign(seed=df["seed"].fillna(42).astype(int))
        df["dataset"], df["regime"], df["nodes"] = ds, REGIME_OF[family], nodes
        df["mol_source"] = mol
        df["variant_tag"] = (variant or
                             (df["variant"].dropna().iloc[0]
                              if "variant" in df.columns and df["variant"].notna().any()
                              else ""))
        df["file"] = p.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"no metrics_*.csv under {root}")
    df = pd.concat(frames, ignore_index=True)
    if args.dataset:
        df = df[df.dataset.isin(args.dataset)]
    if args.mol_source:
        df = df[df.mol_source.isin(args.mol_source)]
    if args.regime:
        df = df[df.regime.isin(args.regime)]
    return df


def cells(df, nodes, arm, alpha):
    """One model's cells, indexed by (fold, seed) so arms can be paired."""
    q = df[(df.nodes == nodes) & (df.arm == arm)]
    q = q[q.alpha.isna()] if alpha is None else q[np.isclose(q.alpha.astype(float),
                                                             alpha, equal_nan=False)]
    if "status" in q.columns:
        q = q[~q["status"].astype(str).str.startswith("failed")]
    return q.set_index(["fold", "seed"])


def ci95(v):
    """mean, half-width of the 95% interval, n. Normal approximation -- with 25 cells
    the t correction is under 5% of the width and this is a scoreboard, not a test."""
    v = pd.to_numeric(v, errors="coerce").dropna()
    if v.empty:
        return np.nan, np.nan, 0
    if len(v) == 1:
        return float(v.iloc[0]), np.nan, 1
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), len(v)


def paired(a, b, metric):
    """Per-(fold, seed) difference between two arms run on the same cells."""
    common = a.index.intersection(b.index)
    if not len(common):
        return np.nan, np.nan, 0, 0
    d = (pd.to_numeric(a.loc[common, metric], errors="coerce")
         - pd.to_numeric(b.loc[common, metric], errors="coerce")).dropna()
    if d.empty:
        return np.nan, np.nan, 0, 0
    hw = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return float(d.mean()), hw, int((d > 0).sum()), int(len(d))


def group_key(df):
    return ["dataset", "regime", "mol_source", "variant_tag"]


def build(df, args):
    """One record per (group, model): the mean, its interval, and the paired delta
    against boost measured on the cells the two actually share."""
    out = []
    for gk, g in df.groupby(group_key(df), dropna=False):
        fam = ("classification"
               if any(c in g.columns and g[c].notna().any()
                      for c in METRIC_SETS["classification"]) else "regression")
        metric = args.metric if args.metric in METRIC_SETS[fam] else OF_RECORD[fam]
        base = cells(g, "esm", "boost_full", None)
        # boost is node-independent -- same features, same head, same seed. It is
        # computed in both node runs, which makes the two copies a free consistency
        # check rather than a duplicate to be silently dropped.
        alt = cells(g, "onehot", "boost_full", None)
        for name, nodes, arm, alpha in MODELS:
            c = cells(g, nodes, arm, alpha)
            fallback = ""
            if c.empty and name == "GNN new":
                # alpha=1 with one-hot nodes and graph_legacy with one-hot nodes are
                # the same model up to one global scalar; the sweeps have matched to
                # 0.006-0.008 every time. Use it, but say so.
                alt_new = cells(g, "onehot", "graph_legacy", None)
                if len(alt_new):
                    c, fallback = alt_new, "legacy"
            have = len(c) and metric in c.columns
            m, hw, n = ci95(c[metric]) if have else (np.nan, np.nan, 0)
            ref = base if len(base) else alt
            d, dhw, won, tot = ((np.nan, np.nan, 0, 0)
                                if name == "boost" or not have
                                else paired(c, ref, metric))
            out.append(dict(zip(group_key(df), gk)) | dict(
                model=name, metric=metric, mean=m, hw=hw, n=n,
                seeds=c.index.get_level_values("seed").nunique() if len(c) else 0,
                folds=c.index.get_level_values("fold").nunique() if len(c) else 0,
                delta=d, delta_hw=dhw, won=won, tot=tot, fallback=fallback))
    return pd.DataFrame(out)


def checks(df, tab):
    """Everything that would make a cell of the table a lie."""
    msgs = []
    bad = df[df.get("status", pd.Series("ok", index=df.index))
             .astype(str).str.startswith("failed")]
    for f, n in bad.groupby("file").size().items():
        msgs.append(f"FAILED  {n} cell(s) in {f} -- excluded from every mean")
    thin = tab[(tab.n > 0) & (tab.seeds < 2)]
    if len(thin):
        msgs.append(f"1 SEED  {len(thin)} row(s) rest on a single model draw, so their "
                    f"interval covers splits only: "
                    + ", ".join(sorted({f"{r.dataset}/{r.regime}/{r.mol_source}"
                                        for r in thin.itertuples()}))[:160])
    miss = tab[tab.n == 0]
    for r in miss.itertuples():
        msgs.append(f"ABSENT  {r.dataset}/{r.regime}/{r.mol_source}"
                    f"{'/' + r.variant_tag if r.variant_tag else ''}: no `{r.model}` run")
    for r in tab[tab.fallback != ""].itertuples():
        msgs.append(f"SUBST   {r.dataset}/{r.regime}/{r.mol_source}: `GNN new` read off "
                    f"graph_legacy (one-hot), not gate alpha=1 -- equivalent model, "
                    f"different arm")
    # The one cross-file identity the table depends on: boost does not touch the node
    # features, so its two copies must agree cell for cell.
    for gk, g in df.groupby(group_key(df), dropna=False):
        a, b = cells(g, "esm", "boost_full", None), cells(g, "onehot", "boost_full", None)
        if not len(a) or not len(b):
            continue
        met = OF_RECORD["classification" if "AUROC" in g.columns
                        and g["AUROC"].notna().any() else "regression"]
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        gap = float((pd.to_numeric(a.loc[common, met], errors="coerce")
                     - pd.to_numeric(b.loc[common, met], errors="coerce")).abs().max())
        if gap > 1e-6:
            msgs.append(f"MISMATCH {gk[0]}/{gk[1]}/{gk[2]}: boost differs between the "
                        f"ESM and one-hot files by up to {gap:.4f} on {met} -- it must "
                        f"not, the graph is not in it")
    return msgs


def readable(tab, df, args):
    print("=" * 96)
    print("=== HEADLINE: boost vs the pre-v8 graph vs the graph with NO protein "
          "embedding anywhere")
    print("=" * 96)
    hdr = (f"  {'dataset':<9}{'regime':<13}{'mol':<11}{'variant':<11}{'model':<9}"
           f"{'metric':<8}{'value':>17}{'n':>5}{'f x s':>8}{'d vs boost':>18}{'won':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    last = None
    for gk, g in tab.groupby(group_key(tab), dropna=False):
        if last is not None:
            print()
        last = gk
        for i, r in enumerate(g.itertuples()):
            lead = ([f"{r.dataset:<9}", f"{r.regime:<13}", f"{r.mol_source:<11}",
                     f"{(r.variant_tag or '-'):<11}"] if i == 0
                    else [" " * 9, " " * 13, " " * 11, " " * 11])
            val = ("      (not run)   " if r.n == 0 else
                   f"{r.mean:>9.3f} +/-{r.hw:>5.3f}" if np.isfinite(r.hw)
                   else f"{r.mean:>9.3f}       ")
            dlt = ("" if not np.isfinite(r.delta) else
                   f"{r.delta:>+10.3f} +/-{r.delta_hw:5.3f}" if np.isfinite(r.delta_hw)
                   else f"{r.delta:>+10.3f}      ")
            won = "" if not r.tot else f"{r.won}/{r.tot}"
            fx = "" if r.n == 0 else f"{r.folds}x{r.seeds}"
            print("  " + "".join(lead) + f"{r.model + (' *' if r.fallback else ''):<9}"
                  f"{r.metric:<8}{val}{r.n:>5}{fx:>8}{dlt:>18}{won:>8}")
    print("\n  CHECKS")
    print("  " + "-" * 8)
    msgs = checks(df, tab)
    for m in msgs or ["all clear: every row has both seeds and folds, no failed cells, "
                      "boost agrees across node runs"]:
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
    for m in checks(df, tab):
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
