#!/usr/bin/env python
"""The protein-representation table: what the receptor side has to be for the head to work.

Rows, in the order the argument needs them: our graph first (the refined receptor,
boosted as `cls+mol`), then real protein language models, then the classical
amino-acid floor, then the controls -- one-hot identity, one-hot without a molecule,
molecule alone. One panel per (dataset, regime), and by default ONE column: the
metric of record. Six panels times a four-metric battery is a table nobody reads
across, and the battery is in the long CSV either way (`--which headline|all`).

The claim the table is built to support: our method makes a receptor representation
ADAPTED TO BINDING, while a pretrained protein embedding -- however large the model
that produced it -- does not solve this task by itself. The one-hot control is what
keeps that honest: a receptor vector that only encodes identity is the floor our
refinement has to clear, and the pLMs have to clear it too.

READ-ONLY. Everything is fitted by `scripts/modeling/analysis/prot_floor_sweep.py`,
which writes one CSV per (dataset, regime) under results/tables/. Our rows come from
the SAME script and the SAME folds -- `--gnn esm3@1 esm3@0 prott5@1` -- and that is
deliberate: the refined receptor is trained on its fold's train pairs, so a vector
imported from another table's run would be a leak dressed up as a shortcut. If those
rows are absent the table still prints, and the run says what to add.

    .venv/bin/python scripts/article_tables/s2_protein_sources.py \\
        --dataset m2or --regime transductive inductive

Writes results/article_tables/protein_sources/: protein_long.csv, protein_sources.tex,
and a printed text block.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tablekit as tk  # noqa: E402

sys.path.insert(0, str(tk.ROOT))
from scripts.analysis import alpha_grid as ag  # noqa: E402

#: Which family a row belongs to, and in which order the families are printed.
#: A row whose name matches nothing here is still shown, in "other" -- an unknown
#: descriptor is a thing to look at, not a thing to drop.
FAMILY_ORDER = ["ours", "plm", "classical", "control", "other"]
FAMILY_LABEL = {"ours": "Our graph (cls+mol)", "plm": "Protein language models",
                "classical": "Classical amino-acid floor", "control": "Controls",
                "other": "Other"}
PLM = {"esm1b", "prott5", "esm3", "esmc"}
CLASSICAL = {"aac", "kmer2", "aaindex", "ctd", "pseaac", "blosum"}
CONTROL = {"onehot", "onehot_only", "mol_only"}
#: How a row is printed. Everything else keeps the name the sweep wrote.
LABEL = {"onehot": "one-hot receptor", "onehot_only": "one-hot, no molecule",
         "mol_only": "molecule only", "aac": "AAC", "kmer2": "k-mer (2)",
         "aaindex": "AAindex", "ctd": "CTD", "pseaac": "PseAAC", "blosum": "BLOSUM",
         "esm1b": "ESM-1b", "prott5": "ProtT5", "esm3": "ESM3"}


def family(name):
    if name.startswith("GNN["):
        return "ours"
    if name in PLM:
        return "plm"
    if name in CLASSICAL:
        return "classical"
    if name in CONTROL:
        return "control"
    return "other"


def label(name):
    """`GNN[esm3]@a1` -> `ours, ESM3 nodes`; everything else, its pretty name."""
    if name.startswith("GNN["):
        src, _, a = name[4:].partition("]@a")
        pretty = LABEL.get(src, src)
        return (f"ours, {pretty} nodes" if float(a) == 1.0 else
                f"ours, identity nodes (alpha={float(a):g})" if float(a) == 0.0 else
                f"ours, {pretty} nodes at alpha={float(a):g}")
    return LABEL.get(name, name)


# ------------------------------------------------------------------ loading

def load(root, dataset, regime):
    """One cell's CSV, or None. Missing is normal while a panel is still running."""
    p = tk.resolve(f"{root}/prot_floor_{dataset}_{regime}.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.attrs["path"] = str(p)
    return df


def metrics_of(df, dataset, which):
    """The metric columns this cell carries, in the repository's canonical order.

    `which="primary"` is the default and the one the paper's table uses: the metric of
    record alone (AUROC on the binary pool, R2 on the continuous panels). Six panels
    times four columns is a table nobody reads across; the rest of the battery lives in
    the long CSV, where deciding later is free.

    `alpha_grid.metrics_available` selects on a `dataset` COLUMN, which a prot_floor
    CSV does not have -- the dataset is in its filename. Adding it here keeps one
    definition of "which metrics does this task emit" instead of a second copy.
    """
    if which == "primary":
        m = ag.OF_RECORD[ag.TASK[dataset]]
        return [m] if m in df.columns else []
    want = ag.metrics_available(df.assign(dataset=dataset), dataset=dataset,
                                which=which)
    return [m for m in want if m in df.columns]


def cell_table(df, dataset, metrics, level=0.95):
    """One row per protein representation: mean +- t half-width over FOLDS.

    Seeds are averaged inside each fold first -- both the boost seed and, for our
    rows, the graph seed. The unit of evidence is the held-out split, and a row that
    counted its seeds as observations would carry an interval about half as wide as
    it has earned.
    """
    out = []
    for name, g in df.groupby("prot", sort=False):
        rec = dict(prot=name, family=family(name), label=label(name),
                   pdim=int(pd.to_numeric(g["pdim"], errors="coerce").max()))
        for m in metrics:
            fold_means = g.groupby("fold")[m].mean()
            mu, hw, n = ag.ci(fold_means, level)
            rec[m] = mu
            rec[f"{m}_hw"] = hw
            rec[f"{m}_n"] = n
        out.append(rec)
    t = pd.DataFrame(out)
    t = t.assign(_order=t["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}))
    return t.sort_values(["_order", metrics[0]],
                         ascending=[True, metrics[0] in ag.LOWER_IS_BETTER]
                         ).drop(columns="_order").reset_index(drop=True)


def best_of(t, metric):
    """The best row on this metric, direction-aware. RMSE and MAE win by being small,
    and a table that bolded their maximum would advertise the worst row."""
    v = pd.to_numeric(t[metric], errors="coerce")
    if not v.notna().any():
        return None
    return int(v.idxmin() if metric in ag.LOWER_IS_BETTER else v.idxmax())


# ------------------------------------------------------------------ the wide shape

def combine(cells, metric_of):
    """Every cell side by side: one row per representation, one column per cell.

    `cells` is [(dataset, regime, per-cell table)]. The join is OUTER on the
    representation name -- a row that only one panel has (a pLM whose npz covers one
    dataset and not another) stays visible with `--` elsewhere, because "not fitted
    here" and "fitted and bad" are different statements.

    Row order: family first (ours, pLMs, the classical floor, the controls), then the
    MEAN RANK across the cells that have the row. Sorting by a value would mean
    sorting by whichever panel happens to come first, and AUROC and R2 are not
    comparable anyway; ranks are.
    """
    wide, families, labels, dims, ranks = {}, {}, {}, {}, {}
    for ds, reg, t in cells:
        m = metric_of[ds]
        col = (ds, reg, m)
        order = t[m].rank(ascending=m in ag.LOWER_IS_BETTER, method="average")
        for i, r in t.iterrows():
            name = r["prot"]
            wide.setdefault(name, {})[col] = (r[m], r[f"{m}_hw"])
            families[name] = r["family"]
            labels[name] = r["label"]
            dims.setdefault(name, r["pdim"])
            ranks.setdefault(name, []).append(float(order.loc[i]))
    cols = [(ds, reg, metric_of[ds]) for ds, reg, _ in cells]
    rows = sorted(wide, key=lambda n: (FAMILY_ORDER.index(families[n]),
                                       float(np.mean(ranks[n]))))
    return rows, cols, wide, families, labels, dims, ranks


def best_in_column(rows, col, wide):
    """The winning row of one column, direction-aware -- RMSE and MAE win by being
    small, and a table that bolded their maximum would advertise the worst row."""
    have = [(wide[n][col][0], n) for n in rows
            if col in wide[n] and np.isfinite(wide[n][col][0])]
    if not have:
        return None
    return (min if col[2] in ag.LOWER_IS_BETTER else max)(have)[1]


def fmt(cellv, nd=3):
    if cellv is None:
        return "--"
    v, hw = cellv
    if not np.isfinite(v):
        return "--"
    return f"{v:.{nd}f}" + ("" if not np.isfinite(hw) else f"+/-{hw:.{nd}f}")


# ------------------------------------------------------------------ rendering

#: Compact names for a header that has to fit six times across a terminal.
SHORT_DS = {"m2or": "M2OR", "cc": "Carey", "hc": "Hallem"}
SHORT_REG = {"transductive": "trans", "inductive": "cold mol",
             "cold_receptor": "cold rec"}


def head_label(ds, reg, m):
    """(top line, second line) for one column: what the panel is, and its metric."""
    return (f"{SHORT_DS.get(ds, ds)}/{SHORT_REG.get(reg, reg)}", m)


def text(rows, cols, wide, families, labels, dims, ranks):
    w = max(len(labels[n]) for n in rows) + 2
    heads = [head_label(*c) for c in cols]
    cw = 18
    pre = f"{'representation':<{w}}{'dim':>6}{'rank':>7}  "
    line = pre + "".join(f"{top:>{cw}}" for top, _ in heads)
    # two header rows: the panel above, its metric below. One row would either
    # repeat 'AUROC' six times or need columns nobody can line up by eye.
    out = ["", "=" * len(line), line,
           " " * len(pre) + "".join(f"{m:>{cw}}" for _, m in heads),
           "-" * len(line)]
    best = {c: best_in_column(rows, c, wide) for c in cols}
    last = None
    for n in rows:
        if families[n] != last:
            out.append(f"-- {FAMILY_LABEL[families[n]]}")
            last = families[n]
        cells = "".join(f"{fmt(wide[n].get(c)) + ('*' if best[c] == n else ' '):>{cw}}"
                        for c in cols)
        out.append(f"{labels[n]:<{w}}{dims[n]:>6}{np.mean(ranks[n]):>7.1f}  {cells}")
    out += ["", "* = best in column. mean +/- 95% CI over folds; seeds (boost and "
            "graph) averaged inside each fold first.",
            "rank = mean place within a column, averaged over the columns the row "
            "appears in (smaller is better)."]
    return "\n".join(out)


def latex(rows, cols, wide, families, labels, dims, ranks):
    best = {c: best_in_column(rows, c, wide) for c in cols}
    body, last = [], None
    for n in rows:
        if families[n] != last:
            if last is not None:
                body.append(r"\addlinespace")
            body.append(rf"\multicolumn{{{len(cols) + 3}}}{{@{{}}l}}"
                        rf"{{\emph{{{FAMILY_LABEL[families[n]]}}}}} \\")
            last = families[n]
        cells = []
        for c in cols:
            v = fmt(wide[n].get(c)).replace("+/-", r"$\pm$")
            cells.append(rf"\textbf{{{v}}}" if best[c] == n else v)
        body.append(f"{labels[n]} & {dims[n]} & {np.mean(ranks[n]):.1f} & "
                    + " & ".join(cells) + r" \\")
    heads = " & ".join(rf"\textbf{{{tk.DATASET_LABEL.get(ds, ds)}}}" for ds, _, _ in cols)
    sub = " & ".join(rf"{reg.replace('_', ' ')} ({m})" for _, reg, m in cols)
    return "\n".join([
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{\textbf{What the receptor side has to be.} The same boosting head "
        r"over $[\text{receptor representation}\,\|\,\text{ChemBERTa}]$ in every "
        r"column; only the receptor block changes. \emph{Our graph} rows are the "
        r"refined receptor vector beside the molecule --- our \texttt{cls+mol} --- "
        r"trained inside each column's own folds rather than imported, since a "
        r"representation fitted on one fold's training pairs is not a fixed feature "
        r"elsewhere. \emph{identity nodes} is the node dial at zero: the graph is told "
        r"who the receptor is and nothing about its sequence. Each column reports that "
        r"panel's metric of record, 5 held-out splits, mean $\pm$ 95\% CI over splits, "
        r"seeds averaged inside each split first. \emph{rank} is the mean place within "
        r"a column, averaged over columns. \textbf{Bold} = best in column, "
        r"\texttt{--} = not fitted in that cell.}",
        r"\label{tab:protsrc}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrr" + "r" * len(cols) + r"@{}}", r"\toprule",
        r"\textbf{Receptor representation} & \textbf{dim} & \textbf{rank} & "
        + heads + r" \\",
        r" & & & " + sub + r" \\", r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


# ------------------------------------------------------------------ driver

def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["m2or"],
                    choices=["m2or", "cc", "hc"])
    ap.add_argument("--regime", nargs="+", default=["transductive", "inductive"])
    ap.add_argument("--which", default="primary",
                    choices=["primary", "headline", "all"],
                    help="primary = the metric of record only, which is what the "
                         "paper's table shows; the full battery is in the long CSV")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--root", default="results/tables",
                    help="where prot_floor_sweep.py wrote its CSVs")
    ap.add_argument("--out", default="results/article_tables/protein_sources")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    out = tk.out_dir(a.out)
    longs, panels, missing, no_gnn = [], [], [], []
    metric_of = {}
    for ds in a.dataset:
        for reg in a.regime:
            df = load(a.root, ds, reg)
            if df is None:
                missing.append((ds, reg))
                continue
            metrics = metrics_of(df, ds, a.which)
            if not metrics:
                missing.append((ds, reg))
                continue
            t = cell_table(df, ds, metrics, a.level)
            if not (t.family == "ours").any():
                no_gnn.append((ds, reg))
            longs.append(t.assign(dataset=ds, regime=reg))
            for m in metrics:
                # one column per (cell, metric): with --which primary that is one
                # column per cell, which is the table the paper prints
                metric_of[(ds, m)] = m
                panels.append((ds, reg, t[["prot", "family", "label", "pdim",
                                           m, f"{m}_hw"]], m))

    if missing:
        cells = ", ".join(f"{d}/{r}" for d, r in missing)
        print(f"\nNOTE: nothing on disk for {cells}. Fit it with:\n"
              f"    .venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py "
              f"--dataset <ds> --regime <regime> --gnn\n")
    if no_gnn:
        cells = ", ".join(f"{d}/{r}" for d, r in no_gnn)
        print(f"NOTE: no GNN rows in {cells} -- the table is the floor without us.\n"
              f"      Add them (same folds, same script):\n"
              f"        .venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py "
              f"--dataset <ds> --regime <regime> --gnn esm3@1 esm3@0 prott5@1\n")
    if not longs:
        return 1
    packed = [(ds, reg, t) for ds, reg, t, _ in panels]
    mof = {ds: m for ds, _, _, m in panels}
    rows, cols, wide, fam, lab, dims, ranks = combine(packed, mof)
    print(text(rows, cols, wide, fam, lab, dims, ranks))
    pd.concat(longs, ignore_index=True).to_csv(out / "protein_long.csv", index=False)
    (out / "protein_sources.tex").write_text(
        latex(rows, cols, wide, fam, lab, dims, ranks) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
