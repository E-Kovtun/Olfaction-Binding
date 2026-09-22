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

    .venv/bin/python scripts/article_tables/07_protein_sources.py \\
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
PLM = {"esm1b", "esm2", "prott5", "esm3", "esmc"}
CLASSICAL = {"aac", "kmer2", "aaindex", "ctd", "pseaac", "blosum"}
CONTROL = {"onehot", "onehot_only", "mol_only"}
#: How a row is printed. Everything else keeps the name the sweep wrote.
LABEL = {"onehot": "one-hot receptor", "onehot_only": "one-hot, no molecule",
         "mol_only": "molecule only", "aac": "AAC", "kmer2": "k-mer (2)",
         "aaindex": "AAindex", "ctd": "CTD", "pseaac": "PseAAC", "blosum": "BLOSUM",
         "esm1b": "ESM-1b", "esm2": "ESM-2", "prott5": "ProtT5", "esm3": "ESM3"}


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


# ------------------------------------------------------------------ rendering

def num(t, i, m, nd=3):
    v, hw = t.loc[i, m], t.loc[i, f"{m}_hw"]
    if not np.isfinite(v):
        return "--"
    return f"{v:.{nd}f}" + ("" if not np.isfinite(hw) else f"+/-{hw:.{nd}f}")


def text(t, metrics, title):
    w = max(len(str(x)) for x in t["label"]) + 2
    head = f"{'representation':<{w}}{'dim':>7}  " + "".join(f"{m:>18}" for m in metrics)
    lines = ["", title, "=" * len(head), head, "-" * len(head)]
    best = {m: best_of(t, m) for m in metrics}
    last = None
    for i, r in t.iterrows():
        if r["family"] != last:
            lines.append(f"-- {FAMILY_LABEL[r['family']]}")
            last = r["family"]
        cells = "".join(f"{num(t, i, m) + ('*' if best[m] == i else ' '):>18}"
                        for m in metrics)
        lines.append(f"{r['label']:<{w}}{r['pdim']:>7}  {cells}")
    lines.append("")
    lines.append("* = best in column. mean +/- 95% CI over folds; seeds (boost and "
                 "graph) averaged inside each fold first.")
    return "\n".join(lines)


def latex(t, metrics, dataset, regime):
    body, last = [], None
    best = {m: best_of(t, m) for m in metrics}
    for i, r in t.iterrows():
        if r["family"] != last:
            body.append(r"\addlinespace" if last is not None else "")
            body.append(rf"\multicolumn{{{len(metrics) + 2}}}{{@{{}}l}}"
                        rf"{{\emph{{{FAMILY_LABEL[r['family']]}}}}} \\")
            last = r["family"]
        cells = []
        for m in metrics:
            s = num(t, i, m).replace("+/-", r"$\pm$")
            cells.append(rf"\textbf{{{s}}}" if best[m] == i else s)
        body.append(f"{r['label']} & {r['pdim']} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{\textbf{What the receptor side has to be.} The same boosting head "
        r"over $[\text{receptor representation}\,\|\,\text{ChemBERTa}]$ on "
        rf"{tk.DATASET_LABEL.get(dataset, dataset)}, {regime}, identical folds "
        r"throughout. \emph{Our graph} rows are the refined receptor vector beside the "
        r"molecule --- our \texttt{cls+mol} --- trained inside these same folds rather "
        r"than imported, since a representation fitted on one fold's training pairs "
        r"is not a fixed feature elsewhere. \emph{identity nodes} is the node dial at "
        r"zero: the graph is told who the receptor is and nothing about its sequence. "
        r"5 held-out splits, mean $\pm$ 95\% CI over splits, seeds averaged inside "
        r"each split first. \textbf{Bold} = best in column.}",
        rf"\label{{tab:protsrc{dataset}{regime[:4]}}}",
        r"\begin{tabular}{@{}lr" + "r" * len(metrics) + r"@{}}", r"\toprule",
        r"\textbf{Receptor representation} & \textbf{dim} & "
        + " & ".join(rf"\textbf{{{m}}}" for m in metrics) + r" \\",
        r"\midrule", *[b for b in body if b != ""],
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])


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
    longs, tex, missing, no_gnn = [], [], [], []
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
            print(text(t, metrics, f"{tk.DATASET_LABEL.get(ds, ds)} / {reg}"))
            tex.append(latex(t, metrics, ds, reg))
            longs.append(t.assign(dataset=ds, regime=reg))

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
    pd.concat(longs, ignore_index=True).to_csv(out / "protein_long.csv", index=False)
    (out / "protein_sources.tex").write_text("\n\n".join(tex) + "\n", encoding="utf-8")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
