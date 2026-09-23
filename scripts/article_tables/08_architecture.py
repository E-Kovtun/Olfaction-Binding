#!/usr/bin/env python
"""The architecture table: does the operator matter? Six numbers per row.

One row per message-passing operator, one column per (dataset, regime), each column
the metric of record -- six numbers, which is the whole table. The boosting base is
the anchor row, because "our graph beats/loses to the base" is the comparison every
other table in this paper makes, and an architecture table that dropped it would be a
league of graphs with no ground under it.

READ-ONLY. Everything is trained by `scripts/article_sweeps/run_architecture.py`,
which pins the graph per dataset and moves the operator alone.

    .venv/bin/python scripts/article_tables/08_architecture.py \\
        --dataset m2or cc hc --regime transductive inductive

WHAT THE TABLE IS FOR, AND WHAT IT IS NOT FOR. It is here to show that the receptor
refinement, not the operator, is what this paper claims. If four operators land inside
each other's intervals, that is the result and it supports the claim; a table that
crowned one of them by a hundredth would be reporting the seed lottery. The reader
therefore prints the spread and marks the best in column, but says nothing about
significance that the intervals do not.

Writes `results/article_tables/architecture/`: `architecture_long.csv` (every row,
every metric), `architecture.tex`, and the same table as text.
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

#: The anchor, and the operator the paper reports. Everything else is printed in the
#: order the sweep produced it, so a new operator does not need a code change here.
ANCHOR = "boost_full"
OURS = "sage:paper"

LABEL = {"boost_full": "Boosting base (no graph)", "sage": "GraphSAGE",
         "gat": "GAT", "graphconv": "GraphConv", "gin": "GIN"}
#: `:paper` = that operator run the way its paper runs it -- neighbour sampling plus
#: per-layer L2 normalisation, the two things our encoder took from neither.
PAPER_SUFFIX = ":paper"
PAPER_NOTE = "sampled + normalised"
#: Which row the paper reports. Until 23.09.2026 this was the un-suffixed `sage`; the
#: ablation moved the default, so it is now the sampled + normalised one, and the plain
#: rows are the historical encoder kept for the comparison.
OURS_NOTE = "ours"
#: Compact names for a header that has to fit six times across a terminal.
SHORT_DS = {"m2or": "M2OR", "cc": "Carey", "hc": "Hallem"}
SHORT_REG = {"transductive": "trans", "inductive": "cold mol",
             "cold_receptor": "cold rec"}


def label(arch):
    """`gat@512` -> `GAT, width 512`; `sage:paper` -> `GraphSAGE (ours), sampled +
    normalised`; a bare operator keeps its pretty name."""
    name = str(arch)
    paper = name.endswith(PAPER_SUFFIX)
    if paper:
        name = name[: -len(PAPER_SUFFIX)]
    base, _, width = name.partition("@")
    out = LABEL.get(base, base)
    if width:
        out += f", width {width}"
    if base == "boost_full":
        return out
    note = f"{PAPER_NOTE}, {OURS_NOTE}" if paper else "full neighbourhood, un-normalised"
    return f"{out} ({note})"


def row_order(arch):
    """Anchor first, ours second, then everything else alphabetically. The table is
    read top-down as 'the base, us, the alternatives', which is its argument."""
    return (0 if arch == ANCHOR else 1 if arch == OURS else 2, str(arch))


# ------------------------------------------------------------------ loading

def load(root, dataset, regime, mol_source):
    """One cell's CSV, or None. Missing is normal while a panel is still running."""
    p = tk.resolve(f"{root}/metrics_{dataset}_{regime}__{mol_source}.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.attrs["path"] = str(p)
    return df


def cell_table(df, dataset, combo, level=0.95):
    """One row per operator: mean +- t half-width over FOLDS, on the test split.

    Model seeds are averaged inside each fold FIRST. The unit of evidence is the
    held-out split; counting (fold, seed) cells as observations would halve every
    interval here, and with four operators sitting close together that is precisely
    the error that would manufacture a winner.

    The anchor keeps its own head (`prot+mol`) -- it has no `cls` to concatenate --
    so it is selected by name and not by `combo`.
    """
    d = df[df["split"].astype(str).eq("test")]
    if "status" in d.columns:
        d = d[d["status"].astype(str).eq("ok")]
    d = d[d["combo"].astype(str).eq(combo) | d["conv"].astype(str).eq(ANCHOR)]
    metric = ag.OF_RECORD[ag.TASK[dataset]]
    if metric not in d.columns or d.empty:
        return None, metric

    out = []
    for arch, g in d.groupby("arch", sort=False):
        fold_means = g.groupby("fold")[metric].mean()
        mu, hw, n = ag.ci(fold_means, level)
        out.append(dict(arch=arch, label=label(arch), value=mu, hw=hw, folds=n,
                        seeds=int(g["seed"].nunique()),
                        hidden=pd.to_numeric(g["hidden"], errors="coerce").max()))
    t = pd.DataFrame(out)
    return t.sort_values("arch", key=lambda s: s.map(row_order)).reset_index(drop=True), metric


# ------------------------------------------------------------------ the wide shape

def combine(cells):
    """Every cell side by side: one row per operator, one column per (dataset, regime).

    The join is OUTER on the operator: an operator run on one panel and not yet on
    another stays visible with `--`, because "not trained here" and "trained and bad"
    are different statements and a half-finished sweep must not read as the second.
    """
    wide, labels, ranks = {}, {}, {}
    for ds, reg, metric, t in cells:
        col = (ds, reg, metric)
        order = t["value"].rank(ascending=metric in ag.LOWER_IS_BETTER, method="average")
        for i, r in t.iterrows():
            wide.setdefault(r["arch"], {})[col] = (r["value"], r["hw"])
            labels[r["arch"]] = r["label"]
            ranks.setdefault(r["arch"], []).append(float(order.loc[i]))
    cols = [(ds, reg, m) for ds, reg, m, _ in cells]
    rows = sorted(wide, key=row_order)
    return rows, cols, wide, labels, ranks


def best_in_column(rows, col, wide, skip_anchor=True):
    """The winning row of one column, direction-aware.

    The anchor is excluded by default: it is the thing being compared against, and
    marking it as the best OPERATOR would say something the table does not mean. Its
    number is right there to be read against the marked one.
    """
    have = [(wide[n][col][0], n) for n in rows
            if col in wide[n] and np.isfinite(wide[n][col][0])
            and not (skip_anchor and n == ANCHOR)]
    if not have:
        return None
    return (min if col[2] in ag.LOWER_IS_BETTER else max)(have)[1]


def overlaps(wide, col, a, b):
    """Do these two rows' intervals overlap in this column? The only claim this table
    makes about significance, and it makes it by arithmetic rather than by adjective."""
    if col not in wide.get(a, {}) or col not in wide.get(b, {}):
        return None
    (va, ha), (vb, hb) = wide[a][col], wide[b][col]
    if not all(np.isfinite(x) for x in (va, ha, vb, hb)):
        return None
    return abs(va - vb) <= (ha + hb)


def fmt(cellv, nd=3):
    if cellv is None:
        return "--"
    v, hw = cellv
    if not np.isfinite(v):
        return "--"
    return f"{v:.{nd}f}" + ("" if not np.isfinite(hw) else f"+/-{hw:.{nd}f}")


# ------------------------------------------------------------------ rendering

def text(rows, cols, wide, labels, ranks):
    w = max(len(labels[n]) for n in rows) + 2
    cw = 18
    pre = f"{'architecture':<{w}}{'rank':>7}  "
    head = [f"{SHORT_DS.get(ds, ds)}/{SHORT_REG.get(reg, reg)}" for ds, reg, _ in cols]
    line = pre + "".join(f"{h:>{cw}}" for h in head)
    out = ["", "=" * len(line), line,
           " " * len(pre) + "".join(f"{m:>{cw}}" for _, _, m in cols),
           "-" * len(line)]
    best = {c: best_in_column(rows, c, wide) for c in cols}
    for n in rows:
        cells = "".join(f"{fmt(wide[n].get(c)) + ('*' if best[c] == n else ' '):>{cw}}"
                        for c in cols)
        rank = "" if n == ANCHOR else f"{np.mean(ranks[n]):.1f}"
        out.append(f"{labels[n]:<{w}}{rank:>7}  {cells}")

    ties = [f"{SHORT_DS.get(ds, ds)}/{SHORT_REG.get(reg, reg)}"
            for (ds, reg, m) in cols
            if overlaps(wide, (ds, reg, m), OURS, best[(ds, reg, m)]) is True
            and best[(ds, reg, m)] != OURS]
    out += ["", "* = best operator in column (the base is the anchor, not a "
            "competitor). mean +/- 95% CI over folds;",
            "    seeds averaged inside each fold first.",
            "rank = mean place among the operators, averaged over columns "
            "(smaller is better)."]
    if ties:
        out += [f"ours overlaps the marked operator's interval in: {', '.join(ties)} "
                f"-- those columns separate nothing."]
    return "\n".join(out)


def latex(rows, cols, wide, labels, ranks):
    best = {c: best_in_column(rows, c, wide) for c in cols}
    body = []
    for n in rows:
        if n == ANCHOR:
            cells = [fmt(wide[n].get(c)).replace("+/-", r"$\pm$") for c in cols]
            body.append(rf"\emph{{{labels[n]}}} & & " + " & ".join(cells) + r" \\")
            body.append(r"\addlinespace")
            continue
        cells = []
        for c in cols:
            v = fmt(wide[n].get(c)).replace("+/-", r"$\pm$")
            cells.append(rf"\textbf{{{v}}}" if best[c] == n else v)
        body.append(f"{labels[n]} & {np.mean(ranks[n]):.1f} & "
                    + " & ".join(cells) + r" \\")
    heads = " & ".join(rf"\textbf{{{tk.DATASET_LABEL.get(ds, ds)}}}" for ds, _, _ in cols)
    sub = " & ".join(rf"{reg.replace('_', ' ')} ({m})" for _, reg, m in cols)
    return "\n".join([
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{\textbf{The operator is not what this paper claims.} The same "
        r"signed bipartite encoder, the same graph, the same decoder, the same epoch "
        r"budget and the same boosting head in every row --- only the message-passing "
        r"operator changes. The graph itself is fixed per dataset at the construction "
        r"the main tables use, so this table moves one axis and not two. Each column "
        r"reports that panel's metric of record, 5 held-out splits, mean $\pm$ 95\\% "
        r"CI over splits, model seeds averaged inside each split first. "
        r"\textbf{Bold} = best operator in column; the boosting base is the anchor the "
        r"graphs are read against and is not marked. \texttt{--} = not trained in that "
        r"cell.}",
        r"\label{tab:arch}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lr" + "r" * len(cols) + r"@{}}", r"\toprule",
        r"\textbf{Architecture} & \textbf{rank} & " + heads + r" \\",
        r" & & " + sub + r" \\", r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


# ------------------------------------------------------------------ driver

def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["m2or", "cc", "hc"],
                    choices=["m2or", "cc", "hc"])
    ap.add_argument("--regime", nargs="+", default=["transductive", "inductive"])
    ap.add_argument("--combo", default="cls+mol",
                    choices=["cls+mol", "cls+prot+mol"],
                    help="which head the graph rows come from; the anchor keeps its "
                         "own prot+mol either way")
    ap.add_argument("--mol-source", default="chemberta",
                    choices=["chemberta", "gin", "ecfp"])
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--root", default="results/article_sweeps/architecture",
                    help="where run_architecture.py wrote its CSVs")
    ap.add_argument("--out", default="results/article_tables/architecture")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    cells, longs, missing = [], [], []
    for ds in a.dataset:
        for reg in a.regime:
            df = load(a.root, ds, reg, a.mol_source)
            if df is None:
                missing.append((ds, reg))
                continue
            t, metric = cell_table(df, ds, a.combo, a.level)
            if t is None or t.empty:
                missing.append((ds, reg))
                continue
            cells.append((ds, reg, metric, t))
            longs.append(t.assign(dataset=ds, regime=reg, metric=metric))

    if missing:
        print("no rows yet for: " + ", ".join(f"{d}/{r}" for d, r in missing))
    if not cells:
        print("\nnothing to render. Train them first:\n"
              "  .venv/bin/python scripts/article_sweeps/run_architecture.py \\\n"
              "      --dataset m2or cc hc --regime transductive inductive \\\n"
              "      --seeds 42 43 44 45 46 --seed-graph")
        return 1

    rows, cols, wide, labels, ranks = combine(cells)
    out = tk.out_dir(a.out)
    pd.concat(longs, ignore_index=True).to_csv(out / "architecture_long.csv",
                                               index=False)
    (out / "architecture.tex").write_text(latex(rows, cols, wide, labels, ranks) + "\n",
                                          encoding="utf-8")
    block = text(rows, cols, wide, labels, ranks)
    print(block)
    (out / "architecture.txt").write_text(block + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
