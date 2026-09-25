#!/usr/bin/env python
"""The architecture table: does the operator matter? Six numbers per row.

One row per message-passing operator, one column per (dataset, regime), each column
the metric of record -- six numbers, which is the whole table. The boosting base is
the anchor row, because "our graph beats/loses to the base" is the comparison every
other table in this paper makes, and an architecture table that dropped it would be a
league of graphs with no ground under it.

READ-ONLY. Everything is trained by `scripts/article_sweeps/s5_run_architecture.py`,
which pins the graph per dataset and moves the operator alone.

    .venv/bin/python scripts/article_tables/s5_architecture.py \\
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
#: The operator this project uses. It carries the `:paper` suffix only because that is
#: how the regime is named on disk; the table prints it as plain GraphSAGE.
OURS = "sage:paper"

LABEL = {"boost_full": "Boosting base (no graph)", "sage": "GraphSAGE",
         "gat": "GAT", "graphconv": "GraphConv", "gin": "GIN",
         "none": "No message passing"}
#: The encoder ablations, as they are spelled in a row name and as they are printed.
#: They are NOT operators, so they are grouped apart and left out of the rank.
ABLATION_NOTE = {"1layer": "one layer", "pos": "positive edges only",
                 "unsigned": "unsigned edges"}
#: `:paper` marks the regime this project now trains in -- neighbour sampling plus
#: per-layer L2 normalisation, GraphSAGE's own. Since it is the only regime the table
#: reports, it is NOT printed: a suffixed row is just that operator. An un-suffixed row
#: is the historical encoder, and if one turns up it says so, because a table mixing the
#: two would be comparing regimes while claiming to compare operators.
PAPER_SUFFIX = ":paper"
HISTORICAL_NOTE = "full neighbourhood, un-normalised"
#: Compact names for a header that has to fit six times across a terminal.
SHORT_DS = {"m2or": "M2OR", "cc": "Carey", "hc": "Hallem"}
SHORT_REG = {"transductive": "trans", "inductive": "cold mol",
             "cold_receptor": "cold rec"}


def split_arch(arch):
    """`sage:paper:pos` -> ('sage', '', True, ['pos']). The row name is the record of
    what was trained, so it is parsed rather than re-derived from the columns."""
    parts = str(arch).split(":")
    head, tail = parts[0], parts[1:]
    paper = "paper" in tail
    notes = [t for t in tail if t != "paper"]
    base, _, width = head.partition("@")
    return base, width, paper, notes


def label(arch):
    """`gat@512` -> `GAT, width 512`; `sage:paper:pos` -> `GraphSAGE, positive edges
    only`; a bare operator keeps its pretty name."""
    base, width, paper, notes = split_arch(arch)
    out = LABEL.get(base, base)
    if width:
        out += f", width {width}"
    for n in notes:
        out += f", {ABLATION_NOTE.get(n, n)}"
    if base in ("boost_full", "none") or paper:
        return out
    return f"{out} ({HISTORICAL_NOTE})"


def is_operator(arch):
    """True for the message-passing operators in the reported regime.

    Not the same question as `is_ranked`: `none` and the anchor are rows that predict
    without an operator, and an encoder ablation is a variant of ours rather than a
    fifth operator."""
    base, _, _, notes = split_arch(arch)
    return base not in (ANCHOR, "none") and not notes


def in_ablation_block(arch):
    """The second, closed set: our own row, its encoder ablations, and the control.

    Our row is in it as the reference -- "one layer" is only meaningful beside the two
    layers it removes one from -- but its PRINTED rank stays the operator one, because
    that is the number the text quotes.
    """
    base, _, _, notes = split_arch(arch)
    return arch == OURS or base == "none" or bool(notes)


def is_ranked(arch):
    """True for the rows the operator rank ranks: the operators, and nothing else.

    The boosting base is deliberately out. It is the thing every graph in this paper is
    compared against, not one of the graphs; ranking it among them would make the column
    answer a different question from the one its name asks. Its number sits in the row
    above, to be read against the marked one.
    """
    return is_operator(arch)


def row_group(arch):
    """Which block a row belongs to: anchor, operators, encoder ablations, the control.

    The blocks are the table's argument. The operator block answers "does the operator
    matter"; the ablation block answers "does the signed two-hop design matter"; the
    control answers "does message passing matter at all". One undivided list would
    imply they are five answers to one question."""
    base, _, _, notes = split_arch(arch)
    if arch == ANCHOR:
        return 0
    if base == "none":
        return 3
    return 1 if not notes else 2


def row_order(arch):
    """Anchor first, ours second, then the other operators, then the encoder ablations,
    then the no-message-passing control. Read top-down it is 'the base, us, the
    alternatives, the parts of us, and no graph at all'."""
    return (row_group(arch), 0 if arch == OURS else 1, str(arch))


# ------------------------------------------------------------------ loading

def load(root, dataset, regime, mol_source):
    """One cell's CSV, or None. Missing is normal while a panel is still running."""
    p = tk.resolve(f"{root}/metrics_{dataset}_{regime}__{mol_source}.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.attrs["path"] = str(p)
    return df


def reported_only(d):
    """Drop the historical-encoder rows, keep the anchor and the reported regime.

    The anchor has no encoder at all and `none` has no layers to sample or normalise,
    so both are kept by name rather than by suffix. Every other row must carry
    `:paper`, wherever in its name it sits."""
    arch = d["arch"].astype(str)
    paper = arch.str.split(":").apply(lambda p: PAPER_SUFFIX[1:] in p[1:])
    return d[paper | arch.eq(ANCHOR) | arch.str.split(":").str[0].eq("none")]


def cell_table(df, dataset, combo, level=0.95, historical=False):
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
    if not historical:
        d = reported_only(d)
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
        # TWO ranks, each over a closed set, because a rank is a mean place among a
        # fixed field and a row added to the field moves every number in it. The
        # operator field answers "which operator"; the ablation field (ours, its
        # encoder ablations and the control) answers "which part of the design is
        # load-bearing". The boosting base is in neither: it is what every graph here
        # is read against. A row belongs to exactly one PRINTED rank -- ours is ranked
        # as an operator, though it takes part in the ablation field as its reference.
        def _order(sub):
            return sub["value"].rank(ascending=metric in ag.LOWER_IS_BETTER,
                                     method="average")

        op_order = _order(t[t["arch"].map(is_ranked)])
        abl_order = _order(t[t["arch"].map(in_ablation_block)])
        for i, r in t.iterrows():
            wide.setdefault(r["arch"], {})[col] = (r["value"], r["hw"])
            labels[r["arch"]] = r["label"]
            use = op_order if i in op_order.index else abl_order
            if i in use.index:
                ranks.setdefault(r["arch"], []).append(float(use.loc[i]))
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
        rank = f"{np.mean(ranks[n]):.1f}" if ranks.get(n) else "--"
        out.append(f"{labels[n]:<{w}}{rank:>7}  {cells}")

    ties = [f"{SHORT_DS.get(ds, ds)}/{SHORT_REG.get(reg, reg)}"
            for (ds, reg, m) in cols
            if overlaps(wide, (ds, reg, m), OURS, best[(ds, reg, m)]) is True
            and best[(ds, reg, m)] != OURS]
    out += ["", "* = best operator in column (the base is the anchor, not a "
            "competitor). mean +/- 95% CI over folds;",
            "    seeds averaged inside each fold first.",
            "rank = mean place averaged over columns (smaller is better), taken "
            "within a field:",
            "    the operators in one; our row, its encoder ablations and the "
            "no-message-passing",
            "    control in the other, where our row is the reference and keeps its "
            "operator rank.",
            "    The base is in neither -- it is what every graph here is read "
            "against."]
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
            rank = f"{np.mean(ranks[n]):.1f}" if ranks.get(n) else "--"
            body.append(rf"\emph{{{labels[n]}}} & \emph{{{rank}}} & "
                        + " & ".join(cells) + r" \\")
            body.append(r"\addlinespace")
            continue
        cells = []
        for c in cols:
            v = fmt(wide[n].get(c)).replace("+/-", r"$\pm$")
            cells.append(rf"\textbf{{{v}}}" if best[c] == n else v)
        rank = f"{np.mean(ranks[n]):.1f}" if ranks.get(n) else "--"
        body.append(f"{labels[n]} & {rank} & " + " & ".join(cells) + r" \\")
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
        r"graphs are read against and is not marked, though it does take part in the "
        r"rank. The rank is a mean place within a field: the operators form one, and "
        r"our row together with its encoder ablations and the no-message-passing "
        r"control forms the other, so that adding an ablation cannot move an "
        r"operator's number. Our row is ranked as an operator; the base is ranked in "
        r"neither, being what every graph here is read against. "
        r"\texttt{--} = not ranked, or not trained in that cell.}",
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
    ap.add_argument("--historical", action="store_true",
                    help="also show the pre-23.09.2026 encoder rows (full "
                         "neighbourhood, un-normalised). Off by default: they are a "
                         "different model, and a table holding both compares regimes "
                         "while claiming to compare operators")
    ap.add_argument("--root", default="results/article_sweeps/architecture",
                    help="where s5_run_architecture.py wrote its CSVs")
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
            t, metric = cell_table(df, ds, a.combo, a.level, a.historical)
            if t is None or t.empty:
                missing.append((ds, reg))
                continue
            cells.append((ds, reg, metric, t))
            longs.append(t.assign(dataset=ds, regime=reg, metric=metric))

    if missing:
        print("no rows yet for: " + ", ".join(f"{d}/{r}" for d, r in missing))
    if not cells:
        print("\nnothing to render. Train them first:\n"
              "  .venv/bin/python scripts/article_sweeps/s5_run_architecture.py \\\n"
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
