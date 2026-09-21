#!/usr/bin/env python
"""The identity control: our graph with NO sequence at all, against two boostings.

On the v9 node dial, alpha=0 replaces the ESM vector with one fixed near-orthogonal
vector per receptor, so the refined receptor carries only what the response profile
put there. The question the table asks is therefore not "is the graph good" but
"how much of what the graph does needs a protein language model at all".

Three rows per cell, on the same folds:

    graph a=0        our cls+mol at alpha 0 -- receptor identity refined by function
    boost [ESM|mol]  the sweep's own boost_full reference
    boost [1hot|mol] the same head over a ONE-HOT receptor block instead of ESM

The first two are read from the sweep; the third comes from `05a_onehot_boost.py`,
which fits it. That one is the honest floor for the first: a one-hot block is receptor
identity with no refinement, so a graph at alpha=0 that does not beat it has learned nothing from the
response profile, and one that beats boost-over-ESM while tying one-hot says the win
was never about sequence.

    python scripts/article_tables/05_alpha0_vs_boost.py \
        --sweep-root results/graph/v13_esm3

READ-ONLY. The one-hot heads are fitted by `05a_onehot_boost.py`, which writes one
CSV per (dataset, regime) under results/article_tables/onehot_boost/; this script only
reads them, so it is fast and safe to re-run while tweaking a label. If those files are
absent the column reads `--` and the run prints the command that makes them.

Writes to results/article_tables/alpha0/: alpha0_long.csv, alpha0.tex, and a printed
text block.
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

#: The row order of the table, and the labels it prints.
GRAPH, BOOST, ONEHOT = "graph a=0", "boost [ESM || mol]", "boost [1hot || mol]"
ROWS = (GRAPH, BOOST, ONEHOT)


# ------------------------------------------------------------------ the one-hot rows

def onehot_table(ds, regime, cache):
    """The one-hot head's rows for this cell, as `05a_onehot_boost.py` wrote them.

    Missing is not an error: the one-hot column simply reads `--`, and the message
    below says what to run. That keeps this script fast and read-only -- fitting a
    1237-wide one-hot block on M2OR is minutes per fold, and it does not belong in
    something you re-run to change a label.
    """
    path = pathlib.Path(cache)
    if not path.is_absolute():
        path = tk.ROOT / path
    path = path / f"{ds}_{regime}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ------------------------------------------------------------------ the comparison

def fold_values(df, metric):
    """{fold: value}, model seeds averaged inside the fold. The unit of evidence."""
    if metric not in df.columns or df.empty:
        return {}
    f = ag.fold_means(df.assign(**{metric: pd.to_numeric(df[metric], errors="coerce")}),
                      metric)
    return {int(r.fold): float(r[metric]) for _, r in f.iterrows()
            if np.isfinite(r[metric])}


def wins(a_vals, b_vals, metric):
    """How many shared folds `a` beat `b` on, and how many they shared.

    The direction comes from the metric, not from the sign: RMSE and MAE are won by
    being SMALLER, and counting them like a score hands the win to the worse model on
    every error column.
    """
    shared = sorted(set(a_vals) & set(b_vals))
    if not shared:
        return None, 0
    sign = -1.0 if metric in ag.LOWER_IS_BETTER else 1.0
    return sum(1 for f in shared if sign * (a_vals[f] - b_vals[f]) > 0), len(shared)


def cell(vals, level):
    """mean +- t half-width over folds, as the table prints it."""
    if not vals:
        return dict(mean=np.nan, hw=np.nan, n=0)
    m, hw, n = ag.ci(list(vals.values()), level)
    return dict(mean=m, hw=hw, n=n)


def build(a):
    df = ag.load(root=a.sweep_root, mol_source=a.mol_source, nodes=a.nodes,
                 dataset=a.dataset, regime=a.regime, combo=a.combo)
    ds_of = dict(zip(df.series, df.dataset))
    reg_of = dict(zip(df.series, df.regime))
    out, missing = [], set()
    for s in sorted(df.series.unique()):
        sub = df[df.series == s]
        ds, regime = ds_of[s], reg_of[s]
        oh = onehot_table(ds, regime, a.cache)
        if oh is None:
            missing.add((ds, regime))
        alphas = pd.to_numeric(sub.loc[sub.arm == "gate", "alpha"], errors="coerce")
        if not np.isclose(alphas.dropna(), a.alpha).any():
            raise SystemExit(
                f"{s}: alpha={a.alpha:g} was never run -- on disk: "
                f"{', '.join(f'{x:g}' for x in sorted(alphas.dropna().unique()))}")
        graph_all = sub[(sub.arm == "gate")
                        & np.isclose(pd.to_numeric(sub.alpha, errors="coerce"), a.alpha)]
        boost_all = sub[sub.arm == "boost_full"]
        for m in ag.metrics_available(sub, dataset=ds, which=a.which):
            g, b = fold_values(graph_all, m), fold_values(boost_all, m)
            o = ({} if oh is None or m not in oh.columns
                 else fold_values(oh, m))
            wb, nb = wins(g, b, m)
            wo, no = wins(g, o, m)
            out.append(dict(
                series=s, dataset=ds, regime=regime, metric=m,
                **{f"{k}_{n}": v for n, d in
                   ((GRAPH, cell(g, a.level)), (BOOST, cell(b, a.level)),
                    (ONEHOT, cell(o, a.level)))
                   for k, v in d.items()},
                won_vs_boost=wb, n_vs_boost=nb, won_vs_onehot=wo, n_vs_onehot=no))
    if missing:
        cells = ", ".join(f"{d}/{r}" for d, r in sorted(missing))
        datasets = " ".join(sorted({d for d, _ in missing}))
        print(f"\nNOTE: no one-hot rows for {cells} -- that column reads '--'.\n"
              f"      Fit them once (slow, then cached) with:\n"
              f"        python scripts/article_tables/05a_onehot_boost.py"
              f" --dataset {datasets}"
              f" --prot-embeddings '<the npz the sweep used>'\n")
    return pd.DataFrame(out)


# ------------------------------------------------------------------ rendering

def num(row, which):
    m, hw = row[f"mean_{which}"], row[f"hw_{which}"]
    if not np.isfinite(m):
        return "--"
    return f"{m:.3f}" + ("" if not np.isfinite(hw) else f"+/-{hw:.3f}")


def score(w, n, flag_at):
    if w is None or not n:
        return "--", ""
    return f"{int(w)}/{int(n)}", ("<<" if int(w) >= flag_at else "")


def text(t, flag_at):
    lines = [f"{'series':<20} {'metric':<9} {GRAPH:>16} {BOOST:>20} {ONEHOT:>21} "
             f"{'vs boost':>9} {'vs 1hot':>9}"]
    lines.append("-" * len(lines[0]))
    for _, r in t.iterrows():
        wb, fb = score(r.won_vs_boost, r.n_vs_boost, flag_at)
        wo, fo = score(r.won_vs_onehot, r.n_vs_onehot, flag_at)
        lines.append(f"{r.series:<20} {r.metric:<9} {num(r, GRAPH):>16} "
                     f"{num(r, BOOST):>20} {num(r, ONEHOT):>21} "
                     f"{wb + fb:>9} {wo + fo:>9}")
    lines.append("")
    lines.append(f"'<<' marks {flag_at}+ wins. Error metrics (RMSE, MAE) are won by "
                 f"being smaller and are counted that way.")
    return "\n".join(lines)


def latex(t, a):
    head = (r"\begin{table}[!ht]" "\n" r"\centering" "\n" r"\small" "\n"
            r"\caption{Our graph with the receptor's sequence removed "
            r"($\alpha=0$ on the node dial: the ESM vector is replaced by one fixed "
            r"near-orthogonal vector per receptor), against the boosting reference "
            r"over $[\mathrm{ESM}\|\mathrm{mol}]$ and against the same head over a "
            r"\textbf{one-hot} receptor block. 5 held-out splits, mean $\pm$ "
            + f"{a.level:.0%}".replace("%", r"\%") +
            r" CI over splits, model seeds averaged inside each split first. "
            r"\emph{Wins} counts the splits our row is ahead on; RMSE and MAE are won "
            r"by being smaller and are counted that way.}" "\n"
            r"\label{tab:alpha0}" "\n"
            r"\resizebox{\textwidth}{!}{%" "\n"
            r"\begin{tabular}{@{}ll ccc cc@{}}" "\n" r"\toprule" "\n"
            r"\textbf{Panel} & \textbf{Metric} & Graph $\alpha{=}0$ & "
            r"Boost [ESM$\|$mol] & Boost [1-hot$\|$mol] & "
            r"\multicolumn{2}{c}{Wins} \\" "\n"
            r"\cmidrule(lr){6-7}" "\n"
            r" & & & & & vs ESM & vs 1-hot \\" "\n" r"\midrule")
    body, last = [], None
    for _, r in t.iterrows():
        if last is not None and r.series != last:
            body.append(r"\midrule")
        last = r.series
        wb, _ = score(r.won_vs_boost, r.n_vs_boost, a.flag_at)
        wo, _ = score(r.won_vs_onehot, r.n_vs_onehot, a.flag_at)
        cells = [num(r, GRAPH).replace("+/-", r"$\pm$"),
                 num(r, BOOST).replace("+/-", r"$\pm$"),
                 num(r, ONEHOT).replace("+/-", r"$\pm$")]
        body.append(f"{tk.DATASET_LABEL.get(r.dataset, r.dataset)} / {r.regime} & "
                    f"{r.metric} & " + " & ".join(cells) + f" & {wb} & {wo} " + r"\\")
    return "\n".join([head] + body + [r"\bottomrule", r"\end{tabular}}",
                                      r"\end{table}"])


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", default="results/graph/v13_esm3")
    ap.add_argument("--nodes", default="nodedial",
                    help="which dial the sweep is; the two must never mix")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="the dial position to read; 0 is identity-only on the v9 dial")
    ap.add_argument("--mol-source", default="chemberta", choices=tk.MOL_SOURCES)
    ap.add_argument("--combo", default="cls+mol",
                    help="which graph head; cls+mol is the one comparable to a "
                         "competitor's own cls")
    ap.add_argument("--dataset", nargs="+", default=None)
    ap.add_argument("--regime", nargs="+", default=None)
    ap.add_argument("--which", default="headline", choices=["headline", "all"])
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--flag-at", type=int, default=3,
                    help="mark a row with this many wins or more")
    ap.add_argument("--cache", default="results/article_tables/onehot_boost",
                    help="where 05a_onehot_boost.py wrote its CSVs")
    ap.add_argument("--out", default="results/article_tables/alpha0")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    t = build(a)
    if t.empty:
        print("nothing on disk for those knobs")
        return 0
    out = tk.out_dir(a.out)
    print("\n" + text(t, a.flag_at))
    t.to_csv(out / "alpha0_long.csv", index=False)
    (out / "alpha0.tex").write_text(latex(t, a) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
