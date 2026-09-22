#!/usr/bin/env python
"""The alpha we report, as a table: chosen on validation, confirmed once on test.

Alpha is a hyperparameter -- fixed before training, used unchanged at inference, like a
learning rate. So the curve it is CHOSEN on must not be the curve it is DEFENDED with,
and this script is the article-facing form of that protocol. It decides nothing new:
the decision lives in `scripts/analysis/alpha_choice.py` and is imported from there, so
the paper's table and the command we actually run can never drift apart. What this adds
is the rendering -- LaTeX, a CSV per panel, and a text block -- and nothing else.

    .venv/bin/python scripts/article_tables/06_alpha_choice.py \
        --sweep-root results/graph/v13_esm3 --nodes nodedial

THE RUN IS AN ARGUMENT, AND IT IS REQUIRED. This table belongs to one sweep root and
says which: the ESM3 grid and an ESM-1b grid are different runs with different numbers,
and a default would let the wrong one be typeset without anybody noticing. So
`--sweep-root` has no default, the output directory is named after the root unless
`--out` says otherwise, and the root is printed under the table.

THREE TABLES, IN THE ORDER THE ARGUMENT NEEDS THEM

  selection    Every competitor -- each alpha, plus the boosting base and the legacy
               graph -- ranked WITHIN each cell on VALIDATION, the ranks averaged
               across cells. A cell is one row of the paper's tables: (dataset, regime,
               molecule source). Ranks and not means, because R2 on Carey and AUROC on
               M2OR cannot be averaged but their orderings can. The alphas within one
               standard error of the leader are marked: they are indistinguishable, and
               picking the argmax out of a flat set is noise-chasing.
  confirmation The chosen alpha read ONCE on test, per cell, beside the boosting base.
               The last column is the OPTIMISM AVOIDED -- how much the luckiest alpha
               on those very folds outscores the one validation picked. That gap is
               what choosing on test would have added to the claim silently, and
               printing it is the difference between a protocol and a assertion that
               one was followed.
  robustness   Leave one dataset out: choose on two panels, report where that alpha
               lands on the third. A choice that does not survive it is a property of
               the folds rather than of the method.

WHAT THIS SCRIPT WILL NOT DO. It computes nothing -- not the sweep, not the validation
split. It reads a root and renders it, so what it reports is what that run actually
produced. And it will not read test unless a validation frame exists: a sweep whose
`val_metrics_*.csv` are missing gets the `val_rescore.py` command for THAT root printed
and nothing else. A "choice" made on test and typeset as if it had been made on
validation is the one failure mode this whole file exists to prevent.
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
from scripts.analysis import alpha_choice as ac   # noqa: E402
from scripts.analysis import alpha_grid as ag     # noqa: E402

#: How a cell is named in a table column narrow enough to typeset.
def cell_label(row):
    return (f"{tk.DATASET_LABEL.get(row['dataset'], row['dataset'])} / "
            f"{row['regime']} / {row['mol_source']}")


# ------------------------------------------------------------------ the three tables

def selection(val_df, metric_of):
    """Mean rank per competitor on validation, with the 1-SE tie set marked."""
    rk = ac.ranked(ac.scores(val_df, metric_of))
    tied, cut = ac.one_se(rk)
    tie_set = set(tied["competitor"]) if len(tied) else set()
    return rk.assign(tied=[c in tie_set for c in rk["competitor"]], se_cut=cut)


def confirmation(test_df, metric_of, alpha):
    """The chosen alpha on test, per cell: value, advantage over boost, optimism.

    Everything here is computed by `alpha_choice`; this only reshapes it into rows. The
    advantage is paired inside each fold and the seeds are averaged first, so `n` counts
    held-out splits and not (fold, seed) cells.
    """
    sc = ac.scores(test_df, metric_of)
    adv = ac.advantage(test_df, metric_of)
    lab = f"a={alpha:g}"
    rows = []
    for key, g in sc.groupby(ac.CELL, sort=True):
        cell = dict(zip(ac.CELL, key))
        metric = metric_of[cell["dataset"]]
        row = g[g.competitor == lab]
        gate = g[g.alpha.notna()].sort_values("value", ascending=False)
        base = g[g.competitor == "boost"]
        rec = cell | dict(metric=metric, alpha=alpha,
                          base=float(base.iloc[0].value) if len(base) else np.nan)
        if row.empty or gate.empty:
            rows.append(rec | dict(value=np.nan, rank=np.nan, n_alphas=len(gate),
                                   best=np.nan, optimism=np.nan, d=np.nan,
                                   won=np.nan, n=np.nan))
            continue
        v = float(row.iloc[0].value)
        best = float(gate.iloc[0].value)
        d = adv[(adv.dataset == cell["dataset"]) & (adv.regime == cell["regime"])
                & (adv.mol_source == cell["mol_source"])
                & np.isclose(adv.alpha.astype(float), alpha)]
        rows.append(rec | dict(
            value=v, rank=int((gate.value > v).sum()) + 1, n_alphas=len(gate),
            best=best, optimism=best - v,
            d=float(d.iloc[0].d) if len(d) else np.nan,
            won=int(d.iloc[0].won) if len(d) else np.nan,
            n=int(d.iloc[0].n) if len(d) else np.nan))
    return pd.DataFrame(rows)


def robustness(val_df, metric_of):
    """Leave one dataset out. Empty on a single-panel run, which is not a failure."""
    return ac.loo(val_df, metric_of, argparse.Namespace())


# ------------------------------------------------------------------ rendering

def tex_selection(rk, level_note):
    body = []
    for _, r in rk.iterrows():
        name = r["competitor"] if not np.isfinite(r["alpha"]) else \
            rf"$\alpha = {r['alpha']:g}$"
        se = "--" if not np.isfinite(r["se"]) else f"{r['se']:.2f}"
        mark = r"\,$\dagger$" if r["tied"] else ""
        body.append(f"{name}{mark} & {r['mean_rank']:.2f} & {se} & "
                    f"{int(r['cells'])} & {int(r['best'])} \\\\")
    return "\n".join([
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{\textbf{Choosing $\alpha$ on validation.} Every competitor --- each "
        r"dial position, the boosting base and the legacy graph --- is ranked "
        r"\emph{within} each cell (one cell = one row of the main tables: dataset "
        r"$\times$ regime $\times$ molecule source), and the ranks are averaged across "
        r"cells. Ranks rather than means because $R^2$ and AUROC cannot be averaged but "
        r"their orderings can. Model seeds are averaged inside each held-out split "
        r"first, so a cell's score rests on splits and not on (split, seed) pairs. "
        r"$\dagger$ marks the dial positions within one standard error of the leader: "
        r"they are not distinguishable, and an argmax taken out of that set would be "
        + level_note + r"}",
        r"\label{tab:alphachoice}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"\textbf{Competitor} & \textbf{Mean rank} & \textbf{SE} & "
        r"\textbf{Cells} & \textbf{Firsts} \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def _cells(r):
    """The five formatted numbers a confirmation row shows, missing ones as `--`.

    `r["rank"]` and never `r.rank`: on a Series that attribute is the ranking METHOD,
    and a comparison against it formats a bound method instead of failing.
    """
    fmt = lambda v, spec: "--" if not np.isfinite(v) else format(v, spec)
    adv = ("--" if not np.isfinite(r["d"])
           else f"{r['d']:+.3f} [{int(r['won'])}/{int(r['n'])}]")
    rank = ("--" if not np.isfinite(r["rank"])
            else f"{int(r['rank'])} of {int(r['n_alphas'])}")
    return (fmt(r["value"], ".3f"), fmt(r["base"], ".3f"), adv, rank,
            fmt(r["optimism"], "+.3f"))


def tex_confirmation(cf, alpha):
    body = []
    for _, r in cf.iterrows():
        val, base, adv, rank, opt = _cells(r)
        body.append(f"{cell_label(r)} & {r['metric']} & {val} & {base} & {adv} & "
                    f"{rank} & {opt} \\\\")
    return "\n".join([
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{\textbf{The chosen $\alpha$ read once on test.} "
        rf"$\alpha = {alpha:g}$ was fixed on validation (Table~\ref{{tab:alphachoice}}) "
        r"and is reported here without further tuning. \emph{vs base} is the paired "
        r"difference against the boosting head on the same held-out splits, with the "
        r"number of splits won in brackets. \emph{Optimism} is what choosing on these "
        r"very folds would have added: the gap between the luckiest dial position on "
        r"them and the one validation picked. It is reported so that the protocol is "
        r"auditable rather than asserted.}",
        r"\label{tab:alphaconfirm}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}ll rr r rr@{}}", r"\toprule",
        r"\textbf{Cell} & \textbf{Metric} & \textbf{Ours} & \textbf{Base} & "
        r"\textbf{vs base} & \textbf{Rank among $\alpha$} & \textbf{Optimism} \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def text(rk, cf, lo, alpha):
    out = ["", "=== SELECTION (validation): mean rank across cells, 1 = best",
           f"{'competitor':<14}{'mean rank':>10}{'SE':>7}{'cells':>7}{'firsts':>8}"
           f"{'1-SE tie':>10}", "-" * 56]
    for _, r in rk.iterrows():
        se = "--" if not np.isfinite(r["se"]) else format(r["se"], ".2f")
        out.append(f"{r['competitor']:<14}{r['mean_rank']:>10.2f}{se:>7}"
                   f"{int(r['cells']):>7}{int(r['best']):>8}"
                   f"{'yes' if r['tied'] else '':>10}")
    out += ["", f"=== CONFIRMATION (test), alpha = {alpha:g}",
            f"{'cell':<46}{'metric':>9}{'ours':>8}{'base':>8}{'vs base':>18}"
            f"{'rank':>12}{'optimism':>10}", "-" * 111]
    for _, r in cf.iterrows():
        val, base, adv, rank, opt = _cells(r)
        out.append(f"{cell_label(r):<46}{r['metric']:>9}{val:>8}{base:>8}"
                   f"{adv:>18}{rank:>12}{opt:>10}")
    if len(lo):
        out += ["", "=== ROBUSTNESS: choose on two panels, look at the third",
                lo.round(3).to_string(index=False)]
    out += ["", "`optimism` is the gap between the best alpha ON THESE FOLDS and the "
            "one validation", "picked. Quoting the first instead is selection on the "
            "test set."]
    return "\n".join(out)


# ------------------------------------------------------------------ driver

def run_name(root):
    """The run a root stands for: its own directory name. `results/graph/v13_esm3` is
    the run `v13_esm3`, and that is what the output directory is named after, so two
    runs rendered on the same day cannot land on top of each other."""
    root = str(root).replace("\\", "/").rstrip("/")
    return pathlib.Path(root).name or "alpha_choice"


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", required=True,
                    help="the sweep root to render, e.g. results/graph/v13_esm3. "
                         "REQUIRED and deliberately without a default: one run, one "
                         "table, chosen here rather than assumed")
    ap.add_argument("--nodes", default="nodedial",
                    help="which dial; the two must never share a table")
    ap.add_argument("--mol-source", nargs="+", default=None)
    ap.add_argument("--dataset", nargs="+", default=None)
    ap.add_argument("--regime", nargs="+", default=None)
    ap.add_argument("--combo", default="cls+mol")
    ap.add_argument("--metric", default=None,
                    help="override the metric of record; a choice that moves when the "
                         "metric changes is inside the noise")
    ap.add_argument("--at-alpha", type=float, default=None,
                    help="confirm THIS alpha instead of the one validation picked "
                         "(for the sensitivity paragraph, not for the table)")
    ap.add_argument("--out", default=None,
                    help="default: results/article_tables/<run>/alpha_choice, named "
                         "after the sweep root, so two runs never overwrite each "
                         "other's table")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    kw = dict(root=a.sweep_root, nodes=a.nodes, mol_source=a.mol_source,
              dataset=a.dataset, regime=a.regime, combo=a.combo)
    try:
        val = ag.load(split="val", **kw)
    except SystemExit as e:
        print(f"no validation rows under {a.sweep_root}: {e}\n\n"
              f"This table cannot be built from test alone -- that is the point of it.\n"
              f"Score the validation split of THIS run, then re-run this script:\n"
              f"    .venv/bin/python scripts/analysis/val_rescore.py "
              f"--root {a.sweep_root}\n")
        return 1
    test = ag.load(split="test", **kw)

    metric_of = ac._metric_of(val, argparse.Namespace(metric=a.metric))
    rk = selection(val, metric_of)
    gate = rk[rk.alpha.notna()]
    if gate.empty:
        print("the validation frame holds no dial arm -- nothing to choose between")
        return 1
    alpha = a.at_alpha if a.at_alpha is not None else float(gate.iloc[0]["alpha"])
    cf = confirmation(test, ac._metric_of(test, argparse.Namespace(metric=a.metric)),
                      alpha)
    lo = robustness(val, metric_of)

    note = (r"noise-chasing. Among a tied set the dial position to report is the one "
            r"with an argument behind it, not the one with the smallest number.")
    out = tk.out_dir(a.out or f"results/article_tables/{run_name(a.sweep_root)}/"
                              f"alpha_choice")
    rk.to_csv(out / "alpha_choice_rank.csv", index=False)
    cf.to_csv(out / "alpha_choice_confirm.csv", index=False)
    if len(lo):
        lo.to_csv(out / "alpha_choice_loo.csv", index=False)
    (out / "alpha_choice.tex").write_text(
        tex_selection(rk, note) + "\n\n" + tex_confirmation(cf, alpha) + "\n",
        encoding="utf-8")
    print(text(rk, cf, lo, alpha))
    src = "chosen on validation" if a.at_alpha is None else "asked for on the command line"
    print(f"\nalpha = {alpha:g} ({src})"
          f"\nrun {run_name(a.sweep_root)}, read from {a.sweep_root}, "
          f"nodes={a.nodes}, combo={a.combo}"
          f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
