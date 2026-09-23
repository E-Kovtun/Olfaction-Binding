#!/usr/bin/env python
"""NOT USED IN THE PAPER (decided 22.09.2026). Kept, not deleted.

The alpha argument is made by a FIGURE, not by this table:
`notebooks/article_figures/alpha_rank_dial.ipynb` draws mean rank against the dial for
both boosting heads and fits a line through it, and the slope test there answers the
same question -- is the dial a slope or a flat line -- in a form a reader can check by
eye. Three typeset panels of ranks answered it in numbers nobody was going to read.

Nothing here is wrong and nothing here is deprecated code: the protocol it implements
(choose on validation, read test once, print the optimism) is still the protocol. It
stays on disk because the figure's aggregation and this file's come from the same
module, so if the figure is ever challenged this is the long form of the answer. It is
not in the README runbook on purpose.

--- what it does, if you do run it -------------------------------------------------

The alpha we report, as a table: chosen on validation, confirmed once on test.

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

BOTH HEADS, SIDE BY SIDE. The sweep fits two boosting heads on the SAME trained
graph: `cls+mol`, where the refined receptor REPLACES the protein vector, and
`cls+prot+mol`, where it is added beside it. They are two readings of one run, they can
prefer different dial positions, and a choice defended on one of them while the paper
reports the other is not a protocol. So every section below is rendered per head, and
`--at-alpha` confirms one alpha in both -- which is how a single reported alpha is
defended.

WHAT IT PRINTS, IN THE ORDER THE ARGUMENT NEEDS IT

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
  dial trend   Is the dial a slope at all? Spearman between the dial position and
               the score, computed INSIDE each cell (the only place alpha and an R2 or
               an AUROC are commensurable), with the cells as the sample: the mean
               correlation, its interval over cells, and a one-sample t-test of the
               null that it is zero. A flat dial is a result -- it is what licenses
               reporting the end of the scale rather than an argmax -- and this is the
               number that says so instead of the eye.

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
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tablekit as tk  # noqa: E402

sys.path.insert(0, str(tk.ROOT))
from scripts.analysis import alpha_choice as ac   # noqa: E402
from scripts.analysis import alpha_grid as ag     # noqa: E402

#: The boosting heads the sweep fits on one trained graph, rendered as one panel
#: each. `cls+mol` is the construction the paper reports; `cls+prot+mol` adds the
#: refined receptor to what the base already reads, so its features are NESTED in the
#: base's and beating the base there is a much weaker statement.
COMBOS = ("cls+mol", "cls+prot+mol")


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


def dial_trend(val_df, metric_of):
    """Is the dial a slope or a flat line? Spearman between alpha and the score.

    Computed WITHIN each cell -- alpha against that cell's own metric, the only place
    the two are commensurable -- and the cells are then the sample. The headline is the
    mean correlation across cells with a t interval over them, because the unit of
    evidence here is the same as everywhere else in this paper: the held-out cell, not
    the (cell, alpha) pair. A single pooled Spearman over every (cell, alpha) point
    would treat scores that are ranked against each other inside a cell as independent
    observations and hand back an interval far too narrow to mean anything.

    Returns (per-cell frame, summary dict). `rho` is against the SCORE, so a positive
    rho means the metric improves as alpha rises; the correlation against the PLACE in
    the table is the same number with the opposite sign.
    """
    sc = ac.scores(val_df, metric_of)
    rows = []
    for key, g in sc.groupby(ac.CELL, sort=True):
        cell = dict(zip(ac.CELL, key))
        g = g[g["alpha"].notna()]
        if g["alpha"].nunique() < 3:
            continue                      # two dial points cannot show a trend
        x = g["alpha"].astype(float).to_numpy()
        y = g["value"].astype(float).to_numpy()
        rho, p = stats.spearmanr(x, y)
        if not np.isfinite(rho):
            continue                      # a cell where every alpha scored the same
        rows.append(cell | dict(metric=metric_of[cell["dataset"]],
                                n_alphas=int(g["alpha"].nunique()),
                                rho=float(rho), p_cell=float(p)))
    per_cell = pd.DataFrame(rows)
    if per_cell.empty:
        return per_cell, {}

    r = per_cell["rho"].to_numpy()
    mu, hw, n = ag.ci(pd.Series(r))
    if n > 1:
        t, p = stats.ttest_1samp(r, 0.0)
        t, p = float(t), float(p)
    else:
        t = p = np.nan
    lo, hi = mu - hw, mu + hw
    if not np.isfinite(p):
        verdict = "one cell only -- no test"
    elif lo <= 0.0 <= hi:
        verdict = "indistinguishable from zero"
    else:
        verdict = "rises with alpha" if mu > 0 else "falls with alpha"
    return per_cell, dict(cells=int(n), rho=float(mu), lo=float(lo), hi=float(hi),
                          t=t, p=p, pos=int((r > 0).sum()), neg=int((r < 0).sum()),
                          verdict=verdict)


# ------------------------------------------------------------------ rendering

def slug(combo):
    """A filename and a LaTeX label cannot hold `+`."""
    return combo.replace("+", "")


def trend_sentence(tr):
    """The Spearman result as one sentence, for a caption or a terminal line."""
    if not tr:
        return "The dial has too few positions in these cells to test for a trend."
    body = (f"Across the {tr['cells']} cells the Spearman correlation between the dial "
            f"position and the score averages {tr['rho']:+.2f} "
            f"(95\\% CI {tr['lo']:+.2f} to {tr['hi']:+.2f}, "
            + ("$t$ and $p$ undefined on one cell"
               if not np.isfinite(tr["p"]) else
               f"$t = {tr['t']:.2f}$, $p = {tr['p']:.2f}$")
            + f"; {tr['pos']} cells positive, {tr['neg']} negative)")
    tail = {"indistinguishable from zero":
            " --- indistinguishable from zero, so the dial is a flat line and not a "
            "slope",
            "rises with alpha": " --- the score rises along the dial",
            "falls with alpha": " --- the score falls along the dial"}
    return body + tail.get(tr["verdict"], "") + "."


def tex_selection(rk, level_note, combo, tr):
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
        r"\caption{\textbf{Choosing $\alpha$ on validation, head "
        rf"\texttt{{{combo}}}.}} Every competitor --- each "
        r"dial position, the boosting base and the legacy graph --- is ranked "
        r"\emph{within} each cell (one cell = one row of the main tables: dataset "
        r"$\times$ regime $\times$ molecule source), and the ranks are averaged across "
        r"cells. Ranks rather than means because $R^2$ and AUROC cannot be averaged but "
        r"their orderings can. Model seeds are averaged inside each held-out split "
        r"first, so a cell's score rests on splits and not on (split, seed) pairs. "
        r"$\dagger$ marks the dial positions within one standard error of the leader: "
        r"they are not distinguishable, and an argmax taken out of that set would be "
        + level_note + " " + trend_sentence(tr) + "}",
        rf"\label{{tab:alphachoice-{slug(combo)}}}",
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


def tex_confirmation(cf, alpha, combo):
    body = []
    for _, r in cf.iterrows():
        val, base, adv, rank, opt = _cells(r)
        body.append(f"{cell_label(r)} & {r['metric']} & {val} & {base} & {adv} & "
                    f"{rank} & {opt} \\\\")
    return "\n".join([
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{\textbf{The chosen $\alpha$ read once on test, head "
        rf"\texttt{{{combo}}}.}} "
        rf"$\alpha = {alpha:g}$ was fixed on validation "
        rf"(Table~\ref{{tab:alphachoice-{slug(combo)}}}) "
        r"and is reported here without further tuning. \emph{vs base} is the paired "
        r"difference against the boosting head on the same held-out splits, with the "
        r"number of splits won in brackets. \emph{Optimism} is what choosing on these "
        r"very folds would have added: the gap between the luckiest dial position on "
        r"them and the one validation picked. It is reported so that the protocol is "
        r"auditable rather than asserted.}",
        rf"\label{{tab:alphaconfirm-{slug(combo)}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}ll rr r rr@{}}", r"\toprule",
        r"\textbf{Cell} & \textbf{Metric} & \textbf{Ours} & \textbf{Base} & "
        r"\textbf{vs base} & \textbf{Rank among $\alpha$} & \textbf{Optimism} \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def text(panels):
    """Every head, one block per section, so the two are read side by side."""
    out = ["", "=== SELECTION (validation): mean rank across cells, 1 = best"]
    for p in panels:
        out += [f"", f"--- head {p['combo']}",
                f"{'competitor':<14}{'mean rank':>10}{'SE':>7}{'cells':>7}"
                f"{'firsts':>8}{'1-SE tie':>10}", "-" * 56]
        for _, r in p["rk"].iterrows():
            se = "--" if not np.isfinite(r["se"]) else format(r["se"], ".2f")
            out.append(f"{r['competitor']:<14}{r['mean_rank']:>10.2f}{se:>7}"
                       f"{int(r['cells']):>7}{int(r['best']):>8}"
                       f"{'yes' if r['tied'] else '':>10}")

    out += ["", "=== DIAL TREND (validation): Spearman between alpha and the score,",
            "    computed inside each cell, the cells being the sample",
            f"{'head':<16}{'cells':>6}{'mean rho':>10}{'95% CI':>18}{'t':>7}"
            f"{'p':>8}  {'+/-':>5}  verdict", "-" * 82]
    for p in panels:
        tr = p["trend"]
        if not tr:
            out.append(f"{p['combo']:<16}{'':>6}{'':>10}{'':>18}{'':>7}{'':>8}  "
                       f"{'':>5}  too few dial positions")
            continue
        ci = f"[{tr['lo']:+.2f}, {tr['hi']:+.2f}]"
        t = "--" if not np.isfinite(tr["t"]) else format(tr["t"], ".2f")
        pv = "--" if not np.isfinite(tr["p"]) else format(tr["p"], ".3f")
        signs = f"{tr['pos']}/{tr['neg']}"
        out.append(f"{p['combo']:<16}{tr['cells']:>6}{tr['rho']:>+10.2f}{ci:>18}"
                   f"{t:>7}{pv:>8}  {signs:>5}  {tr['verdict']}")
    out += ["", "    rho is against the SCORE: positive means the metric improves as "
            "alpha rises,", "    so the correlation with the PLACE in the table above "
            "is the same number negated.", "    The per-cell values are in "
            "alpha_trend_<head>.csv."]

    for p in panels:
        out += ["", f"=== CONFIRMATION (test), head {p['combo']}, "
                    f"alpha = {p['alpha']:g}",
                f"{'cell':<46}{'metric':>9}{'ours':>8}{'base':>8}{'vs base':>18}"
                f"{'rank':>12}{'optimism':>10}", "-" * 111]
        for _, r in p["cf"].iterrows():
            val, base, adv, rank, opt = _cells(r)
            out.append(f"{cell_label(r):<46}{r['metric']:>9}{val:>8}{base:>8}"
                       f"{adv:>18}{rank:>12}{opt:>10}")

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
    ap.add_argument("--combo", nargs="+", default=list(COMBOS),
                    help="which boosting head(s) to render, each as its own panel. "
                         "The sweep fits both on the SAME trained graph, so they are "
                         "two readings of one run and belong side by side")
    ap.add_argument("--metric", default=None,
                    help="override the metric of record; a choice that moves when the "
                         "metric changes is inside the noise")
    ap.add_argument("--at-alpha", type=float, default=None,
                    help="confirm THIS alpha in every head instead of the one each "
                         "head's validation picked -- which is how a single reported "
                         "alpha is defended, and what the sensitivity paragraph needs")
    ap.add_argument("--out", default=None,
                    help="default: results/article_tables/<run>/alpha_choice, named "
                         "after the sweep root, so two runs never overwrite each "
                         "other's table")
    return ap


def panel(a, combo):
    """One head: its leader board, its chosen alpha, its confirmation, its trend.

    Returns (status, panel). `ok` with the panel; `absent` when the run simply does not
    hold this head, which is a fact about the run and not an error; `refused` when it
    holds it on test but not on validation, which IS an error -- see the docstring.
    """
    kw = dict(root=a.sweep_root, nodes=a.nodes, mol_source=a.mol_source,
              dataset=a.dataset, regime=a.regime, combo=combo)
    try:
        test = ag.load(split="test", **kw)
    except SystemExit:
        print(f"head {combo}: not in this run, skipping")
        return "absent", None
    # a run that never fitted this head still answers with the reference arms, which
    # are head-independent. No dial arm on TEST is therefore "this head is not here",
    # while a dial arm on test and none on validation is the refusal case below.
    if not test["alpha"].notna().any():
        print(f"head {combo}: not in this run, skipping")
        return "absent", None
    try:
        val = ag.load(split="val", **kw)
    except SystemExit as e:
        print(f"no validation rows for head {combo} under {a.sweep_root}: {e}\n\n"
              f"This table cannot be built from test alone -- that is the point of it.\n"
              f"Score the validation split of THIS run, then re-run this script:\n"
              f"    .venv/bin/python scripts/analysis/val_rescore.py "
              f"--root {a.sweep_root}\n")
        return "refused", None

    metric_of = ac._metric_of(val, argparse.Namespace(metric=a.metric))
    rk = selection(val, metric_of)
    gate = rk[rk.alpha.notna()]
    if gate.empty:
        print(f"head {combo}: the validation frame holds no dial arm -- nothing to "
              f"choose between")
        return "refused", None
    alpha = a.at_alpha if a.at_alpha is not None else float(gate.iloc[0]["alpha"])
    cf = confirmation(test, ac._metric_of(test, argparse.Namespace(metric=a.metric)),
                      alpha)
    per_cell, tr = dial_trend(val, metric_of)
    return "ok", dict(combo=combo, rk=rk, cf=cf, alpha=alpha, trend=tr,
                      per_cell=per_cell)


def main(argv=None):
    a = parser().parse_args(argv)
    panels, refused = [], False
    for combo in a.combo:
        status, p = panel(a, combo)
        refused |= status == "refused"
        if p is not None:
            panels.append(p)
    # a head this run never fitted is not a failure; a head whose validation is missing
    # is, and it must not be papered over by the other head having rendered
    if refused or not panels:
        return 1

    note = (r"noise-chasing. Among a tied set the dial position to report is the one "
            r"with an argument behind it, not the one with the smallest number.")
    out = tk.out_dir(a.out or f"results/article_tables/{run_name(a.sweep_root)}/"
                              f"alpha_choice")
    tex = []
    for p in panels:
        tag = slug(p["combo"])
        p["rk"].to_csv(out / f"alpha_choice_rank_{tag}.csv", index=False)
        p["cf"].to_csv(out / f"alpha_choice_confirm_{tag}.csv", index=False)
        if len(p["per_cell"]):
            p["per_cell"].to_csv(out / f"alpha_trend_{tag}.csv", index=False)
        tex += [tex_selection(p["rk"], note, p["combo"], p["trend"]),
                tex_confirmation(p["cf"], p["alpha"], p["combo"])]
    (out / "alpha_choice.tex").write_text("\n\n".join(tex) + "\n", encoding="utf-8")
    print(text(panels))

    src = "chosen on validation" if a.at_alpha is None else "asked for on the command line"
    chosen = ", ".join(f"{p['combo']}: {p['alpha']:g}" for p in panels)
    print(f"\nalpha per head ({src}) -- {chosen}"
          f"\nrun {run_name(a.sweep_root)}, read from {a.sweep_root}, "
          f"nodes={a.nodes}"
          f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
