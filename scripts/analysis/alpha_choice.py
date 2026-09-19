#!/usr/bin/env python
"""Which alpha do we report as primary? -- the decision, as one command.

    python scripts/analysis/alpha_choice.py                    # every cell on disk
    python scripts/analysis/alpha_choice.py --mol-source chemberta
    python scripts/analysis/alpha_choice.py --metric Spearman  # does the choice move?
    python scripts/analysis/alpha_choice.py --loo              # the stability check

WHAT A CELL IS. The paper's graph-vs-boost comparison is three tables -- M2OR, Carey,
Hallem-Carlson -- and each has a row per (regime x molecule source). One CELL here is
one of those rows: (dataset, regime, mol_source), scored on that dataset's metric of
record (R2 on the continuous insect panels, AUROC on the binary pool) averaged over its
five splits. Three sources and two regimes make eighteen cells.

TWO CRITERIA, AND WHY BOTH.

  mean rank   Every competitor -- each alpha, plus `boost` and `legacy` -- is ranked
              WITHIN each cell, and the ranks are averaged across cells. This is the
              only summary that is not a unit error: R2 on Carey and AUROC on M2OR
              cannot be averaged, but their ORDERINGS can. `boost` is in the ranking on
              purpose, so "which row of the table wins on average" is a number and not
              a reading of three tables by eye.
  advantage   The paired difference against boost, cell by cell. Reported per table in
              its own units (never pooled across tables) and, pooled, only as `dz` --
              the difference divided by its own spread across splits, which is
              dimensionless. `won` counts the cells in favour.

THE TRAP THIS SCRIPT IS BUILT AROUND. Choosing alpha by the score on the very folds the
paper reports is selection on the test set: the winner's margin is partly the luck of
those folds, and quoting it as if alpha had been fixed in advance overstates it. Three
things keep that honest here, and none of them is optional:

  1. THE 1-SE RULE. Alphas whose mean rank is within one standard error of the best are
     statistically indistinguishable from it. Picking the argmax out of a flat set is
     noise-chasing. Among the tied set, pick the alpha you would defend on grounds that
     are not this table -- and alpha=1 is the one with an argument behind it: no protein
     embedding enters the model anywhere, which is the claim the paper is making.
  2. LEAVE ONE TABLE OUT (`--loo`). Choose on two datasets, look at where that alpha
     lands on the third. If the choice does not survive that, it is a property of the
     folds and not of the method.
  3. A SECOND METRIC (`--metric`). If the argmax moves when R2 becomes Spearman, or
     AUROC becomes AUPRC, the ordering is inside the noise.

The scoreboard of record stays `headline_table.py`; the shape of the dial is
`alpha_grid.py` and the notebook. This file only answers "which column do we print".
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.analysis import alpha_grid as ag                    # noqa: E402

NL = chr(10)
# One row of one of the three tables.
CELL = ["dataset", "regime", "mol_source"]
DATASET_LABEL = {"cc": "Carey", "hc": "Hallem", "m2or": "M2OR",
                 "cc_shrinked": "Carey-shrunk", "hc_shrinked": "Hallem-shrunk",
                 "cc_shrinked50": "Carey-shrunk50",
                 "hc_shrinked50": "Hallem-shrunk50"}
# Competitors that are not the dial. `naive` is deliberately absent: it is the floor a
# table quotes, not a candidate to report.
REF_ARMS = {"boost_full": "boost", "graph_legacy": "legacy"}


def _metric_of(df, args):
    """Metric per dataset. `--metric` overrides, and is refused where the task does not
    emit it rather than silently ranking a column of NaN."""
    out = {}
    for ds in sorted(df.dataset.unique()):
        m = args.metric or ag.OF_RECORD[ag.TASK[ds]]
        have = ag.metrics_available(df, ds)
        if m not in have:
            raise SystemExit(f"{ds}: {m!r} is not written for a {ag.TASK[ds]} task; "
                             f"have {have}")
        out[ds] = m
    return out


def scores(df, metric_of):
    """Per (cell, competitor): the mean over the cell's splits, and n.

    Model seeds are averaged inside each fold before anything else, so `n` counts FOLDS
    -- the independent unit -- and five seeds agreeing on one fold stay one observation.

    The competitors are every alpha the sweep ran plus boost and legacy. A cell that is
    missing one of them is dropped from that competitor's rank -- not filled in -- and
    `coverage` below reports how ragged that makes the comparison, because a mean rank
    taken over a different set of cells for each row is not a comparison at all."""
    rows = []
    for key, g in df.groupby(CELL, sort=True):
        cell = dict(zip(CELL, key))
        m = metric_of[cell["dataset"]]
        for arm, label in REF_ARMS.items():
            q = g[g.arm == arm]
            if q.empty:
                continue
            mu, _, n = ag.ci_folds(q, m)
            rows.append(cell | dict(competitor=label, alpha=np.nan, value=mu, n=n))
        for a, q in g[g.arm == "gate"].groupby("alpha"):
            mu, _, n = ag.ci_folds(q, m)
            rows.append(cell | dict(competitor=f"a={a:g}", alpha=float(a),
                                    value=mu, n=n))
    return pd.DataFrame(rows)


def ranked(sc):
    """Mean rank across cells, 1 = best.

    Every metric of record is higher-is-better, so the rank is descending. Ties take the
    average rank, which is what keeps two identical arms from splitting a place between
    them by row order."""
    sc = sc.assign(rank=sc.groupby(CELL)["value"]
                   .rank(ascending=False, method="average"))
    out = []
    for c, g in sc.groupby("competitor", sort=False):
        mu, hw, n = ag.ci(g["rank"])
        se = float(g["rank"].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
        out.append(dict(competitor=c, alpha=g["alpha"].iloc[0], mean_rank=mu,
                        se=se, hw=hw, cells=n, best=int((g["rank"] == 1).sum())))
    return pd.DataFrame(out).sort_values("mean_rank").reset_index(drop=True)


def advantage(df, metric_of, ref_arm="boost_full"):
    """Per (cell, alpha): the paired difference against boost, and its spread.

    Paired on (fold, seed) -- both arms ran on the same split with the same draw, so the
    difference removes the split and the draw at once -- and then averaged over model
    seeds within each fold, so `n` is folds and not (fold, seed) cells. `dz` is that
    difference in units of its own across-FOLD SD: dimensionless, so it is the only
    form in which Carey's R2 and M2OR's AUROC can be put in one column."""
    rows = []
    for key, g in df.groupby(CELL, sort=True):
        cell = dict(zip(CELL, key))
        m = metric_of[cell["dataset"]]
        # pair on (fold, seed), THEN average the seeds inside each fold: the model
        # noise cancels in the difference and what survives is one advantage per
        # held-out set, which is the sample the interval is over
        ref = (g[g.arm == ref_arm].set_index(ag.SPLIT)[m]
               .pipe(pd.to_numeric, errors="coerce"))
        ref = ref[~ref.index.duplicated()]
        if ref.empty:
            continue
        for a, q in g[g.arm == "gate"].groupby("alpha"):
            v = pd.to_numeric(q.set_index(ag.SPLIT)[m], errors="coerce")
            d = (v - ref.reindex(v.index)).dropna()
            if isinstance(d.index, pd.MultiIndex):
                d = d.groupby(level="fold").mean()
            if d.empty:
                continue
            sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
            rows.append(cell | dict(
                alpha=float(a), d=float(d.mean()), sd=sd, n=len(d),
                won=int((d > 0).sum()),
                dz=float(d.mean() / sd) if sd and sd > 1e-12 else np.nan))
    return pd.DataFrame(rows)


def one_se(rk):
    """The alphas indistinguishable from the best by mean rank.

    Everything within one standard error of the leader. This is the standard rule for
    refusing to read an ordering that the data does not support, and here it is the
    difference between "alpha=0.85 wins" and "everything from 0.6 up is the same, so
    pick the end of the dial, which is also the claim"."""
    gate = rk[rk.alpha.notna()]
    if gate.empty:
        return gate, np.nan
    lead = gate.iloc[0]
    cut = lead.mean_rank + (lead.se if np.isfinite(lead.se) else 0.0)
    return gate[gate.mean_rank <= cut], cut


def chosen_alpha(df, metric_of):
    """The dial position the leader board picks, and the set it is tied with.

    The leader alone is not the answer -- `one_se` exists because an argmax out of a
    flat set is noise-chasing -- so both come back and the tied set is printed. What
    the leader IS good for is the confirmation slice below: SOME alpha has to be
    carried to test, and carrying the val argmax is the strictest version of the
    procedure (no human eye in the loop between the two splits)."""
    rk = ranked(scores(df, metric_of))
    gate = rk[rk.alpha.notna()]
    if gate.empty:
        return None, gate
    tied, _ = one_se(rk)
    return float(gate.iloc[0].alpha), tied


def confirm(val_df, test_df, args):
    """Report the chosen alpha on TEST -- the only place test is allowed to be read.

    Two numbers matter and both are printed. The first is what the paper would quote.
    The second is the OPTIMISM AVOIDED: how much better the best-on-test alpha looks
    than the one val picked. That gap is exactly what choosing on test would have
    silently added to the claim, and printing it is the difference between a protocol
    and an assertion that one was followed."""
    metric_of = _metric_of(test_df, args)
    a = args.at_alpha if args.at_alpha is not None else chosen_alpha(val_df, metric_of)[0]
    if a is None:
        print(f"{NL}  nothing to confirm: the val frame holds no gate arm")
        return
    src = "you asked for it" if args.at_alpha is not None else "chosen on val"
    print(f"{NL}{'=' * 78}{NL}CONFIRMATION SLICE: alpha = {a:g} ({src}), read on TEST"
          f"{NL}{'=' * 78}")
    sc = scores(test_df, metric_of)
    adv = advantage(test_df, metric_of)
    lab = f"a={a:g}"
    print(f"{NL}  {'cell':<34}{'metric':>9}{'value':>9}{'vs boost':>19}"
          f"{'rank on test':>14}{'best there':>12}")
    print("  " + "-" * 95)
    for key, g in sc.groupby(CELL, sort=True):
        cell = dict(zip(CELL, key))
        name = (f"{DATASET_LABEL.get(cell['dataset'], cell['dataset'])}/"
                f"{cell['regime'][:5]}/{cell['mol_source']}")
        m = metric_of[cell["dataset"]]
        row = g[g.competitor == lab]
        if row.empty:
            print(f"  {name:<34}{m:>9}{'--':>9}{'not run here':>19}")
            continue
        v = float(row.iloc[0].value)
        gate = g[g.alpha.notna()].sort_values("value", ascending=False)
        rank = int((gate.value > v).sum()) + 1
        best = float(gate.iloc[0].value)
        d = adv[(adv.dataset == cell["dataset"]) & (adv.regime == cell["regime"])
                & (adv.mol_source == cell["mol_source"])
                & np.isclose(adv.alpha.astype(float), a)]
        dtxt = ("--" if d.empty else
                f"{float(d.iloc[0].d):+.3f} [{int(d.iloc[0].won)}/{int(d.iloc[0].n)}]")
        print(f"  {name:<34}{m:>9}{v:>9.3f}{dtxt:>19}"
              f"{f'{rank} of {len(gate)}':>14}{best - v:>+12.3f}")
    print(f"{NL}  `best there` is the OPTIMISM AVOIDED: how much the luckiest alpha on "
          f"these very folds{NL}  outscores the one val picked. Quoting that instead "
          f"would be selection on the test set.")


def loo(df, metric_of, args):
    """Choose on two tables, look at where that alpha lands on the third.

    A choice that does not survive this is a property of the folds, not of the method --
    and it is the cheapest honest check available, because the three datasets are
    genuinely different panels and not three resamples of one."""
    out = []
    dsets = sorted(df.dataset.unique())
    for held in dsets:
        tr = df[df.dataset != held]
        te = df[df.dataset == held]
        if tr.empty or te.empty:
            continue
        pick = ranked(scores(tr, metric_of))
        pick = pick[pick.alpha.notna()]
        te_rank = ranked(scores(te, metric_of))
        te_rank = te_rank[te_rank.alpha.notna()].set_index("alpha")
        if pick.empty or te_rank.empty:
            continue
        a = float(pick.iloc[0].alpha)
        out.append(dict(held_out=DATASET_LABEL.get(held, held),
                        chosen_on_other_two=a,
                        its_rank_on_held_out=float(te_rank.loc[a, "mean_rank"])
                        if a in te_rank.index else np.nan,
                        best_alpha_there=float(te_rank["mean_rank"].idxmin()),
                        best_rank_there=float(te_rank["mean_rank"].min())))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------- reporting

def report(df, args):
    metric_of = _metric_of(df, args)
    sc = scores(df, metric_of)
    rk = ranked(sc)
    adv = advantage(df, metric_of)
    n_cells = sc.groupby(CELL).ngroups

    tables = ", ".join(f"{DATASET_LABEL.get(d, d)}={m}" for d, m in metric_of.items())
    print("=" * 100)
    print(f"=== WHICH ALPHA -- {n_cells} cells "
          f"({df.dataset.nunique()} tables x {df.regime.nunique()} regimes "
          f"x {df.mol_source.nunique()} sources), {tables}")
    print("=" * 100)

    ragged = rk[rk.cells != rk.cells.max()]
    if len(ragged):
        print(f"  !! {len(ragged)} competitor(s) are scored on fewer cells than the "
              f"others -- the mean ranks below are NOT over the same set:")
        for r in ragged.itertuples():
            print(f"     {r.competitor:<10}{r.cells}/{rk.cells.max()} cells")
        print()

    dsets = sorted(metric_of)
    head = (f"  {'':<10}{'mean rank':>11}{'SE':>7}{'best in':>9}"
            f"{'cells won':>11}{'dz med':>8}   "
            + "".join(f"{DATASET_LABEL.get(d, d):>10}" for d in dsets))
    print(head)
    print(f"  {'':<10}{'1 = best':>11}{'':>7}{'of ' + str(n_cells):>9}"
          f"{'vs boost':>11}{'d / sd':>8}   "
          + "".join(f"{'mean d':>10}" for _ in dsets))
    print("  " + "-" * (len(head) - 2))
    per_ds = (adv.groupby(["alpha", "dataset"])["d"].mean().unstack("dataset")
              if len(adv) else pd.DataFrame())
    # cells in favour, not splits: one row of one table is the unit these three
    # tables are read in, and 49/90 splits would answer a question nobody asked
    cellw = (adv.assign(w=adv.d > 0).groupby("alpha")
             .agg(won=("w", "sum"), cells=("w", "size"))
             if len(adv) else pd.DataFrame())
    # MEDIAN dz across cells, not mean: dz is d / sd(d), and a cell whose five paired
    # differences happen to agree closely has a near-zero denominator that would drag
    # a mean anywhere it liked
    dzm = adv.groupby("alpha")["dz"].median() if len(adv) else pd.Series(dtype=float)
    for r in rk.itertuples():
        a = r.alpha
        if np.isnan(a):
            won = dz = ""
            cols = "".join(f"{'':>10}" for _ in dsets)
            won = dz = ""
        else:
            won = (f"{int(cellw.loc[a, 'won'])}/{int(cellw.loc[a, 'cells'])}"
                   if a in cellw.index else "")
            dz = f"{dzm.get(a, np.nan):+.2f}" if a in dzm.index else ""
            cols = "".join(
                f"{per_ds.loc[a, d]:>+10.3f}"
                if (len(per_ds) and a in per_ds.index and d in per_ds.columns
                    and np.isfinite(per_ds.loc[a, d])) else f"{'':>10}"
                for d in dsets)
        se = f"{r.se:.2f}" if np.isfinite(r.se) else "-"
        print(f"  {r.competitor:<10}{r.mean_rank:>11.2f}{se:>7}{r.best:>9}"
              f"{won:>11}{dz:>8}   {cols}")

    tied, cut = one_se(rk)
    gate = rk[rk.alpha.notna()]
    print()
    if not gate.empty:
        lead = gate.iloc[0]
        print(f"  BEST by mean rank: {lead.competitor} ({lead.mean_rank:.2f})")
        names = ", ".join(f"{a:g}" for a in sorted(tied.alpha))
        print(f"  WITHIN 1 SE of it ({cut:.2f}): {names}")
        if len(tied) > 1:
            print("    -> those are not distinguishable. Picking the argmax out of a "
                  "flat set is\n"
                  "       noise-chasing; pick the one with an argument behind it. "
                  "alpha=1 is the end\n"
                  "       of the dial: no protein embedding enters the model anywhere.")
        boost = rk[rk.competitor == "boost"]
        if len(boost):
            b = boost.iloc[0].mean_rank
            beat = gate[gate.mean_rank < b]
            print(f"  boost sits at mean rank {b:.2f}; "
                  f"{len(beat)}/{len(gate)} alphas rank above it"
                  + (f" (from alpha={min(beat.alpha):g} up)" if len(beat) else ""))

    if args.loo:
        print()
        print("  LEAVE ONE TABLE OUT -- choose on two, look at the third")
        print("  " + "-" * 60)
        L = loo(df, metric_of, args)
        if L.empty:
            print("    not enough datasets on disk for this check")
        else:
            print(L.to_string(index=False).replace("\n", "\n    ").rjust(4))
            spread = L.chosen_on_other_two.nunique()
            print(f"\n    the choice is {'STABLE' if spread == 1 else 'NOT stable'} "
                  f"across the three holdouts "
                  f"({sorted(set(L.chosen_on_other_two))})")

    print()
    print("  READ THIS BEFORE QUOTING A NUMBER")
    print("  " + "-" * 60)
    if getattr(args, "select_on", "test") == "val":
        print("    This ranking is on the VALIDATION rows, which the paper does not")
        print("    report -- so the choice made here is legitimate and the test slice")
        print("    below is the first and only time test is read. What it does NOT buy")
        print("    is a sharper claim: if the 1-SE set has more than one member -- it")
        print("    usually will -- the dial is flat there, and the honest sentence is")
        print("    'alpha was fixed at the end of the dial, and nothing was lost'.")
    else:
        print("    This ranking is computed on the same folds the paper reports, so the")
        print("    winner's margin is partly the luck of those folds. Quoting the best")
        print("    alpha's score as if alpha had been fixed in advance overstates it. "
              "If the")
        print("    1-SE set has more than one member -- it usually will -- say in the "
              "paper")
        print("    that alpha was fixed at the END of the dial on the argument, not "
              "tuned,")
        print("    and let this table be the evidence that nothing was lost by doing so.")
    if not args.metric:
        alt = {"regression": "Spearman", "classification": "AUPRC"}
        tasks = {ag.TASK[d] for d in metric_of}
        sug = ", ".join(sorted({alt[t] for t in tasks}))
        print(f"    Re-run with --metric ({sug}) -- if the argmax moves, the ordering")
        print("    is inside the noise and only the 1-SE set means anything.")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ag.DEFAULT_ROOT)
    ap.add_argument("--mol-source", nargs="+", default=None,
                    help="default: every source on disk, each its own cell")
    ap.add_argument("--nodes", nargs="+", default=["onehot"],
                    choices=["esm", "onehot", "nodedial"],
                    help="which dial. Never two at once: alpha runs structure -> "
                         "function on the v8 gate and function -> structure on the v9 "
                         "node dial, so one axis would carry two meanings")
    ap.add_argument("--select-on", choices=["val", "test"], default="val",
                    help="WHICH SPLIT THE CHOICE IS MADE ON. `val` is the only honest "
                         "one and needs val_rescore.py to have been run; `test` is a "
                         "diagnostic -- it answers 'what would we have picked with "
                         "hindsight', which is a different question and must never be "
                         "the number the paper defends")
    ap.add_argument("--at-alpha", type=float, default=None,
                    help="confirm this alpha on test instead of the one val chose")
    ap.add_argument("--dataset", nargs="+", default=None)
    ap.add_argument("--regime", nargs="+", default=None,
                    choices=["transductive", "inductive"])
    ap.add_argument("--seed", type=int, nargs="+", default=None)
    ap.add_argument("--metric", default=None,
                    help="override the metric of record on BOTH tasks; the robustness "
                         "check, not a preference")
    ap.add_argument("--loo", action="store_true",
                    help="leave one table out: choose on two datasets, report where "
                         "that alpha lands on the third")
    ap.add_argument("--csv", default=None, help="write the per-cell scores here")
    ap.add_argument("--graph-combo", default="cls+mol",
                    choices=["cls+mol", "cls+prot+mol"],
                    help="which boosting head the graph rows are read from")
    a = ap.parse_args()

    kw = dict(root=a.root, mol_source=a.mol_source, nodes=a.nodes,
              dataset=a.dataset, regime=a.regime, seed=a.seed, combo=a.graph_combo)
    try:
        df = ag.load(split=a.select_on, **kw)
    except SystemExit:
        if a.select_on != "val":
            raise
        raise SystemExit(
            f"no val_metrics_*.csv under {a.root} -- the sweep writes TEST only."
            f"{NL}Produce them first (one XGBoost refit per cell, no graph training):"
            f"{NL}    python scripts/analysis/val_rescore.py --root {a.root} --dry-run"
            f"{NL}    python scripts/analysis/val_rescore.py --root {a.root}"
            f"{NL}Or pass --select-on test, which answers a different question: what "
            f"hindsight would have picked.")
    print(f"{NL}  ALPHA IS BEING CHOSEN ON: {a.select_on.upper()}"
          + ("" if a.select_on == "val" else
             "   <-- these are the rows the paper reports. This is the "
             "hindsight view, not a choice."))
    report(df, a)
    if a.select_on == "val":
        confirm(df, ag.load(split="test", **kw), a)
    if a.csv:
        metric_of = _metric_of(df, a)
        # drop the duplicate split count the merge would otherwise emit as n_x/n_y
        adv = advantage(df, metric_of).drop(columns=["n"])
        out = scores(df, metric_of).merge(adv, on=CELL + ["alpha"], how="left")
        out.rename(columns={"n": "n_splits"}).to_csv(a.csv, index=False)
        print(f"\n  per-cell scores -> {a.csv}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
