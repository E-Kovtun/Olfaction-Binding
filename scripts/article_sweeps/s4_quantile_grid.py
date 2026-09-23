#!/usr/bin/env python
"""Reader for the (criterion x quantile) sweep. The notebook draws; this aggregates.

The split of labour is the same one `alpha_grid` has with the dial notebooks, and for
the same reason: the unit of evidence is the HELD-OUT FOLD, and a figure that took its
own mean over the raw rows would treat the (fold, seed) cells of a multi-seed grid as
independent observations -- t(n*k-1) instead of t(n-1), divided by sqrt(n*k) instead of
sqrt(n). Every interval on the page would come out about half of what it should be.

So the averaging here is done once, in `alpha_grid.fold_means` and `alpha_grid.ci`,
which are the repository's definition of that rule. This module only reshapes their
output for an x axis that is the quantile instead of alpha.

    from scripts.article_sweeps import quantile_grid as qg
    df = qg.load(dataset="m2or", regime="inductive")
    qg.curve(df, "AUROC")        # one row per (series, criterion, quantile)
    qg.best_q(df, "AUROC")       # where each criterion peaks, and by how much

There is deliberately no command line here. This sweep is read by eye, in
`notebooks/article_figures/quantile_criteria.ipynb`, and the figure saved from there is
the artifact -- a second text rendering of the same numbers would be one more thing to
keep in agreement with it.
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

from scripts.analysis import alpha_grid as ag   # noqa: E402

DEFAULT_ROOT = "results/article_sweeps/quantile_criteria"
#: The reference arm, written by the sweep with a NaN quantile. It is a horizontal
#: line on every panel, never a curve: it does not depend on the knob.
BOOST = "boost_full"
#: The construction each dataset's reported numbers stand on -- drawn as the marked
#: point so a figure says which grid cell the paper actually used.
PAPER_POINT = {"m2or": ("greedy_pair_cover", 0.99),
               "cc": ("coverage", 0.0), "hc": ("coverage", 0.0)}
LOWER_IS_BETTER = ag.LOWER_IS_BETTER


def sign_of(metric):
    """+1 if larger is better. RMSE and MAE are won by being SMALLER, and a reader
    that forgets it hands every error column to the worse model."""
    return -1.0 if metric in LOWER_IS_BETTER else 1.0


# ------------------------------------------------------------------ loading

def load(root=DEFAULT_ROOT, dataset=None, regime=None, mol_source=None,
         combo="cls+mol", split="test", seeds=None, drop_failed=True):
    """Every CSV under `root`, filtered and given a `series` label.

    `drop_failed` removes the NaN rows the sweep writes for a cell it could not
    compute (a degenerate graph at a tiny K). They are kept on disk on purpose -- the
    hole is a finding about that construction, not a gap in the run -- but they must
    not reach an aggregate, where a NaN would silently shorten an interval's n.
    """
    d = pathlib.Path(root)
    if not d.is_absolute():
        d = _root / d
    files = sorted(d.glob("metrics_*.csv"))
    if not files:
        raise SystemExit(f"no metrics_*.csv under {d} -- run "
                         f"scripts/article_sweeps/s4_run_quantile_criteria.py first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col, want in (("dataset", dataset), ("regime", regime),
                      ("mol_source", mol_source)):
        if want is not None:
            df = df[df[col].isin([want] if isinstance(want, str) else list(want))]
    if split is not None and "split" in df.columns:
        df = df[df.split == split]
    if seeds is not None:
        df = df[df.seed.isin(list(seeds))]
    # The head filter must not drop the reference arm, which has its own combo.
    if combo is not None:
        df = df[(df.combo == combo) | (df.criterion == BOOST)]
    if drop_failed and "status" in df.columns:
        df = df[df.status.astype(str) == "ok"]
    df = df.assign(series=df.dataset + " / " + df.regime)
    df.attrs["files"] = [str(f) for f in files]
    return df.reset_index(drop=True)


def graph_rows(df):
    return df[df.criterion != BOOST]


def metric_for(df):
    """The metric of record per dataset, as the rest of the repository defines it."""
    return {ds: ag.OF_RECORD[ag.TASK[ds]] for ds in sorted(df.dataset.unique())}


def metrics_available(df, dataset=None, which="headline"):
    return [m for m in ag.metrics_available(df, dataset=dataset, which=which)
            if m in df.columns]


def coverage(df):
    """What is on disk: cells per series, and how many of them failed."""
    g = graph_rows(df)
    out = g.groupby("series").agg(
        criteria=("criterion", "nunique"), quantiles=("quantile", "nunique"),
        folds=("fold", "nunique"), seeds=("seed", "nunique"), cells=("K", "size"))
    return out


# ------------------------------------------------------------------ aggregation

def _agg(df, metric, by, level=0.95):
    """mean +- Student-t half-width over FOLD MEANS. The one aggregation in this file.

    Model seeds are averaged inside each fold first (`alpha_grid.fold_means`); the
    interval is then over folds, which is the only unit an inference may be made over.
    """
    by = list(by)
    if metric not in df.columns or df.empty:
        return _empty(by)
    d = df.assign(**{metric: pd.to_numeric(df[metric], errors="coerce")}).dropna(
        subset=[metric])
    if d.empty:
        return _empty(by)
    f = ag.fold_means(d, metric, by)
    rows = []
    for k, g in f.groupby(by, dropna=False, sort=True):
        m, hw, n = ag.ci(g[metric], level)
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)))
                    | dict(mean=m, hw=hw, lo=m - hw, hi=m + hw, n_folds=n,
                           seeds=float(pd.to_numeric(g["_nseed"],
                                                     errors="coerce").mean())))
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def _empty(by):
    """Typed, not merely empty: an object-dtype frame reaches `fill_between` as
    `object[]` and dies on `isfinite` -- which is what a half-finished sweep looks
    like, i.e. exactly when the notebook is most likely to be open."""
    cols = list(by) + ["mean", "hw", "lo", "hi", "n_folds", "seeds"]
    out = pd.DataFrame(columns=cols)
    floats = {c: float for c in ("mean", "hw", "lo", "hi", "seeds", "quantile")
              if c in cols}
    return out.astype(floats | {"n_folds": "int64"})


def curve(df, metric, by=("series", "criterion"), level=0.95):
    """One row per (series, criterion, quantile): the curve the notebook plots."""
    return _agg(graph_rows(df), metric, list(by) + ["quantile"], level)


def levels(df, metric, by=("series",), level=0.95):
    """The reference arm as a scalar per series -- a horizontal line, not a curve."""
    return _agg(df[df.criterion == BOOST], metric, list(by), level)


def k_curve(df):
    """K per (series, quantile): how many molecules the knob actually keeps.

    Its own panel, never a second y axis on the metric plot. Two quantiles that
    resolve to the same K are one experiment drawn twice, and this is where that
    shows up.
    """
    g = graph_rows(df)
    if g.empty:
        return _empty(["series"])
    out = (g.groupby(["series", "quantile"])["K"]
           .agg(["mean", "min", "max"]).reset_index()
           .rename(columns={"mean": "K", "min": "K_min", "max": "K_max"}))
    return out


def delta_vs_boost(df, metric, by=("series", "criterion"), level=0.95):
    """The graph minus the reference, PAIRED INSIDE EACH FOLD.

    The absolute curves show two levels; this shows their difference where it is
    actually defined -- on the same held-out rows. It is the comparison the paper
    makes, and the only one whose interval means what it appears to mean.

    The sign is NOT flipped for error metrics: this returns a difference in the
    metric's own units, so on RMSE a negative value is the graph winning. Anything
    that COUNTS wins must apply `sign_of`.
    """
    by = list(by)
    g, b = graph_rows(df), df[df.criterion == BOOST]
    if metric not in df.columns or g.empty or b.empty:
        return _empty(by + ["quantile"])
    num = lambda d: d.assign(**{metric: pd.to_numeric(d[metric], errors="coerce")})
    gf = ag.fold_means(num(g).dropna(subset=[metric]), metric, by + ["quantile"])
    bf = ag.fold_means(num(b).dropna(subset=[metric]), metric, ["series"])
    m = gf.merge(bf[["series", "fold", metric]].rename(columns={metric: "_ref"}),
                 on=["series", "fold"], how="inner")
    if m.empty:
        return _empty(by + ["quantile"])
    m = m.assign(**{metric: m[metric] - m["_ref"]})
    rows = []
    for k, grp in m.groupby(by + ["quantile"], dropna=False, sort=True):
        mean, hw, n = ag.ci(grp[metric], level)
        rows.append(dict(zip(by + ["quantile"], k)) |
                    dict(mean=mean, hw=hw, lo=mean - hw, hi=mean + hw, n_folds=n,
                         seeds=float(pd.to_numeric(grp["_nseed"],
                                                   errors="coerce").mean()),
                         wins=int((sign_of(metric) * grp[metric] > 0).sum())))
    return pd.DataFrame(rows).sort_values(by + ["quantile"]).reset_index(drop=True)


def best_q(df, metric, level=0.95):
    """Where each criterion peaks, and whether the peak is distinguishable.

    `sep` is the gap between the best quantile and the runner-up, in half-widths of
    the best point's own interval. Below 1 the "optimum" is a ranking of noise, and
    the honest sentence is that the knob does not matter over that range -- which is
    a result, and a much easier one to defend than a peak that moves with the seed.
    """
    c = curve(df, metric, level=level)
    if c.empty:
        return c
    s = sign_of(metric)
    out = []
    for (series, crit), g in c.groupby(["series", "criterion"], sort=True):
        g = g.sort_values("mean", ascending=(s < 0))
        top = g.iloc[0]
        runner = g.iloc[1] if len(g) > 1 else None
        gap = (np.nan if runner is None
               else float(s * (top["mean"] - runner["mean"])))
        out.append(dict(series=series, criterion=crit, best_q=float(top["quantile"]),
                        mean=float(top["mean"]), hw=float(top["hw"]),
                        runner_q=(np.nan if runner is None
                                  else float(runner["quantile"])),
                        gap=gap,
                        sep=(np.nan if runner is None or not np.isfinite(top["hw"])
                             or top["hw"] == 0 else gap / float(top["hw"]))))
    return pd.DataFrame(out).sort_values(["series", "criterion"]).reset_index(drop=True)


def paper_point(df, metric, level=0.95):
    """The grid cell each dataset's reported numbers stand on, beside its series' best.

    The question the construction ablation has to answer is not "which cell is best"
    but "is the one we used defensible". `behind` is how far the paper's point sits
    below the best cell of the same series, in that cell's half-widths.
    """
    c = curve(df, metric, level=level)
    if c.empty:
        return c
    s = sign_of(metric)
    rows = []
    for series, g in c.groupby("series", sort=True):
        ds = series.split(" / ")[0]
        if ds not in PAPER_POINT:
            continue
        crit, q = PAPER_POINT[ds]
        mine = g[(g.criterion == crit) & np.isclose(g["quantile"].astype(float), q)]
        best = g.sort_values("mean", ascending=(s < 0)).iloc[0]
        rows.append(dict(
            series=series, criterion=crit, quantile=q,
            mean=(np.nan if mine.empty else float(mine.iloc[0]["mean"])),
            hw=(np.nan if mine.empty else float(mine.iloc[0]["hw"])),
            best_criterion=best["criterion"], best_q=float(best["quantile"]),
            best_mean=float(best["mean"]),
            behind=(np.nan if mine.empty or not np.isfinite(best["hw"])
                    or best["hw"] == 0
                    else float(s * (best["mean"] - mine.iloc[0]["mean"])
                               / best["hw"])),
            in_grid=not mine.empty))
    return pd.DataFrame(rows)


def floor(df, metric, by=("series",)):
    """How small a difference this grid can resolve: the model noise left in a fold
    mean after the seeds are averaged.

    `within` is the seed spread inside one (criterion, quantile, fold) cell; `floor`
    is that divided by sqrt(seeds), which is what survives into the numbers the curve
    is made of. `between` is the spread of the fold means themselves -- reported only
    so the ratio is visible, because it is the much larger quantity and drawing IT on
    the curve would answer "how different are the folds" on a figure asking "did the
    knob change anything".

    A bump shorter than `floor` is not a finding, whatever the band says. With one
    seed it is undefined and the notebook draws no mark -- the honest statement, not
    a zero.
    """
    by = list(by)
    g = graph_rows(df)
    empty = pd.DataFrame(columns=by + ["within", "between", "floor", "seeds"])
    if metric not in g.columns or g.empty or g.seed.nunique() < 2:
        return empty
    d = g.assign(**{metric: pd.to_numeric(g[metric], errors="coerce")}).dropna(
        subset=[metric])
    cell = (d.groupby(by + ["criterion", "quantile", "fold"])[metric]
            .agg(sd="std", k="count", m="mean").reset_index())
    cell = cell[cell["k"] > 1]
    if cell.empty:
        return empty
    out = []
    for key, grp in cell.groupby(by, sort=True):
        within = float(grp["sd"].mean())
        seeds = float(grp["k"].mean())
        between = float(grp.groupby(["criterion", "quantile"])["m"].std().mean())
        out.append(dict(zip(by, key if isinstance(key, tuple) else (key,)))
                   | dict(within=within, between=between,
                          floor=within / np.sqrt(seeds), seeds=seeds))
    return pd.DataFrame(out)
