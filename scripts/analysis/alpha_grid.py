#!/usr/bin/env python
"""The v8 alpha grid, aggregated: the layer the notebooks draw and compute nothing.

    from scripts.analysis.alpha_grid import load, curve, geometry, delta_vs
    df = load(mol_source="chemberta")          # every dataset x regime on disk
    curve(df, "R2")                            # mean +- CI per (series, alpha)

The producer is `scripts/modeling/train/run_alpha_gate_sweep.py`; `headline_table.py`
is the scoreboard at one alpha; this is the shape of the whole dial. It is a module and
not a script because a notebook needs to re-ask the same question with a different knob,
not to re-read a CSV somebody else already melted.

WHAT ONE ROW OF THE SWEEP IS. One (arm, alpha, fold, seed) cell:

    arm     `gate` is the dial itself. `boost_full` (XGBoost on raw ESM || molecule) and
            `naive` (the constant train mean) are references with no alpha at all;
            `graph_legacy` is the pre-v8 model.
    alpha   z_prot = (1 - alpha) * frozen_SVD(ESM) + alpha * graph(...). With ONE-HOT
            receptor nodes -- what the whole v8 grid runs -- ESM reaches the model
            nowhere else, so alpha IS the protein-embedding axis.
    fold    WHICH ROWS are held out: folds 1-5, or the cold-molecule seeds 42-46 on
            m2or/inductive. This is the axis the error bars are over.
    seed    the model's own draw (graph init, the bag, the head's subsample/colsample).

THE UNIT OF EVIDENCE IS THE FOLD, NOT THE (fold, seed) CELL. Every interval in this
module first averages the model seeds inside each fold and is then taken over the fold
means, so `n` counts folds and `seeds` says how many draws went into each one. Five
seeds on one fold rest on the same held-out rows: they are one observation about
generalisation, not five, and counting them as five halves the interval. See
`fold_means` for the argument and `noise_split` for the two spreads it separates.

At n=5 the 1.96 normal approximation is 29% too narrow, so the interval here is
Student-t -- see `ci`. An interval is still a refusal to look at five numbers;
`per_fold_delta`, `place_matrix`, `coverage` and `spread` are how they get looked at.

DIRECTION OF EVERY GEOMETRY COLUMN. RSA, CCA and Procrustes are all reported so that
LARGER IS MORE ALIGNED (Procrustes as 1 - disparity), against both references. The dial
working therefore means `*_esm` falls with alpha and `*_fun` rises, and `audit` is that
statement as a number.
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

from orbind.dataset import METRIC_HEADLINE, METRIC_NAMES            # noqa: E402

DEFAULT_ROOT = "results/graph/v8_alpha_gate"
GEOMS = ["rsa", "cca", "procrustes"]
REFS = ["esm", "fun"]
# The identity of a cell, and the axis every interval in this module is taken over.
SPLIT = ["fold", "seed"]
# Everything that makes two rows different experiments rather than two repeats of one.
IDENTITY = ["dataset", "regime", "mol_source", "variant_tag", "nodes"]
TASK = {"cc": "regression", "hc": "regression", "m2or": "classification",
        "cc_shrinked": "regression", "hc_shrinked": "regression",
        "cc_shrinked50": "regression", "hc_shrinked50": "regression"}
OF_RECORD = {"regression": "R2", "classification": "AUROC"}
# The arms with no alpha: horizontal references a curve is read against, never points
# on it. `naive` is the constant train mean -- R2's honest zero, and AUROC 0.5.
LEVEL_ARMS = ["boost_full", "graph_legacy", "naive"]

# The metrics a SMALLER value wins. It lives here, in the base reader, so every layer
# above ranks the same way -- `paper_tables` imports it rather than keeping a second
# copy that could drift.
LOWER_IS_BETTER = {"RMSE", "MAE"}

GEOM_LABEL = {"rsa": "RSA - neighbour order",
              "cca": "CCA - shared linear subspace",
              "procrustes": "Procrustes - same shape"}
REF_LABEL = {"esm": "vs ESM (structure)", "fun": "vs response profile (function)"}
ARM_LABEL = {"boost_full": "boost [ESM || mol]", "graph_legacy": "legacy graph",
             "naive": "naive (train mean)"}


# --------------------------------------------------------------------------- loading

def _parser():
    """Reuse the headline table's filename parser and its canonical-variant rule, so
    the two readers can never disagree about what a file on disk is."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_ht_for_grid", _root / "scripts/analysis/headline_table.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _as_list(v):
    if v is None:
        return None
    return [v] if isinstance(v, (str, int, np.integer)) else list(v)


SPLIT_PREFIX = {"test": "metrics_", "val": "val_metrics_"}


def load(root=DEFAULT_ROOT, mol_source=None, nodes="onehot", dataset=None, regime=None,
         variant=None, seed=None, folds=None, drop_failed=True, split="test",
         combo="cls+mol"):
    """Every metrics CSV under `root` as one frame, filtered by the notebook's knobs.

    `nodes` defaults to "onehot" because that is the only setting in which the v8 gate's
    alpha is an honest fraction of structure: with ESM node features the protein
    embedding still reaches the receptor vector through message passing at alpha=1, so
    that dial has no upper end. Pass `nodes=None` to look at an older ESM-node series
    anyway, and `nodes="nodedial"` for the v9 runs.

    THE TWO DIALS DO NOT MIX. On a v9 run alpha is `prot_mix` and runs the other way --
    alpha=0 is receptor identity alone, alpha=1 is the legacy graph -- so a frame
    holding both would put opposite meanings on one axis. The default keeps them apart;
    if you widen `nodes`, split on it before plotting anything.

    `seed` selects the MODEL seed and defaults to every one present -- unlike the
    headline table, which pins 42 to stay line-for-line comparable with the published
    series. Here more seeds simply widen the interval, and `coverage` says how many.

    `split` picks WHICH HELD-OUT SET the metrics are on. "test" is the sweep's own
    output. "val" is the validation rescore from `val_rescore.py`, and it is the only
    frame alpha may be CHOSEN on -- choosing on "test" and then reporting "test" is
    choosing the number and its defence from the same rows. The two live in separate
    files with separate globs, so the frames cannot silently merge.

    `combo` picks WHICH BOOSTING HEAD the graph rows come from. The sweep fits
    `cls+mol` ([z_prot || molecule]) and `cls+prot+mol` ([z_prot || raw ESM ||
    molecule]) on the same trained graph and writes both under one arm/alpha/fold/seed;
    a frame holding both would average two models into one number, so a frame holds
    exactly one. The reference arms are kept whichever is chosen.
    """
    import argparse
    ht = _parser()
    root = pathlib.Path(root)
    if not root.is_absolute() and not root.exists():
        root = _root / root
    ns = argparse.Namespace(dataset=_as_list(dataset), mol_source=_as_list(mol_source),
                            regime=_as_list(regime), nodes=_as_list(nodes),
                            variant=_as_list(variant), all_variants=False,
                            all_seeds=True, graph_combo=combo)
    if split not in SPLIT_PREFIX:
        raise ValueError(f"split must be one of {sorted(SPLIT_PREFIX)}, got {split!r}")
    df = ht.load(root, ns, prefix=SPLIT_PREFIX[split])
    if "split" not in df.columns:
        df = df.assign(split=split)
    if seed is not None:
        df = df[df.seed.isin(_as_list(seed))]
    if folds is not None:
        # A two-fold canary against a five-fold series is not a comparison: on the cold
        # regimes a single fold moves R2 by 0.1, so the difference would mostly be which
        # rows each side happened to hold out. Restrict both to the same folds first.
        df = df[df.fold.isin([int(f) for f in _as_list(folds)])]
    if drop_failed and "status" in df.columns:
        df = df[~df["status"].astype(str).str.startswith("failed")]
    if df.empty:
        raise SystemExit(f"nothing left under {root} after the knobs -- check `nodes` "
                         f"and `mol_source` against what is actually on disk")
    return add_series(df.copy())


def add_series(df):
    """A single string key per experiment, built from the identity columns that ACTUALLY
    VARY in this frame.

    Two reasons it is not an f-string of all five. A blank `variant_tag` round-trips
    through CSV as NaN and `NaN == NaN` is false, which is how a whole dataset once
    vanished from a plot; and a knob that pins `mol_source` and then prints it in every
    label is noise the reader has to subtract by hand on every panel.
    """
    # `assign`, not `df[col] = ...`: pandas decides whether an assignment is "chained"
    # by the frame's refcount, so a local inside a helper trips the ChainedAssignment
    # FutureWarning every time -- a false positive, but one that would print a
    # twenty-line block into the notebook on every load.
    df = df.assign(**{c: (df[c] if c in df.columns
                          else pd.Series("", index=df.index)).fillna("").astype(str)
                      for c in IDENTITY})
    # "Varies" means varies WITHIN a (dataset, regime), not across the frame. The edge
    # variant is fixed by the dataset -- q0cov on the insects, q99greedy on m2or -- so
    # a frame-wide nunique() calls it an axis and welds "/ q0cov" onto every insect
    # label for no information at all. The molecule source, given twice for the same
    # cell, is a real axis and stays.
    varies = [c for c in IDENTITY if c not in ("dataset", "regime")
              and int(df.groupby(["dataset", "regime"])[c].nunique().max()) > 1]
    parts = ["dataset", "regime"] + varies
    out = df.assign(series=df[parts].agg(" / ".join, axis=1))
    out.attrs["series_parts"] = parts
    out.attrs["pinned"] = {c: df[c].iloc[0] for c in IDENTITY
                           if c not in parts and df[c].nunique() == 1}
    return out


def metric_for(df):
    """The metric of record per dataset -- R2 on the continuous insect panels, AUROC on
    the binary pool. A notebook that picks ONE metric for the whole grid leaves the
    m2or panels empty, which is exactly what happened to the previous one."""
    return {d: OF_RECORD[TASK[d]] for d in df.dataset.unique()}


def metrics_available(df, dataset=None, which="all"):
    """Metric columns actually written for this task, in the order the battery emits
    them, so a selector never offers a name that reads as NaN everywhere.

    `which="headline"` returns the short list instead -- five names on a regression
    panel rather than twelve. The wide battery is the right thing to WRITE (deciding
    later is free) and the wrong thing to PLOT: the regression names run
    R2/RMSE/MAE/Pearson/Spearman/Kendall plus a whole binarised discrete set, and a
    figure that puts one panel on each is a figure nobody reads across.
    """
    q = df if dataset is None else df[df.dataset == dataset]
    task = TASK[dataset] if dataset else None
    table = {"all": METRIC_NAMES, "headline": METRIC_HEADLINE}[which]
    names = (table[task] if task else
             list(table["regression"]) + list(table["classification"]))
    return [c for c in names if c in q.columns and q[c].notna().any()]


# ----------------------------------------------------------------------- aggregation

def ci(v, level=0.95):
    """(mean, half-width, n) over the repeat cells, Student-t.

    NOT the 1.96 normal approximation. A point here rests on five splits, where t is
    2.776 -- the normal interval would be 29% too narrow, and the whole job of the bar
    is to say whether two arms are distinguishable at that width."""
    from scipy.stats import t as _t
    v = pd.to_numeric(pd.Series(list(v)), errors="coerce").dropna()
    if v.empty:
        return np.nan, np.nan, 0
    n = len(v)
    if n == 1:
        return float(v.iloc[0]), np.nan, 1
    hw = float(_t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return float(v.mean()), hw, n


def fold_means(df, value, by=()):
    """One number per (by..., fold): the mean over MODEL SEEDS.

    THE UNIT OF EVIDENCE IS THE FOLD, and this is the function that says so.

    A fold changes which rows are held out -- that is the thing a claim about
    generalisation is over. A model seed changes only the draw the same model made on
    the same rows: graph init, the bag, the head's subsample/colsample. It is a nuisance
    to be averaged away, not a second sample of the world.

    Treating the 25 (fold, seed) cells of a five-seed grid as 25 observations is
    therefore wrong twice over. They are not independent -- five of them share a fold,
    and folds differ enormously in difficulty -- so a Student-t interval over them uses
    t(24) = 2.06 where the honest one is t(4) = 2.78, and it divides by sqrt(25) where
    the effective sample size is 5. The interval comes out roughly half of what it
    should be, and every "significant" gap read off it inherits that.

    Averaging first still buys the seeds' worth: each fold mean carries 1/5 of the seed
    variance, so the between-fold spread shrinks toward the pure fold-difficulty
    component. The narrowing arrives through the numerator, which is real, rather than
    through the denominator, which would be invented.
    """
    by = list(by)
    if "fold" not in df.columns or df.empty:
        return df.assign(_nseed=1) if len(df) else df.assign(_nseed=pd.Series(dtype=int))
    keys = by + ["fold"]
    g = df.groupby(keys, dropna=False, sort=False)
    out = g[value].mean().reset_index()
    n = (g["seed"].nunique() if "seed" in df.columns else g.size()).reset_index(
        name="_nseed")
    return out.merge(n, on=keys, how="left")


def _agg(df, value, by, level=0.95, unit="fold"):
    """mean +- t interval of `value` over the independent unit.

    `unit="fold"` (the default, and the only one that supports an inference) averages
    model seeds inside each fold first -- see `fold_means`. `unit="cell"` keeps every
    (fold, seed) row as its own observation; it is kept for looking at the raw scatter
    and must not be used for an interval anyone will quote.
    """
    by = list(by)
    seeds = np.nan
    if unit == "fold" and len(df) and "fold" in df.columns:
        df = fold_means(df, value, by)
        seeds = float(pd.to_numeric(df["_nseed"], errors="coerce").mean())
    rows = []
    for k, g in df.groupby(by, dropna=False, sort=True):
        m, hw, n = ci(g[value], level)
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)))
                    | dict(mean=m, hw=hw, lo=m - hw, hi=m + hw, n=n,
                           seeds=(float(pd.to_numeric(g["_nseed"],
                                                      errors="coerce").mean())
                                  if "_nseed" in g.columns else seeds)))
    if not rows:
        # Typed, not merely empty. An object-dtype empty frame reaches matplotlib as
        # `fill_between(object[], object[])` and dies on `isfinite` -- which is what a
        # series that has only reached its baselines looks like MID-RUN, i.e. exactly
        # when a notebook is most likely to be opened.
        empty = pd.DataFrame(columns=by + ["mean", "hw", "lo", "hi", "n", "seeds"])
        return empty.astype({c: float for c in ["mean", "hw", "lo", "hi", "seeds"]}
                            | {"n": "int64"}
                            | ({"alpha": float} if "alpha" in by else {}))
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def ci_folds(df, metric, level=0.95):
    """(mean, half-width, n_folds) over fold means. The scalar form of `_agg`."""
    if metric not in df.columns or df.empty:
        return float("nan"), float("nan"), 0
    f = fold_means(df.assign(**{metric: pd.to_numeric(df[metric], errors="coerce")}),
                   metric)
    return ci(f[metric].dropna(), level)


def curve(df, metric, arm="gate", by=("series",), level=0.95):
    """mean +- CI of `metric` along alpha. One row per (series, alpha)."""
    if metric not in df.columns:
        raise KeyError(f"{metric!r} is not in the sweep; have {metrics_available(df)}")
    q = df[df.arm == arm]
    q = q[pd.to_numeric(q[metric], errors="coerce").notna()]
    return _agg(q, metric, list(by) + ["alpha"], level)


def levels(df, metric, arms=tuple(LEVEL_ARMS), by=("series",), level=0.95):
    """The horizontal references a curve is read against, one row per (series, arm)."""
    by = list(by)
    if metric not in df.columns:
        return _agg(df.iloc[:0], by[0], by).assign(arm=pd.Series(dtype=str))
    out = []
    for a in arms:
        q = df[df.arm == a]
        q = q[pd.to_numeric(q[metric], errors="coerce").notna()]
        if q.empty:
            continue
        out.append(_agg(q, metric, by, level).assign(arm=a))
    if out:
        return pd.concat(out, ignore_index=True)
    return _agg(df.iloc[:0], metric, by).assign(arm=pd.Series(dtype=str))


def geometry(df, scale="value", by=("series",), level=0.95):
    """The dial's geometry: one row per (series, alpha, geom, ref).

    `scale="value"` is the measure itself -- RSA and CCA are correlations in [0, 1] and
    are directly comparable across panels. `scale="z"` is the same number against that
    cell's own permutation null: it says whether an alignment is distinguishable from
    chance at all, which the raw value cannot, but its size grows with the receptor
    count (1237 on M2OR against 24 on HC), so z is never an effect size ACROSS panels.
    """
    by = list(by)
    gate, out = df[df.arm == "gate"], []
    for g in GEOMS:
        for r in REFS:
            col = f"{g}_{r}" if scale == "value" else f"{g}_{r}_z"
            if col not in gate.columns:
                continue
            blk = gate[by + ["alpha"] + SPLIT].assign(
                geom=g, ref=r, v=pd.to_numeric(gate[col], errors="coerce"))
            out.append(blk.dropna(subset=["v"]))
    if not out:
        raise SystemExit("no geometry columns on any gate row -- was the sweep run with "
                         "--n-perm 0 (which writes the raw values but no z), or "
                         "--no-gate?")
    return _agg(pd.concat(out, ignore_index=True), "v",
                by + ["geom", "ref", "alpha"], level)


def per_fold_delta(df, metric, ref_arm="boost_full", arm="gate", by=("series",)):
    """The advantage over `ref_arm`, ONE ROW PER FOLD -- the sample `delta_vs` summarises.

    `delta_vs` returns the interval; this returns the five numbers inside it. They are
    worth drawing next to it, because an advantage of +0.01 that holds on five folds and
    one that is +0.06 on one fold and zero on four are the same mean and are not the same
    result. The fold is exactly what an interval hides.
    """
    by = list(by)
    cols = by + ["alpha", "fold", "d", "n_seeds"]
    if metric not in df.columns:
        return pd.DataFrame(columns=cols).astype({"d": float, "n_seeds": float})
    ref = df[df.arm == ref_arm]
    g = df[df.arm == arm]
    if ref.empty or g.empty:
        return pd.DataFrame(columns=cols).astype({"d": float, "n_seeds": float})
    ref = (ref[by + SPLIT + [metric]].rename(columns={metric: "_ref"})
           .drop_duplicates(subset=by + SPLIT))
    m = g[by + SPLIT + ["alpha", metric]].merge(ref, on=by + SPLIT, how="inner")
    m = m.assign(_d=pd.to_numeric(m[metric], errors="coerce")
                 - pd.to_numeric(m["_ref"], errors="coerce")).dropna(subset=["_d"])
    if m.empty:
        return pd.DataFrame(columns=cols).astype({"d": float, "n_seeds": float})
    out = fold_means(m, "_d", by + ["alpha"])
    return out.rename(columns={"_d": "d", "_nseed": "n_seeds"})[cols]


def noise_split(df, metric, by=("series",), arm="gate"):
    """How much of the spread is the MODEL and how much is the FOLD.

    Two numbers per (by..., alpha), and they answer different questions:

      within   the SD across model seeds INSIDE a fold, pooled over folds. This is the
               noise averaging removes -- the graph's initialisation, the bag, the head's
               subsample. It says nothing about generalisation.
      between  the SD of the fold means. This is the quantity every interval in this
               module is built from: how much the answer depends on WHICH rows were held
               out, which is the thing a claim about generalisation is about.

    `between` still contains within/sqrt(n_seeds) -- averaging shrinks the model
    component but does not delete it -- so `floor` reports that residual. When `between`
    is close to `floor`, the folds agree and what is left is model noise; when it towers
    over it, the folds genuinely disagree and no number of seeds will help.
    """
    by = list(by)
    cols = by + ["alpha", "within", "between", "floor", "n_folds", "n_seeds"]
    q = df[df.arm == arm]
    if metric not in df.columns or q.empty or "seed" not in q.columns:
        return pd.DataFrame(columns=cols)
    q = q.assign(_v=pd.to_numeric(q[metric], errors="coerce")).dropna(subset=["_v"])
    rows = []
    for k, g in q.groupby(by + ["alpha"], dropna=False, sort=True):
        key = dict(zip(by + ["alpha"], k if isinstance(k, tuple) else (k,)))
        per = g.groupby("fold")["_v"]
        # pooled within-fold SD: the average VARIANCE across folds, then the root --
        # averaging the SDs instead would be a different and smaller number
        v = per.var(ddof=1).dropna()
        within = float(np.sqrt(v.mean())) if len(v) else np.nan
        means = per.mean()
        between = float(means.std(ddof=1)) if len(means) > 1 else np.nan
        ns = float(g.groupby("fold")["seed"].nunique().mean())
        rows.append(key | dict(within=within, between=between,
                               floor=(within / np.sqrt(ns) if np.isfinite(within)
                                      and ns else np.nan),
                               n_folds=int(len(means)), n_seeds=ns))
    return pd.DataFrame(rows, columns=cols)


def place_matrix(df, metric, by=("series",), arms=("boost_full", "graph_legacy")):
    """The ranking PER FOLD -- one row per (series, competitor, fold).

    What `places` averages away. A model with mean place 2.0 that is second on every
    fold and one that is first on three and fourth on two are the same number and are
    not the same result, and the fold column is where that shows.

    Model seeds are averaged inside each fold FIRST, then every alpha AND the reference
    arms are ranked within that fold, then the ranks are averaged over folds. Ranking at
    the (fold, seed) level instead would let one lucky initialisation take a place away
    from a better model, and would count five seeds agreeing on one fold as five pieces
    of evidence rather than one.

    This is the summary that survives the fact that folds differ wildly in
    difficulty -- a cold-molecule fold moves every competitor at once, so a curve of
    MEANS partly plots which folds happened to be hard, while a curve of PLACES cannot.
    Read the two side by side: where they disagree, the mean is being moved by one fold.

    The ranking runs on the splits every competitor in that series shares.
    """
    by = list(by)
    out = []

    def _per_fold(q):
        """One value per fold, seeds averaged -- the row the ranking sees."""
        f = fold_means(q.assign(_v=pd.to_numeric(q[metric], errors="coerce")), "_v")
        return f.set_index("fold")["_v"].dropna()

    for k, g in df.groupby(by, dropna=False, sort=True):
        key = dict(zip(by, k if isinstance(k, tuple) else (k,)))
        if metric not in g.columns:
            continue
        parts = {}
        for a, q in g[g.arm == "gate"].groupby("alpha"):
            parts[f"a={float(a):g}"] = _per_fold(q)
        for arm in arms:
            q = g[g.arm == arm]
            if not q.empty:
                parts[ARM_LABEL.get(arm, arm)] = _per_fold(q)
        parts = {n: v[~v.index.duplicated()].dropna() for n, v in parts.items()}
        parts = {n: v for n, v in parts.items() if len(v)}
        if len(parts) < 2:
            continue
        common = set.intersection(*(set(v.index) for v in parts.values()))
        if not common:
            continue
        idx = sorted(common)
        M = pd.DataFrame({n: v.reindex(idx) for n, v in parts.items()})
        # descending for the usual metrics, ascending for RMSE -- ranking an error the
        # same way as a score hands first place to the worst model and looks fine
        R = M.rank(axis=1, ascending=metric in LOWER_IS_BETTER, method="average")
        for name, col in R.items():
            a = float(name[2:]) if name.startswith("a=") else np.nan
            for fold, rank in col.items():
                out.append(key | dict(competitor=name, alpha=a, fold=fold,
                                      rank=float(rank), value=float(M.loc[fold, name]),
                                      k=M.shape[1]))
    return pd.DataFrame(out, columns=by + ["competitor", "alpha", "fold", "rank",
                                           "value", "k"])


def places(df, metric, by=("series",), arms=("boost_full", "graph_legacy")):
    """`place_matrix` averaged over folds: one row per (series, competitor), 1 = best.

    `n` is how many folds the race ran on and `k` how many competitors were in it -- a
    place of 3.0 out of 5 and out of 13 are not the same statement.
    """
    by = list(by)
    pm = place_matrix(df, metric, by, arms)
    cols = by + ["competitor", "alpha", "place", "n", "k"]
    if pm.empty:
        return pd.DataFrame(columns=cols)
    g = pm.groupby(by + ["competitor"], dropna=False, sort=True)
    out = g.agg(alpha=("alpha", "first"), place=("rank", "mean"),
                n=("fold", "nunique"), k=("k", "first")).reset_index()
    return out[cols]


def duel_matrix(df, metric, by=("series",), ref_arm="boost_full", arm="gate"):
    """Head-to-head place against ONE opponent, split by split: 1 = ahead on that
    held-out set, 2 = behind, 1.5 = exactly level.

    The k-way race in `place_matrix` answers "where does this dial position stand among
    everything that ran". This answers the narrower question the story turns on -- does
    the model beat the boosting baseline on THIS split -- and it is not the same
    question: a position can slip from 3rd to 4th in the k-way race because two OTHER
    positions moved, while its own verdict against boost never changed.

    Built on `per_fold_delta`, so it inherits that reduction exactly: paired on
    (fold, seed) first, seeds averaged inside the fold, then one verdict per fold. A
    lucky initialisation therefore cannot win a split on its own.
    """
    by = list(by)
    cols = by + ["alpha", "fold", "place", "d", "n_seeds"]
    pf = per_fold_delta(df, metric, ref_arm=ref_arm, arm=arm, by=by)
    if pf.empty:
        return pd.DataFrame(columns=cols).astype({"place": float, "d": float})
    # an error metric is won by being SMALLER -- ranking it like a score hands first
    # place to the worse model, silently and on every panel
    lead = -pf["d"] if metric in LOWER_IS_BETTER else pf["d"]
    return pf.assign(place=np.where(lead > 0, 1.0,
                                    np.where(lead < 0, 2.0, 1.5)))[cols]


def duel(df, metric, by=("series",), ref_arm="boost_full", arm="gate"):
    """`duel_matrix` averaged over folds: one row per (by..., alpha), place in [1, 2].

    The scale is the point. With n folds the place moves in steps of 1/n -- 1.0 means
    ahead of the opponent on every split, 2.0 behind on every one, 1.5 an even
    division -- so the number says both how often the model wins and, by its distance
    from 1.5, how consistently. `wins`/`losses` are the same count in raw form.

    It carries no magnitude: a fold won by 0.001 and one won by 0.1 both score 1. Read
    it against `delta_vs`, which carries magnitude and no consistency.
    """
    by = list(by)
    cols = by + ["alpha", "place", "wins", "losses", "n", "seeds"]
    dm = duel_matrix(df, metric, by, ref_arm=ref_arm, arm=arm)
    if dm.empty:
        return pd.DataFrame(columns=cols)
    g = dm.groupby(by + ["alpha"], dropna=False, sort=True)
    out = g.agg(place=("place", "mean"),
                wins=("place", lambda s: int((s == 1.0).sum())),
                losses=("place", lambda s: int((s == 2.0).sum())),
                n=("fold", "nunique"),
                seeds=("n_seeds", "mean")).reset_index()
    return out[cols]


def _paired(df, metric, ref_arm, arm, by):
    """`arm` and `ref_arm` differenced on (fold, seed), or None if either is absent.

    Step 1 of the two reductions described on `delta_vs`, alone, so that every public
    function below differences the same way.
    """
    by = list(by)
    if metric not in df.columns:
        return None
    ref = df[df.arm == ref_arm].copy()
    g = df[df.arm == arm].copy()
    if ref.empty or g.empty:
        return None
    ref = ref[by + SPLIT + [metric]].rename(columns={metric: "_ref"})
    ref = ref.drop_duplicates(subset=by + SPLIT)
    m = g[by + SPLIT + ["alpha", metric]].merge(ref, on=by + SPLIT, how="inner")
    m = m.assign(_d=pd.to_numeric(m[metric], errors="coerce")
                 - pd.to_numeric(m["_ref"], errors="coerce")).dropna(subset=["_d"])
    return None if m.empty else m


def delta_folds(df, metric, ref_arm="boost_full", arm="gate", by=("series",)):
    """The paired advantage as ONE NUMBER PER (by..., alpha, fold).

    Steps 1 and 2 of `delta_vs` and nothing after them: pair on (fold, seed), then
    average the model seeds inside each fold. That is the frame a per-fold slope has to
    be fitted on -- the fold is the unit of evidence everywhere in this project, and a
    slope fitted on (fold, seed) rows would count five seeds agreeing on one fold as five
    pieces of evidence. Column `delta`.
    """
    by = list(by)
    m = _paired(df, metric, ref_arm, arm, by)
    if m is None:
        return pd.DataFrame(columns=by + ["alpha", "fold", "delta"])
    return (fold_means(m, "_d", by + ["alpha"])
            .rename(columns={"_d": "delta"})
            .sort_values(by + ["alpha", "fold"], ignore_index=True))


def delta_vs(df, metric, ref_arm="boost_full", arm="gate", by=("series",), level=0.95):
    """The paired difference `arm - ref_arm`: per (fold, seed) first, then per FOLD.

    Two reductions, in this order, and the order is the whole point.

    1. PAIR ON (fold, seed). Both arms ran on the same split with the same draw, so
       differencing removes the split and the draw at once. That is far stronger than
       two overlapping intervals and it is the only form in which a 0.01 effect is
       readable at all.
    2. AVERAGE THE SEEDS INSIDE EACH FOLD. What is left is one advantage per held-out
       set -- the model noise gone, the fold's own character intact -- and those five
       numbers are the independent sample the interval is taken over. `won` counts
       FOLDS in favour, out of five, not (fold, seed) cells out of twenty-five: five
       seeds agreeing on one fold is one piece of evidence, not five.
    """
    by = list(by)
    # Every early return hands back a frame with the RIGHT COLUMNS, not a bare empty
    # one: a reader that does `d[np.isclose(d.alpha, 1)]` has no way to tell "this arm
    # has not run yet" from "this object is not the thing I asked for", and mid-run the
    # first is the normal case.
    empty = _agg(df.iloc[:0], "alpha", by + ["alpha"]).assign(won=pd.Series(dtype=int))
    m = _paired(df, metric, ref_arm, arm, by)
    if m is None:
        return empty
    out = _agg(m, "_d", by + ["alpha"], level)
    # counted on the SAME reduction the interval uses, or the two would disagree about
    # what an observation is
    per_fold = fold_means(m, "_d", by + ["alpha"])
    won = (per_fold.assign(w=per_fold["_d"] > 0)
           .groupby(by + ["alpha"], sort=True)["w"].sum().reset_index(name="won"))
    return out.merge(won, on=by + ["alpha"], how="left")


# ---------------------------------------------------------------- checks and coverage

def coverage(df):
    """What is on disk, per series: the alphas, the repeats, the seeds, the arms.

    Read this BEFORE any plot. A curve drawn over a ragged grid -- one alpha finished on
    two folds and the rest on five -- is a curve whose wiggles are the schedule, and
    nothing in a mean +- CI will say so. `ragged` is that condition as a flag."""
    rows = []
    for s, g in df.groupby("series", sort=True):
        gate = g[g.arm == "gate"]
        per = gate.groupby("alpha").size()
        rows.append(dict(
            series=s, alphas=int(gate.alpha.nunique()),
            splits=int(g.fold.nunique()), seeds=int(g.seed.nunique()),
            cells_per_alpha=("0" if not len(per) else
                             (str(int(per.iloc[0])) if per.min() == per.max()
                              else f"{int(per.min())}-{int(per.max())}")),
            ragged=bool(len(per) and per.min() != per.max()),
            refs=", ".join(sorted(set(g.arm) - {"gate"}))))
    return pd.DataFrame(rows)


def audit(geo):
    """Does the dial travel, and which way. The manipulation check, before any reading.

    `mono` is the Spearman of the mean curve against alpha. The gate DEFINES esm to fall
    and fun to rise; where a curve does not, the knob is not doing what its name says
    and nothing downstream of it means anything."""
    from scipy.stats import spearmanr
    rows = []
    for k, g in geo.groupby(["series", "geom", "ref"], sort=True):
        m = g.set_index("alpha")["mean"].sort_index()
        if len(m) < 3:
            continue
        rows.append(dict(series=k[0], geom=k[1], ref=k[2], n_alpha=len(m),
                         at0=float(m.iloc[0]), at1=float(m.iloc[-1]),
                         travel=float(m.max() - m.min()),
                         mono=float(spearmanr(m.index.to_numpy(),
                                              m.to_numpy()).statistic)))
    cols = ["series", "geom", "ref", "n_alpha", "at0", "at1", "travel", "mono", "ok"]
    if not rows:
        # Two-point canaries land here, and so does any series still on its first
        # alphas. A bare DataFrame() has no columns, so the caller's groupby("series")
        # raises KeyError three frames up -- which is a crash where the honest answer is
        # "not enough of the dial yet".
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out.assign(ok=np.sign(out.mono) == np.where(out.ref == "esm", -1.0, 1.0))


def crossover(geo):
    """The alpha where the cloud stops being structural and starts being functional.

    Each reference curve is min-max scaled WITHIN its own (series, geom) panel first:
    raw RSA against ESM and raw RSA against a response matrix are not on a common scale,
    so their literal intersection would be an artefact of that. Scaled, both run 0..1
    and the crossing is where the cloud is equally far along both dials."""
    rows = []
    for k, g in geo.groupby(["series", "geom"], sort=True):
        m = g.pivot_table(index="alpha", columns="ref", values="mean")
        if not {"esm", "fun"}.issubset(m.columns) or len(m) < 3:
            continue
        n = (m - m.min()) / (m.max() - m.min()).replace(0, np.nan)
        d = (n["esm"] - n["fun"]).dropna()
        if d.empty or d.iloc[0] * d.iloc[-1] > 0:
            rows.append(dict(series=k[0], geom=k[1], alpha_cross=np.nan))
            continue
        a = d.index.to_numpy(float)
        i = int(np.argmax(np.sign(d.to_numpy()) != np.sign(d.iloc[0])))
        x0, x1, y0, y1 = a[i - 1], a[i], d.iloc[i - 1], d.iloc[i]
        rows.append(dict(series=k[0], geom=k[1],
                         alpha_cross=float(x0 - y0 * (x1 - x0) / (y1 - y0))))
    return pd.DataFrame(rows, columns=["series", "geom", "alpha_cross"])


def anchor_check(df, tol=0.05):
    """Each dial's END has a model it must reproduce. This is the grid's own smoke test.

    v8 GATE at alpha=0 the receptor vector is a frozen rotation of the very ESM that
             `boost` reads raw -- the same information through two different readers, so
             the two must nearly agree.
    v9 NODES at alpha=1 the node features ARE the embedding file, so the run IS the
             legacy graph and must land on the `graph_legacy` arm.

    A gap wider than `tol` means that end is not what it claims, and every position
    along the dial inherits the fault. Which end is checked follows the `nodes` tag, so
    a v9 file is never audited against v8's expectation -- the two dials run in opposite
    directions and the wrong check would fail on a perfectly good run."""
    rows = []
    for s, g in df.groupby("series", sort=True):
        m = OF_RECORD[TASK[g.dataset.iloc[0]]]
        if m not in g.columns:
            continue
        nodedial = str(g["nodes"].iloc[0]) == "nodedial"
        end, ref_arm = (1.0, "graph_legacy") if nodedial else (0.0, "boost_full")
        q = g[(g.arm == "gate") & np.isclose(g.alpha.astype(float), end)]
        ref = g[g.arm == ref_arm]
        if q.empty or ref.empty:
            continue
        va, _, na = ci(q[m])
        vb, _, nb = ci(ref[m])
        rows.append(dict(series=s, metric=m, end=f"alpha={end:g}", against=ref_arm,
                         value=va, reference=vb, gap=va - vb,
                         n=min(na, nb), ok=bool(abs(va - vb) <= tol)))
    return pd.DataFrame(rows)


def spread(df, metric, arm="gate", alpha=None):
    """The individual cells behind one point -- the five numbers a mean hides.

    An interval says how uncertain the mean is; it does not say whether the five splits
    agreed. On the cold-molecule regimes they routinely do not (+-0.1 R2 per fold), and
    a reader who never looks at this will over-read a 0.01 difference between two means
    that each span 0.2."""
    q = df[df.arm == arm]
    if alpha is not None:
        q = q[np.isclose(q.alpha.astype(float), float(alpha))]
    keep = [c for c in ["series", "alpha"] + SPLIT + [metric] if c in q.columns]
    return q[keep].sort_values([c for c in ["series", "alpha"] + SPLIT
                                if c in keep]).reset_index(drop=True)


# --------------------------------------------------------------------------- console

def _report(df, level=0.95):
    """The notebook's two blocks as text, for an ssh session with no browser."""
    geo = geometry(df, level=level)
    aud, cross = audit(geo), crossover(geo)
    W = max([len(s) for s in df.series.unique()] + [8]) + 2
    n_alpha = int(df.loc[df.arm == "gate", "alpha"].nunique())

    print("=" * (W + 76))
    print("=== ALPHA GRID")
    print("=" * (W + 76))
    print(coverage(df).to_string(index=False), "\n")
    anc = anchor_check(df)
    if len(anc):
        print(anc.to_string(index=False))
        print("  anchor OK everywhere" if anc.ok.all() else
              "  !! anchor off -- read nothing above alpha=0 on those rows", "\n")

    if aud.empty:
        print(f"  GEOMETRY: {n_alpha} alpha(s) on the dial -- the direction check "
              "and the crossover both need 3. The raw values are still in the "
              "frame; a canary is read on the ENDS below, not on the shape.")
        print()
    if not aud.empty:
        print(f"  {'series':<{W}}{'geom':<12}{'vs ESM  a=0 -> a=1':>26}{'mono':>7}"
              f"{'   ':<3}{'vs PROFILE  a=0 -> a=1':>26}{'mono':>7}{'  cross':>8}")
        print("  " + "-" * (W + 82))
    for s, g in aud.groupby("series", sort=True):
        for i, geom in enumerate(GEOMS):
            e, f = (g[(g.geom == geom) & (g.ref == r)] for r in REFS)
            if e.empty and f.empty:
                continue
            cell = lambda q: ("" if q.empty else                       # noqa: E731
                              f"{q.iloc[0].at0:>11.3f} ->{q.iloc[0].at1:>9.3f}")
            mono = lambda q: "" if q.empty else f"{q.iloc[0].mono:>+7.2f}"   # noqa: E731
            c = cross[(cross.series == s) & (cross.geom == geom)]
            xc = ("" if c.empty else "  never" if not np.isfinite(c.alpha_cross.iloc[0])
                  else f"{c.alpha_cross.iloc[0]:>8.2f}")
            print(f"  {s if i == 0 else '':<{W}}{geom:<12}{cell(e):>26}{mono(e)}"
                  f"{'   ':<3}{cell(f):>26}{mono(f)}{xc}")
        print()
    off = aud[~aud.ok.astype(bool)] if len(aud) else aud
    if aud.empty:
        pass
    else:
        print("  every curve travels the way the dial defines"
              if not len(off) else f"  !! {len(off)} curve(s) travel the WRONG way")
    for r in off.head(8).itertuples():
        print(f"     {r.series:<{W}}{r.geom}/{r.ref}  mono {r.mono:+.2f}  "
              f"travel {r.travel:.3f}")

    print("\n" + "=" * (W + 76))
    print("=== PREDICTION -- mean +- CI over the splits, and the paired gap at alpha=1")
    print("=" * (W + 76))
    print(f"  {'series':<{W}}{'metric':<9}{'boost':>16}{'alpha=0':>16}{'alpha=1':>16}"
          f"{'legacy':>16}{'a1 - boost':>20}")
    print("  " + "-" * (W + 91))
    for s in sorted(df.series.unique()):
        one = df[df.series == s]
        m = OF_RECORD[TASK[one.dataset.iloc[0]]]
        if m not in one.columns:
            continue
        cur = curve(one, m, level=level).set_index("alpha")
        lev = levels(one, m, level=level).set_index("arm")
        d1 = delta_vs(one, m, level=level)
        d1 = d1[np.isclose(d1.alpha, 1.0)]

        def _c(fr, k):
            if k not in fr.index:
                return f"{'-':>16}"
            r = fr.loc[k]
            # ASCII "+/-", matching headline_table.py: this is read over ssh, and a
            # terminal in a non-UTF-8 locale turns the glyph into mojibake mid-number.
            return (f"{r['mean']:.3f}" + (f" +/-{r['hw']:.3f}"
                                          if np.isfinite(r["hw"]) else "")).rjust(16)
        gap = (f"{'-':>20}" if d1.empty else
               (f"{d1['mean'].iloc[0]:+.3f} [{int(d1['won'].iloc[0])}/"
                f"{int(d1['n'].iloc[0])}]").rjust(20))
        print(f"  {s:<{W}}{m:<9}{_c(lev, 'boost_full')}{_c(cur, 0.0)}{_c(cur, 1.0)}"
              f"{_c(lev, 'graph_legacy')}{gap}")
    print("\n  the scoreboard of record is scripts/analysis/headline_table.py;\n"
          "  the figures are notebooks/graph/alpha_gate/alpha_gate.ipynb")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Read the v8 alpha grid from a terminal: the same two blocks the "
                    "notebook draws, as text.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--mol-source", nargs="+", default=None)
    ap.add_argument("--nodes", nargs="+", default=["onehot"],
                    choices=["esm", "onehot", "nodedial"],
                    help="which series. `onehot` is the v8 gate (alpha: structure -> "
                         "function), `nodedial` the v9 node dial (alpha: function -> "
                         "structure), `esm` the pre-separation runs. Do NOT pass two: "
                         "alpha means opposite things on the first two")
    ap.add_argument("--dataset", nargs="+", default=None)
    ap.add_argument("--regime", nargs="+", default=None,
                    choices=["transductive", "inductive"])
    ap.add_argument("--seed", type=int, nargs="+", default=None,
                    help="model seed(s); default: every one on disk")
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="restrict to these splits. Use it to read a canary against a "
                         "finished series on the folds they share")
    ap.add_argument("--level", type=float, default=0.95)
    a = ap.parse_args()
    _report(load(root=a.root, mol_source=a.mol_source, nodes=a.nodes,
                 dataset=a.dataset, regime=a.regime, seed=a.seed, folds=a.folds),
            level=a.level)


if __name__ == "__main__":
    main()
