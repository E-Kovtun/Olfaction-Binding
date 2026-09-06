"""Choosing the alpha the paper reports.

This file pins the two properties that make the summary legal at all, because both are
easy to lose and neither fails loudly:

  the rank is SCALE-FREE     Carey is scored in R2 and M2OR in AUROC. Averaging those
                             is a unit error; averaging their orderings is not. So a
                             monotone change of one table's units must leave every mean
                             rank untouched.
  dz is DIMENSIONLESS        the one pooled effect-size column. Same requirement, for
                             the same reason.

and the two that decide what gets printed: `won` counts rows of the three tables (not
splits, which would silently change the denominator mid-table), and the 1-SE set is the
refusal to read an ordering the data does not support.
"""
import argparse

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import alpha_choice as ac
from scripts.analysis import alpha_grid as ag


def _grid(alphas=(0.0, 0.5, 1.0), folds=(1, 2, 3, 4, 5),
          sources=("chemberta",), gate=None, boost=None, legacy=0.40, noise=0.0):
    """A grid with a KNOWN winner per dataset, so a rank can be asserted.

    `gate` maps alpha -> score offset and `boost` is a flat level; both are per dataset
    so the two tasks can disagree, which is the case the pooling has to survive.
    """
    gate = gate or {a: 0.40 + 0.10 * a for a in alphas}
    boost = boost or {"cc": 0.45, "hc": 0.45, "m2or": 0.45}
    rng = np.random.default_rng(0)
    rows = []
    for ds, regime in (("cc", "transductive"), ("cc", "inductive"),
                       ("hc", "transductive"), ("hc", "inductive"),
                       ("m2or", "transductive"), ("m2or", "inductive")):
        metric = ag.OF_RECORD[ag.TASK[ds]]
        for mol in sources:
            for f in folds:
                base = dict(dataset=ds, regime=regime, mol_source=mol, fold=f, seed=42,
                            status="ok", nodes="onehot",
                            variant_tag="q99greedy" if ds == "m2or" else "q0cov")
                jit = rng.normal(0, noise) if noise else 0.0
                rows.append(base | {"arm": "boost_full", "alpha": np.nan,
                                    metric: boost[ds] + jit})
                rows.append(base | {"arm": "graph_legacy", "alpha": np.nan,
                                    metric: legacy + jit})
                for a in alphas:
                    rows.append(base | {"arm": "gate", "alpha": a,
                                        metric: gate[a] + jit})
    return ag.add_series(pd.DataFrame(rows))


def _mof(df, metric=None):
    return ac._metric_of(df, argparse.Namespace(metric=metric))


# ------------------------------------------------------------------- the two invariants

def test_the_mean_rank_does_not_move_when_one_table_changes_units():
    """The whole reason the headline criterion is a RANK. Carey in R2 and M2OR in AUROC
    cannot be averaged; a rank can. Rescaling one table must be invisible."""
    df = _grid(noise=0.01)
    before = ac.ranked(ac.scores(df, _mof(df))).set_index("competitor")["mean_rank"]
    rescaled = df.copy()
    m = rescaled.dataset == "m2or"
    rescaled.loc[m, "AUROC"] = rescaled.loc[m, "AUROC"] * 100 + 7   # monotone
    after = ac.ranked(ac.scores(rescaled, _mof(rescaled))).set_index("competitor")["mean_rank"]
    pd.testing.assert_series_equal(before.sort_index(), after.sort_index())


def test_dz_is_dimensionless():
    """The one column that pools the three tables. Scale a table and both the difference
    and its spread scale with it, so the ratio must not budge."""
    df = _grid(noise=0.02)
    a = ac.advantage(df, _mof(df)).set_index(["dataset", "regime", "alpha"])["dz"]
    sc = df.copy()
    m = sc.dataset == "cc"
    sc.loc[m, "R2"] = sc.loc[m, "R2"] * 25.0
    b = ac.advantage(sc, _mof(sc)).set_index(["dataset", "regime", "alpha"])["dz"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


# ------------------------------------------------------------------------ the ordering

def test_the_best_alpha_is_the_one_that_actually_wins():
    """A grid where the score rises monotonically with alpha: alpha=1 must come first
    and alpha=0 last, on every dataset at once."""
    df = _grid(alphas=(0.0, 0.25, 0.5, 0.75, 1.0), noise=0.005)
    rk = ac.ranked(ac.scores(df, _mof(df)))
    gate = rk[rk.alpha.notna()]
    assert gate.iloc[0].alpha == 1.0
    assert gate.iloc[-1].alpha == 0.0
    assert gate.mean_rank.is_monotonic_increasing


def test_boost_is_a_competitor_and_lands_where_it_belongs():
    """`boost` is in the ranking on purpose -- "which row of the table wins on average"
    has to be a number. Put it above every alpha and it must come first."""
    df = _grid(boost={"cc": 0.99, "hc": 0.99, "m2or": 0.99})
    rk = ac.ranked(ac.scores(df, _mof(df)))
    assert rk.iloc[0].competitor == "boost"
    assert rk.iloc[0].mean_rank == pytest.approx(1.0)


def test_identical_arms_share_a_place_instead_of_taking_it_by_row_order():
    """Two arms with the same score must tie. Ranking by position would hand one of them
    a spurious win in every cell and make the mean rank a function of column order."""
    df = _grid(alphas=(0.0, 1.0), gate={0.0: 0.5, 1.0: 0.5},
               boost={"cc": 0.1, "hc": 0.1, "m2or": 0.1})
    rk = ac.ranked(ac.scores(df, _mof(df))).set_index("competitor")
    assert rk.loc["a=0", "mean_rank"] == pytest.approx(rk.loc["a=1", "mean_rank"])
    assert rk.loc["a=0", "mean_rank"] == pytest.approx(1.5)     # (1 + 2) / 2


# --------------------------------------------------------------------- what gets printed

def test_won_counts_rows_of_the_tables_not_splits():
    """Six cells here, five splits each. A `won` that reported 30 would be answering a
    different question than the three tables ask."""
    df = _grid(gate={0.0: 0.1, 0.5: 0.9, 1.0: 0.9},
               boost={"cc": 0.5, "hc": 0.5, "m2or": 0.5})
    adv = ac.advantage(df, _mof(df))
    cells = adv.assign(w=adv.d > 0).groupby("alpha").agg(won=("w", "sum"),
                                                        cells=("w", "size"))
    assert int(cells.loc[1.0, "cells"]) == 6          # not 30
    assert int(cells.loc[1.0, "won"]) == 6
    assert int(cells.loc[0.0, "won"]) == 0


def test_the_one_se_set_is_the_leader_plus_everything_within_its_error():
    """Constructed by hand: a clear leader, one alpha just inside one SE, one far out."""
    rk = pd.DataFrame([
        dict(competitor="a=1", alpha=1.0, mean_rank=2.0, se=0.5),
        dict(competitor="a=0.85", alpha=0.85, mean_rank=2.4, se=0.5),   # inside 2.5
        dict(competitor="a=0.5", alpha=0.5, mean_rank=2.6, se=0.5),     # outside
        dict(competitor="boost", alpha=np.nan, mean_rank=1.0, se=0.5),  # not a candidate
    ])
    tied, cut = ac.one_se(rk)
    assert cut == pytest.approx(2.5)
    assert sorted(tied.alpha) == [0.85, 1.0]
    assert "boost" not in set(tied.competitor)      # the dial only


def test_a_flat_criterion_keeps_more_than_one_alpha_in_the_tied_set():
    """The case the script exists for. When every alpha scores the same, the 1-SE set
    must be ALL of them -- an argmax picked out of that is noise."""
    df = _grid(alphas=(0.0, 0.5, 1.0), gate={0.0: 0.5, 0.5: 0.5, 1.0: 0.5})
    tied, _ = ac.one_se(ac.ranked(ac.scores(df, _mof(df))))
    assert len(tied) == 3


# ------------------------------------------------------------------------- ragged input

def test_an_arm_missing_from_a_cell_is_dropped_not_imputed():
    """Mid-run, or with --no-legacy. Filling the gap would invent a number; carrying the
    competitor on fewer cells is honest but not comparable, so the count has to differ
    visibly -- the report keys its ragged warning off exactly this."""
    df = _grid()
    thin = df[~((df.arm == "graph_legacy") & (df.dataset == "m2or"))]
    rk = ac.ranked(ac.scores(thin, _mof(thin))).set_index("competitor")
    assert rk.loc["legacy", "cells"] == 4
    assert rk.loc["a=1", "cells"] == 6


def test_leave_one_table_out_reports_one_row_per_dataset():
    df = _grid(noise=0.01)
    L = ac.loo(df, _mof(df), argparse.Namespace())
    assert sorted(L.held_out) == ["Carey", "Hallem", "M2OR"]
    assert L.chosen_on_other_two.notna().all()


def test_an_unwritten_metric_is_refused_rather_than_ranked_as_nan():
    """`--metric AUROC` on the insect panels would rank a column of NaN and print an
    ordering with no data under it."""
    df = _grid()
    with pytest.raises(SystemExit, match="not written"):
        _mof(df, metric="AUPRC")          # regression tables have no AUPRC column
