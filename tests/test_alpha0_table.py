"""The alpha=0 identity control table.

The one thing that silently inverts a conclusion here is the direction of an error
metric. RMSE and MAE are won by being SMALLER, and `alpha_grid.delta_vs` — the
obvious function to reach for — counts `d > 0` without flipping them, so a table built
on it reports the worse model as the winner on every error column. These tests pin the
flip, and the fold pairing it rides on.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location(
    "_a0", _root / "scripts/article_tables/s3_alpha0_vs_boost.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


@pytest.mark.parametrize("metric,higher_wins", [("R2", True), ("AUROC", True),
                                                ("Pearson", True), ("Spearman", True),
                                                ("RMSE", False), ("MAE", False)])
def test_an_error_metric_is_won_by_being_smaller(metric, higher_wins):
    bigger = {1: 0.9, 2: 0.9, 3: 0.9}
    smaller = {1: 0.1, 2: 0.1, 3: 0.1}
    w, n = A.wins(bigger, smaller, metric)
    assert n == 3
    assert w == (3 if higher_wins else 0)


def test_only_shared_folds_are_compared():
    # A run that reached four folds against one that reached five must be judged on the
    # four they share -- the fifth would otherwise silently count as a loss.
    a = {1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0}
    b = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5}
    w, n = A.wins(a, b, "R2")
    assert (w, n) == (3, 4)


def test_no_shared_folds_is_not_a_zero_score():
    w, n = A.wins({1: 1.0}, {2: 0.0}, "R2")
    assert w is None and n == 0
    assert A.score(w, n, 3) == ("--", "")


def test_an_exact_tie_counts_for_neither_side():
    w, n = A.wins({1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.4}, "R2")
    assert (w, n) == (1, 2)


def test_seeds_are_averaged_inside_a_fold_before_anything_is_compared():
    # Two seeds on one fold are one observation about generalisation, not two.
    df = pd.DataFrame({"fold": [1, 1, 2, 2], "seed": [42, 43, 42, 43],
                       "R2": [0.4, 0.6, 0.1, 0.3]})
    assert A.fold_values(df, "R2") == {1: pytest.approx(0.5), 2: pytest.approx(0.2)}


def test_a_metric_the_run_never_wrote_yields_no_folds():
    df = pd.DataFrame({"fold": [1], "seed": [42], "R2": [0.5]})
    assert A.fold_values(df, "AUROC") == {}


def test_the_flag_appears_at_the_threshold_and_not_below():
    assert A.score(3, 5, 3) == ("3/5", "<<")
    assert A.score(2, 5, 3) == ("2/5", "")


def test_a_cell_with_one_fold_prints_a_value_but_no_interval():
    one = A.cell({1: 0.42}, 0.95)
    assert one["mean"] == pytest.approx(0.42) and one["n"] == 1
    assert not np.isfinite(one["hw"])
    assert A.num(pd.Series({"mean_x": one["mean"], "hw_x": one["hw"]}), "x") == "0.420"


def test_an_empty_cell_renders_as_a_dash():
    assert A.num(pd.Series({"mean_x": np.nan, "hw_x": np.nan}), "x") == "--"


def test_a_paired_difference_is_oriented_so_positive_means_the_graph_is_ahead():
    graph = {1: 0.30, 2: 0.30, 3: 0.30}
    other = {1: 0.50, 2: 0.50, 3: 0.50}
    # R2: the graph is behind, so the difference is negative
    assert A.paired(graph, other, "R2", 0.95)["mean"] == pytest.approx(-0.2)
    # RMSE: the graph's error is smaller, so it is AHEAD and the difference is positive
    assert A.paired(graph, other, "RMSE", 0.95)["mean"] == pytest.approx(+0.2)


def test_pairing_removes_the_split_difficulty_that_both_rows_share():
    # Two rows that move together over splits of very different difficulty, one always
    # 0.01 ahead: unpaired they overlap completely, paired the difference is exact.
    hard_easy = {1: 0.40, 2: 0.60, 3: 0.80, 4: 0.50, 5: 0.70}
    ahead = {f: v + 0.01 for f, v in hard_easy.items()}
    ahead[5] += 0.002                         # not exactly constant, so t is defined
    d = A.paired(ahead, hard_easy, "R2", 0.95)
    assert d["n"] == 5 and d["mean"] == pytest.approx(0.0104)
    assert d["hw"] < 0.002 and d["p"] < 0.001


def test_a_paired_difference_uses_only_the_shared_splits():
    d = A.paired({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, {2: 0.5, 3: 0.5, 4: 0.5, 5: 0.0},
                 "R2", 0.95)
    assert d["n"] == 3 and d["mean"] == pytest.approx(0.5)


def test_the_star_follows_the_corrected_p_and_not_the_sign():
    assert A.dnum(0.046, 0.012, 0.01).endswith("$^{*}$")
    assert "*" not in A.dnum(0.046, 0.012, 0.20)
    assert A.dnum(-0.056, 0.02, 0.01, tex=False) == "-0.056+/-0.020*"
    assert A.dnum(np.nan, np.nan, np.nan) == "--"
