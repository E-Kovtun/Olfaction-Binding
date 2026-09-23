"""The (criterion x quantile) construction sweep and its reader.

Three things here are load-bearing and easy to break silently:

* the unit of evidence -- model seeds are averaged INSIDE a fold, and the interval is
  over folds. Get it wrong and every band on the figure comes out about half as wide;
* the direction of an error metric. `best_q` picks an argmax, and on RMSE the best
  quantile is the SMALLEST one, not the largest;
* the resume key. A sweep that is restarted must skip what it has and must not skip a
  head or a split it has not computed yet.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from scripts.article_sweeps import s4_quantile_grid as qg  # noqa: E402


def _spec(name, rel):
    s = importlib.util.spec_from_file_location(name, _root / rel)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def sweep():
    """The sweep module. It imports the extractor, hence torch; conftest stubs
    torch_geometric where it is missing."""
    return _spec("_qsweep", "scripts/article_sweeps/s4_run_quantile_criteria.py")


# ----------------------------------------------------------------- fixtures

def frame(**over):
    """A small grid: 2 criteria x 3 quantiles x 3 folds x 2 seeds, plus the reference."""
    rows = []
    for crit, bump in (("coverage", 0.0), ("greedy_pair_cover", 0.02)):
        for q, shape in ((0.0, 0.0), (0.9, 0.01), (0.99, 0.03)):
            for fold in (1, 2, 3):
                for seed, jitter in ((42, -0.005), (43, +0.005)):
                    rows.append(dict(dataset="m2or", regime="inductive",
                                     criterion=crit, quantile=q, K=int(100 * (1 - q)),
                                     fold=fold, seed=seed, combo="cls+mol",
                                     split="test", status="ok", mol_source="chemberta",
                                     AUROC=0.80 + bump + shape + 0.01 * fold + jitter,
                                     RMSE=0.50 - bump - shape,
                                     n_receptors=10))
    for fold in (1, 2, 3):
        for seed in (42, 43):
            rows.append(dict(dataset="m2or", regime="inductive", criterion="boost_full",
                             quantile=np.nan, K=-1, fold=fold, seed=seed,
                             combo="prot+mol", split="test", status="ok",
                             mol_source="chemberta", AUROC=0.85 + 0.01 * fold,
                             RMSE=0.45, n_receptors=10))
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    return df.assign(series=df.dataset + " / " + df.regime)


# ----------------------------------------------------------------- aggregation

def test_the_interval_is_over_folds_not_over_cells():
    """THE rule of this repository. Three folds x two seeds is n=3, not n=6.

    With n=6 the Student-t factor would be 2.57 over sqrt(6) instead of 4.30 over
    sqrt(3), so the half-width would come out roughly half of the honest one -- and
    every 'significant' gap read off the figure would inherit that.
    """
    c = qg.curve(frame(), "AUROC")
    assert (c.n_folds == 3).all()
    assert (c.seeds == 2).all()


def test_seeds_are_averaged_before_the_interval():
    """Two seeds straddling the fold mean must leave the mean alone and shrink nothing
    but their own contribution -- the fold spread is what remains."""
    c = qg.curve(frame(), "AUROC")
    row = c[(c.criterion == "coverage") & (c["quantile"] == 0.0)].iloc[0]
    # folds contribute 0.81, 0.82, 0.83 after the +-0.005 seeds cancel
    assert row["mean"] == pytest.approx(0.82, abs=1e-9)


def test_an_empty_frame_is_typed_not_merely_empty():
    """`fill_between` on object-dtype columns dies on `isfinite`, and a half-finished
    sweep is exactly when the notebook is open."""
    c = qg.curve(frame().iloc[0:0], "AUROC")
    assert c.empty
    assert c["mean"].dtype.kind == "f"


def test_the_reference_arm_is_a_level_not_a_curve():
    lv = qg.levels(frame(), "AUROC")
    assert len(lv) == 1
    assert lv.iloc[0]["mean"] == pytest.approx(0.87, abs=1e-9)
    assert "quantile" not in lv.columns


def test_the_reference_arm_is_never_drawn_as_a_criterion():
    c = qg.curve(frame(), "AUROC")
    assert qg.BOOST not in set(c.criterion)


def test_a_head_filter_keeps_the_reference_arm(tmp_path):
    """The reference has its own combo (`prot+mol`). A naive `combo == 'cls+mol'`
    filter drops it, and the panels lose their baseline without saying so."""
    d = tmp_path / "quantile_criteria"
    d.mkdir()
    frame().to_csv(d / "metrics_m2or_inductive__chemberta.csv", index=False)
    df = qg.load(root=d, combo="cls+mol")
    assert (df.criterion == qg.BOOST).any()


# ----------------------------------------------------------------- direction

@pytest.mark.parametrize("metric,expect_q", [("AUROC", 0.99), ("RMSE", 0.99)])
def test_best_q_follows_the_metric_direction(metric, expect_q):
    """In the fixture q=0.99 is both the highest AUROC and the lowest RMSE. A reader
    that took an argmax on both would name q=0.0 on RMSE."""
    b = qg.best_q(frame(), metric)
    for _, r in b.iterrows():
        assert r["best_q"] == pytest.approx(expect_q)


def test_a_win_count_flips_for_error_metrics():
    d = qg.delta_vs_boost(frame(), "RMSE")
    # every graph cell has RMSE below the 0.45 reference except the flat q=0 pair
    row = d[(d.criterion == "greedy_pair_cover") & (d["quantile"] == 0.99)].iloc[0]
    assert row["mean"] < 0            # the difference stays in the metric's own units
    assert row["wins"] == 3           # ... but the count knows smaller is better


def test_delta_is_paired_inside_the_fold():
    """The reference rises with the fold exactly as the graph does, so a paired
    difference is constant and its interval collapses -- an unpaired one would carry
    the whole between-fold spread."""
    d = qg.delta_vs_boost(frame(), "AUROC")
    row = d[(d.criterion == "coverage") & (d["quantile"] == 0.0)].iloc[0]
    assert row["mean"] == pytest.approx(-0.05, abs=1e-9)
    assert row["hw"] == pytest.approx(0.0, abs=1e-9)


def test_sep_says_when_a_peak_is_noise():
    """A grid whose quantiles differ by nothing must not report an optimum with a
    straight face: `sep` is the gap to the runner-up in half-widths."""
    flat = frame()
    flat = flat.assign(AUROC=0.80 + 0.01 * flat["fold"])
    b = qg.best_q(flat, "AUROC")
    assert (b["gap"].abs() < 1e-9).all()


# ----------------------------------------------------------------- paper point

def test_the_reported_construction_is_located_in_the_grid():
    p = qg.paper_point(frame(), "AUROC")
    assert p.iloc[0]["criterion"] == "greedy_pair_cover"
    assert p.iloc[0]["quantile"] == pytest.approx(0.99)
    assert bool(p.iloc[0]["in_grid"])
    assert p.iloc[0]["behind"] == pytest.approx(0.0, abs=1e-9)


def test_a_missing_paper_point_is_reported_not_invented():
    """Leaving the paper's own cell out of the grid is the easiest way to produce an
    ablation that compares nothing. It must show up as a flag, not as a NaN in a
    column nobody reads."""
    df = frame()
    df = df[~((df.criterion == "greedy_pair_cover") & (df["quantile"] == 0.99))]
    p = qg.paper_point(df, "AUROC")
    assert not bool(p.iloc[0]["in_grid"])
    assert np.isnan(p.iloc[0]["mean"])


# ----------------------------------------------------------------- resolution

def test_the_floor_is_the_seed_spread_left_in_a_fold_mean():
    f = qg.floor(frame(), "AUROC")
    # the two seeds sit at +-0.005, so sd = 0.005*sqrt(2) and the floor is sd/sqrt(2)
    assert f.iloc[0]["floor"] == pytest.approx(0.005, abs=1e-9)
    assert f.iloc[0]["between"] > f.iloc[0]["floor"]


def test_one_seed_gives_no_floor_rather_than_zero():
    single = frame()
    single = single[single.seed == 42].copy()
    assert qg.floor(single, "AUROC").empty


# ----------------------------------------------------------------- the sweep

def test_failed_cells_never_reach_an_aggregate(tmp_path):
    """A degenerate graph at a tiny K is written as NaN with a reason. Kept on disk --
    the hole is a finding about that construction -- but a NaN inside an interval
    would shorten its n without saying so."""
    d = tmp_path / "q"
    d.mkdir()
    df = frame()
    df.loc[df.index[:6], "status"] = "failed: RuntimeError: one sign only"
    df.to_csv(d / "metrics_m2or_inductive__chemberta.csv", index=False)
    assert (qg.load(root=d).status == "ok").all()
    assert (qg.load(root=d, drop_failed=False).status != "ok").any()


def test_the_resume_key_separates_heads_and_splits(sweep):
    """A run that later adds `cls+prot+mol`, or asks for val scores, must fill those
    cells and skip the ones it has."""
    base = dict(criterion="coverage", quantile=0.9, fold=1, seed=42,
                combo="cls+mol", split="test")
    k = sweep.cell_key(base)
    assert k != sweep.cell_key(base | {"combo": "cls+prot+mol"})
    assert k != sweep.cell_key(base | {"split": "val"})
    assert k == sweep.cell_key(base | {"K": 7})      # K is a record, not an identity


def test_the_reference_arm_has_one_key_per_fold_and_seed(sweep):
    """It carries a NaN quantile; a key that compared NaN to NaN by equality would
    refit it on every resume."""
    a = sweep.cell_key(dict(criterion="boost_full", quantile=np.nan, fold=1, seed=42,
                            combo="prot+mol", split="test"))
    b = sweep.cell_key(dict(criterion="boost_full", quantile=float("nan"), fold=1,
                            seed=42, combo="prot+mol", split="test"))
    assert a == b


def test_quantiles_are_fractions_not_percents(sweep):
    """The old study script took percents. A 99 that reaches the extractor as `q`
    keeps nothing at all, and the failure looks like a modelling result."""
    assert max(sweep.QUANTILES) <= 1.0
    assert 0.99 in sweep.QUANTILES


def test_the_paper_point_is_in_the_default_grid(sweep):
    """The ablation exists to judge the construction we report. If the default grid
    does not contain it, the figure compares six alternatives to nothing."""
    for ds, (_crit, q) in qg.PAPER_POINT.items():
        assert any(np.isclose(q, x) for x in sweep.QUANTILES), ds
