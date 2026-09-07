"""The reader behind the alpha-gate notebook.

Two kinds of test here, and the second kind is the reason the file exists.

The arithmetic -- the interval, the pairing, the crossover -- is checked against values
computed by hand, because every one of them is a number somebody will quote.

The MID-RUN shapes are checked because this grid takes hours on a GPU and is meant to
be looked at while it is still filling in. A series that has reached its baselines and
no further, an arm that was not requested, a cell that died on the GPU: each of those
is a normal state of the directory, and each must come back as an empty frame with the
right columns rather than as a bare `DataFrame()` that explodes three layers up inside
matplotlib.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.analysis import alpha_grid as ag


def _rows(series=(("cc", "rand"),), alphas=(0.0, 0.5, 1.0), folds=(1, 2, 3, 4, 5),
          arms=("boost_full", "naive", "graph_legacy"), metric="R2", seed=42):
    """A grid in the shape the sweep writes it, with a monotone dial in both the
    geometry and the metric so direction tests have something to find."""
    rng = np.random.default_rng(0)
    out = []
    for ds, fam in series:
        for f in folds:
            base = dict(dataset=ds, regime=ag.OF_RECORD and "transductive", fold=f,
                        seed=seed, status="ok", mol_source="chemberta",
                        variant_tag="q0cov", nodes="onehot")
            for arm in arms:
                out.append(base | {"arm": arm, "alpha": np.nan,
                                   metric: {"boost_full": 0.50, "naive": 0.0,
                                            "graph_legacy": 0.46}[arm]
                                   + rng.normal(0, 0.01)})
            for a in alphas:
                out.append(base | {
                    "arm": "gate", "alpha": a,
                    metric: 0.50 - 0.05 * a + rng.normal(0, 0.01),
                    "rsa_esm": 0.9 - 0.8 * a, "rsa_fun": 0.1 + 0.4 * a,
                    "rsa_esm_z": 40 - 38 * a, "rsa_fun_z": 1 + 9 * a})
    return ag.add_series(pd.DataFrame(out))


# ------------------------------------------------------------------ the arithmetic

def test_the_interval_is_students_t_and_not_the_normal_one():
    """At n=5 the difference is 42% of the half-width, which is the difference between
    an interval that covers 95% of the time and one that covers 89%. The grid rests on
    five splits, so this is the operative case and not a corner."""
    v = [0.10, 0.12, 0.11, 0.15, 0.13]
    m, hw, n = ag.ci(v)
    sd = float(np.std(v, ddof=1))
    assert n == 5
    assert m == pytest.approx(np.mean(v))
    assert hw == pytest.approx(2.7764 * sd / np.sqrt(5), rel=1e-3)
    assert hw > 1.96 * sd / np.sqrt(5)


def test_one_cell_gets_a_mean_and_no_bar():
    """Not a zero-width bar. Half of a curve drawn with hw=0 would claim a precision
    the single run cannot support."""
    m, hw, n = ag.ci([0.4])
    assert (m, n) == (0.4, 1) and np.isnan(hw)


def test_the_delta_is_paired_on_the_split_not_a_difference_of_means():
    """Give the two arms a large per-fold offset and a tiny constant gap. Unpaired,
    the fold noise swamps it; paired, it is exact -- which is the whole reason the
    sweep runs both arms on the same split with the same draw."""
    rows = []
    for f in range(1, 6):
        noise = f * 0.10                       # huge, shared, and not the effect
        for arm, v in (("boost_full", 0.50 + noise), ("gate", 0.52 + noise)):
            rows.append(dict(dataset="cc", regime="transductive", arm=arm, fold=f,
                             seed=42, alpha=np.nan if arm != "gate" else 1.0,
                             R2=v, mol_source="chemberta", variant_tag="q0cov",
                             nodes="onehot", status="ok"))
    df = ag.add_series(pd.DataFrame(rows))
    d = ag.delta_vs(df, "R2")
    assert d["mean"].iloc[0] == pytest.approx(0.02, abs=1e-9)
    assert d["hw"].iloc[0] == pytest.approx(0.0, abs=1e-9)   # identical every fold
    assert int(d["won"].iloc[0]) == 5
    # the unpaired comparison would have been swamped by the fold offset
    assert ag.ci(df[df.arm == "gate"].R2)[1] > 0.05


def test_the_crossover_is_interpolated_between_the_sampled_alphas():
    """The dial is sampled, so the crossing almost never lands on a sampled alpha. A
    reported crossover that is always one of the sampled values is a reader snapping to
    the grid, and the number would then be an artefact of the schedule."""
    # deliberately NOT symmetric: two mirror-image curves would cross at a sampled
    # alpha and the test would pass on a reader that only ever returns sampled values
    esm, fun = {0.0: 1.0, 0.5: 0.6, 1.0: 0.0}, {0.0: 0.0, 0.5: 0.7, 1.0: 1.0}
    geo = pd.DataFrame(
        [dict(series="s", geom="rsa", ref="esm", alpha=a, mean=v) for a, v in esm.items()]
        + [dict(series="s", geom="rsa", ref="fun", alpha=a, mean=v) for a, v in fun.items()])
    x = ag.crossover(geo).alpha_cross.iloc[0]
    # by hand: scaled d = (+1, -0.1, -1), so the root sits at 0.5 * 1 / 1.1
    assert x == pytest.approx(0.5 / 1.1, rel=1e-9)
    assert x not in (0.0, 0.5, 1.0)


def test_a_curve_that_never_crosses_says_so_instead_of_guessing():
    geo = pd.DataFrame([dict(series="s", geom="rsa", ref="esm", alpha=a, mean=1.0)
                        for a in (0.0, 0.5, 1.0)]
                       + [dict(series="s", geom="rsa", ref="fun", alpha=a, mean=0.1 * a)
                          for a in (0.0, 0.5, 1.0)])
    assert np.isnan(ag.crossover(geo).alpha_cross.iloc[0])


def test_audit_flags_a_dial_that_travels_the_wrong_way():
    """The gate DEFINES esm to fall and fun to rise. This is a manipulation check, so
    it has to be able to fail -- an audit that passes on reversed data checks nothing."""
    geo = ag.geometry(_rows())
    assert ag.audit(geo).ok.all()
    flipped = geo.assign(mean=np.where(geo.ref == "esm", geo["mean"] * -1, geo["mean"]))
    assert not ag.audit(flipped).ok.all()


def test_the_anchor_check_can_fail():
    """At alpha=0 the receptor vector is a frozen rotation of the ESM `boost` reads
    raw, so a wide gap means the anchor is broken and everything above it is void."""
    df = _rows()
    assert ag.anchor_check(df).ok.all()
    broken = df.copy()
    m = (broken.arm == "gate") & np.isclose(broken.alpha, 0.0)
    broken.loc[m, "R2"] = broken.loc[m, "R2"] - 0.3
    assert not ag.anchor_check(broken).ok.any()


# ------------------------------------------------------------------- mid-run shapes

def test_a_series_with_only_baselines_returns_an_empty_curve_that_can_be_plotted():
    """The normal state of the last dataset in the queue. An object-dtype empty frame
    reaches matplotlib as fill_between(object[], object[]) and dies on `isfinite`."""
    df = _rows(alphas=())
    cur = ag.curve(df, "R2")
    assert cur.empty
    assert set(["series", "alpha", "mean", "hw", "lo", "hi", "n"]) <= set(cur.columns)
    for c in ("mean", "hw", "lo", "hi"):
        assert cur[c].dtype.kind == "f"


def test_delta_against_an_arm_that_was_not_run_keeps_its_columns():
    """`--no-legacy` is a normal flag, and a reader that then does
    `d[np.isclose(d.alpha, 1)]` must not meet a frame with no `alpha` at all."""
    df = _rows(arms=("boost_full", "naive"))
    d = ag.delta_vs(df, "R2", ref_arm="graph_legacy")
    assert d.empty and "alpha" in d.columns and "won" in d.columns
    assert d[np.isclose(d.alpha, 1.0)].empty          # the call the notebook makes


def test_levels_for_an_absent_arm_is_empty_but_indexable():
    lev = ag.levels(_rows(arms=("boost_full",)), "R2")
    assert set(lev.arm) == {"boost_full"}
    assert ag.levels(_rows(arms=()), "R2").empty


def test_a_failed_cell_is_dropped_and_the_rest_still_read():
    """One cell can die on the GPU without the sweep dying; the reader must not average
    its NaN into the fold it belonged to."""
    df = _rows()
    n_before = len(ag.curve(df, "R2"))
    hurt = df.copy()
    m = (hurt.arm == "gate") & np.isclose(hurt.alpha, 0.5) & (hurt.fold == 3)
    hurt.loc[m, "status"] = "failed: CUDA out of memory"
    hurt.loc[m, "R2"] = np.nan
    hurt = hurt[~hurt["status"].astype(str).str.startswith("failed")]
    cur = ag.curve(hurt, "R2")
    assert len(cur) == n_before
    assert int(cur[np.isclose(cur.alpha, 0.5)].n.iloc[0]) == 4
    assert int(cur[np.isclose(cur.alpha, 1.0)].n.iloc[0]) == 5


def test_coverage_flags_a_ragged_dial():
    """A curve over a ragged grid has wiggles that are the run SCHEDULE, and no mean
    +- CI will say so. This flag is the only warning the reader gets."""
    df = _rows()
    assert not ag.coverage(df).ragged.any()
    thin = df[~((df.arm == "gate") & np.isclose(df.alpha, 0.5) & (df.fold > 2))]
    assert ag.coverage(thin).ragged.all()


# ------------------------------------------------------------------ series labelling

def test_the_label_names_only_the_axes_that_actually_vary():
    """The edge variant is FIXED by the dataset, so a frame-wide nunique calls it an
    axis and welds '/ q0cov' onto every insect label for no information. The molecule
    source, present twice for the same cell, is a real axis."""
    one = _rows(series=(("cc", "rand"), ("m2or", "transductive")))
    one.loc[one.dataset == "m2or", "variant_tag"] = "q99greedy"
    assert ag.add_series(one).attrs["series_parts"] == ["dataset", "regime"]
    two = pd.concat([one, one.assign(mol_source="gin")], ignore_index=True)
    assert "mol_source" in ag.add_series(two).attrs["series_parts"]
    assert any(s.endswith("gin") for s in ag.add_series(two).series.unique())


def test_a_blank_variant_does_not_delete_a_dataset():
    """A blank variant_tag round-trips through CSV as NaN, and NaN == NaN is false --
    which once dropped four of six series out of a plot silently."""
    df = _rows()
    df.loc[:, "variant_tag"] = np.nan
    got = ag.add_series(df)
    assert got.series.notna().all() and got.series.nunique() == 1


# ------------------------------------------------------------------------ the canary

def test_a_two_point_dial_reads_without_crashing():
    """The canary runs `--alphas 0 1.0`, so the shape checks have nothing to work with.
    `audit` and `crossover` need three points and correctly return nothing -- but they
    must return nothing WITH COLUMNS, or the reader's groupby("series") raises KeyError
    three frames up and the first command anyone runs after a canary dies."""
    df = _rows(alphas=(0.0, 1.0), folds=(1, 2))
    geo = ag.geometry(df)
    aud, cross = ag.audit(geo), ag.crossover(geo)
    assert aud.empty and {"series", "geom", "ref", "mono", "ok"} <= set(aud.columns)
    assert cross.empty and {"series", "geom", "alpha_cross"} <= set(cross.columns)
    assert aud.groupby("series").ngroups == 0          # the call that used to raise
    ag._report(df)                                     # and the whole report renders


def test_the_v9_end_is_audited_against_legacy_not_against_boost():
    """The two dials run in opposite directions, so the wrong end check would fail on a
    perfectly good v9 run: its alpha=0 is receptor identity alone and has no reason to
    land on boost."""
    df = _rows(alphas=(0.0, 1.0))
    v9 = df.assign(nodes="nodedial")
    got = ag.anchor_check(ag.add_series(v9))
    assert set(got["end"]) == {"alpha=1"} and set(got["against"]) == {"graph_legacy"}
    got8 = ag.anchor_check(df)
    assert set(got8["end"]) == {"alpha=0"} and set(got8["against"]) == {"boost_full"}
