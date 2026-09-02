"""The wide metric batteries, and the one property that makes them trustworthy.

A discrete metric on a continuous response needs a rule for calling something a
response, and that rule is a model of the data. If it is fit on the test rows, the
label definition itself has seen the held-out data -- and the flattery lands on
whichever arm is being scored, silently, in a column that looks like every other.
So the leak tests come first here; the arithmetic follows.

The reference of record is `rec0`: above that receptor's own TRAIN centre. It is the
convention the graph's own edge signs already use
(`orbind.gnn_extractor._mp_edges`, `edge_center="per_receptor"`), and it removes the
between-receptor baseline heterogeneity that a fixed cut inherits.
"""
import numpy as np
import pytest

from orbind.dataset import (METRIC_NAMES, binarisations,
                            classification_metrics_full, regression_metrics_full)


@pytest.fixture
def panel():
    """A small insect-shaped panel: four receptors with different baselines, a
    prediction that tracks the truth, and a separate train split."""
    rng = np.random.default_rng(0)
    rec_te = rng.choice(["R1", "R2", "R3", "R4"], 240)
    base = {"R1": 1.2, "R2": 0.0, "R3": -0.8, "R4": 0.3}
    y_te = np.array([base[r] for r in rec_te]) + rng.normal(size=240)
    pred = y_te * 0.7 + rng.normal(size=240) * 0.5
    rec_tr = rng.choice(["R1", "R2", "R3", "R4"], 600)
    y_tr = np.array([base[r] for r in rec_tr]) + rng.normal(size=600)
    return dict(y_te=y_te, pred=pred, rec_te=rec_te, y_tr=y_tr, rec_tr=rec_tr)


def _cuts(**kw):
    """The cut each convention applied, recovered as `pred - score`. `which=None`
    asks for every convention, not just the reference of record."""
    kw.setdefault("which", None)
    return {n: np.round(np.asarray(kw["pred"]) - np.asarray(score), 6)
            for n, _, score, _ in binarisations(**kw)}


# ------------------------------------------------------------------- no leaks

def test_the_cuts_are_fit_on_train_and_nothing_else(panel):
    """Move the TEST response and the cuts must not follow it; move the TRAIN
    response and they must."""
    base = dict(y_te=panel["y_te"], pred=panel["pred"], rec_te=panel["rec_te"],
                y_tr=panel["y_tr"], rec_tr=panel["rec_tr"])
    ref = _cuts(**base)
    shifted = _cuts(**(base | {"y_te": panel["y_te"] + 5.0}))
    for name in ref:
        assert np.allclose(ref[name], shifted[name]), f"{name} moved with the TEST rows"
    moved = _cuts(**(base | {"y_tr": panel["y_tr"] + 5.0}))
    assert not np.allclose(ref["rec0"], moved["rec0"]), "rec0 ignored the train rows"
    assert np.allclose(ref["glob0"], moved["glob0"]), "the fixed cut must not move at all"


def test_without_a_train_split_only_the_unfitted_cut_is_offered(panel):
    """Refusing to guess. Falling back to the test response would be a leak that
    looks exactly like a legitimate column, so the fitted cuts simply do not appear
    -- and `rec0`, the reference of record, is one of them."""
    names = [n for n, *_ in binarisations(panel["y_te"], panel["pred"],
                                          rec_te=panel["rec_te"], which=None)]
    assert names == ["glob0"], "a fitted cut appeared with nothing to fit it on"
    assert not list(binarisations(panel["y_te"], panel["pred"],
                                  rec_te=panel["rec_te"])),         "rec0 was asked for and cannot be built -- nothing should be yielded"
    got = regression_metrics_full(panel["y_te"], panel["pred"], rec_te=panel["rec_te"])
    assert not [k for k in got if k.startswith("rec0_")]


def test_a_receptor_unseen_in_train_falls_back_to_the_global_centre(panel):
    """It cannot happen in the two primary regimes -- every test receptor is warm --
    but a KeyError here would be a crash in the middle of a 720-cell run."""
    rec_te = np.array(["R9"] * 10 + list(panel["rec_te"][:10]))
    got = _cuts(y_te=panel["y_te"][:20], pred=panel["pred"][:20], rec_te=rec_te,
                y_tr=panel["y_tr"], rec_tr=panel["rec_tr"])
    assert "rec0" in got
    assert np.allclose(got["rec0"][:10], float(np.mean(panel["y_tr"])))


# ------------------------------------------------------------------ semantics

def test_per_receptor_centring_changes_who_counts_as_a_responder(panel):
    """Why `rec0` and not a fixed cut. R1 sits well above the pool mean, so under a
    global zero most of its rows are positive; against its OWN centre about half
    are. A discrete metric on the fixed cut would be scoring which receptors are
    active, not whether the odorants were called right."""
    got = {n: y for n, y, _, _ in binarisations(
        panel["y_te"], panel["pred"], rec_te=panel["rec_te"], y_tr=panel["y_tr"],
        rec_tr=panel["rec_tr"], which=None)}
    r1 = panel["rec_te"] == "R1"
    # the claim is the CONTRAST, not an exact balance: the train centre is an
    # estimate, so the per-receptor split lands near half rather than on it
    assert got["glob0"][r1].mean() > 0.7
    assert abs(got["rec0"][r1].mean() - 0.5) < abs(got["glob0"][r1].mean() - 0.5) - 0.1


def test_the_hard_call_uses_the_same_cut_as_the_truth(panel):
    """On a z-scored response the prediction is in the truth's units, so `>= 0.5`
    -- what the binary helper does -- would score the model against a boundary it
    was never asked to respect."""
    for name, y_bin, score, hard in binarisations(
            panel["y_te"], panel["pred"], rec_te=panel["rec_te"],
            y_tr=panel["y_tr"], rec_tr=panel["rec_tr"]):
        assert np.array_equal(hard, score > 0), name


# ------------------------------------------------------------------- contract

def test_the_battery_is_exactly_what_the_readers_select_on(panel):
    """METRIC_NAMES is what a table's `--metric` validates against; a name no
    function emits is a column that silently reads as NaN everywhere."""
    got = regression_metrics_full(panel["y_te"], panel["pred"], rec_te=panel["rec_te"],
                                  y_tr=panel["y_tr"], rec_tr=panel["rec_tr"])
    assert set(got) == set(METRIC_NAMES["regression"])
    yb = (panel["y_te"] > 0).astype(int)
    got = classification_metrics_full(yb, 1 / (1 + np.exp(-panel["pred"])))
    assert set(got) == set(METRIC_NAMES["classification"])


def test_the_pooled_columns_still_match_the_narrow_function(panel):
    """The battery is a SUPERSET: every number already reported has to come out of
    it unchanged."""
    from orbind.dataset import regression_metrics
    narrow = regression_metrics(panel["y_te"], panel["pred"])
    wide = regression_metrics_full(panel["y_te"], panel["pred"])
    for k, v in narrow.items():
        assert wide[k] == pytest.approx(v), k


def test_a_single_class_test_fold_does_not_crash(panel):
    """AUROC is undefined when every row clears the cut; it has to come back NaN
    rather than raise in the middle of a sweep."""
    y = np.abs(panel["y_te"]) + 50.0        # every row far above every receptor centre
    m = regression_metrics_full(y, panel["pred"], rec_te=panel["rec_te"],
                                y_tr=panel["y_tr"], rec_tr=panel["rec_tr"])
    assert np.isnan(m["rec0_AUROC"])
    assert np.isfinite(m["rec0_F1"])
