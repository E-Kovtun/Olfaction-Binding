"""The sweep's three artifacts, its resume, and which splits it scores.

One computation now produces three files, and the whole point is that they cannot
disagree: `metrics_*.csv` is the test view, `val_metrics_*.csv` the validation view, and
`records_*.csv` every row of both plus train, timings, sizes, geometry and provenance.
The tests below pin the parts where a mistake would be silent -- a resume that thinks an
old directory is finished, a val column that is quietly the test column, a train row that
is scored against test labels.

No graph is trained here: everything under test is bookkeeping, and bookkeeping is where
a sweep loses a day.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _sweep():
    spec = importlib.util.spec_from_file_location(
        "_sweep_under_test",
        ROOT / "scripts/modeling/train/run_alpha_gate_sweep.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_sweep_under_test"] = m
    spec.loader.exec_module(m)
    return m


sw = _sweep()


def _args(out, **kw):
    return argparse.Namespace(**(dict(out=str(out), variant=None, mol_source="chemberta",
                                      nodes="esm", dial="nodes", force=False,
                                      score_train=True, seed_graph=True) | kw))


# ------------------------------------------------------------------- the three files

def test_the_three_files_share_one_stem(tmp_path):
    """A dump that cannot be matched to its own test view by name is a dump nobody
    will trust six months from now."""
    m, v, r = sw.sibling_paths("hc", "transductive", _args(tmp_path))
    assert m.name == "metrics_hc_rand_chemberta_nodedial.csv"
    assert v.name == "val_metrics_hc_rand_chemberta_nodedial.csv"
    assert r.name == "records_hc_rand_chemberta_nodedial.csv"
    assert m.parent == v.parent == r.parent
    # the dial tag rides on all three, so a v8 and a v9 run in one directory stay
    # three-for-three distinct rather than two files colliding
    g = sw.sibling_paths("hc", "transductive", _args(tmp_path, dial="gate"))
    assert [x.name for x in g] == ["metrics_hc_rand_chemberta.csv",
                                   "val_metrics_hc_rand_chemberta.csv",
                                   "records_hc_rand_chemberta.csv"]


# ------------------------------------------------------------------------- the resume

def test_an_old_directory_is_refused_rather_than_half_filled(tmp_path):
    """THE trap. A sweep from before the dump has test rows and nothing else. Resuming
    into it would mark every cell done and leave a records file covering only whatever
    happened to be missing -- a dump that looks complete and is not."""
    m, _, r = sw.sibling_paths("hc", "transductive", _args(tmp_path))
    m.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm="gate", alpha=1.0, fold=1, seed=42, R2=0.5)]).to_csv(
        m, index=False)
    with pytest.raises(SystemExit, match="before the full dump"):
        sw.load_done(r, m, _args(tmp_path))
    # --force is the deliberate way through, and it starts clean
    rows, cells = sw.load_done(r, m, _args(tmp_path, force=True))
    assert rows == [] and cells == set()


def test_a_cell_counts_as_done_only_when_its_TEST_row_is_there(tmp_path):
    """Resume is at cell granularity and the test row is written last, so a cell whose
    train and val rows landed before a kill is recomputed rather than left ragged."""
    m, _, r = sw.sibling_paths("hc", "transductive", _args(tmp_path))
    m.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(arm="gate", alpha=1.0, fold=1, seed=42, split="train", R2=0.9),
        dict(arm="gate", alpha=1.0, fold=1, seed=42, split="val", R2=0.6),
        dict(arm="gate", alpha=1.0, fold=1, seed=42, split="test", R2=0.5),
        dict(arm="gate", alpha=0.5, fold=1, seed=42, split="train", R2=0.9),
        dict(arm="gate", alpha=0.5, fold=1, seed=42, split="val", R2=0.6),
    ]).to_csv(r, index=False)
    rows, cells = sw.load_done(r, m, _args(tmp_path))
    assert len(rows) == 5
    assert cells == {sw.key("gate", 1.0, 1, 42)}          # 0.5 is NOT done


def test_a_records_file_from_before_seeds_reads_as_seed_42(tmp_path):
    m, _, r = sw.sibling_paths("hc", "transductive", _args(tmp_path))
    m.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm="gate", alpha=1.0, fold=3, R2=0.5)]).to_csv(r, index=False)
    _, cells = sw.load_done(r, m, _args(tmp_path))
    assert cells == {sw.key("gate", 1.0, 3, 42)}


# -------------------------------------------------------------------------- the splits

def _P(n_tr=6, n_va=4, n_te=5):
    rng = np.random.default_rng(0)
    mk = lambda n, tag: np.array([f"{tag}{i}" for i in range(n)])   # noqa: E731
    return {"tr": np.arange(n_tr), "va": np.arange(n_va), "te": np.arange(n_te),
            "y_tr": rng.normal(size=n_tr).astype(np.float32),
            "y_va": rng.normal(size=n_va).astype(np.float32),
            "y_te": rng.normal(size=n_te).astype(np.float32),
            "rec_tr": mk(n_tr, "r"), "mol_tr": mk(n_tr, "m"),
            "rec_va": mk(n_va, "R"), "mol_va": mk(n_va, "M"),
            "rec_te": mk(n_te, "t"), "mol_te": mk(n_te, "u")}


def test_an_empty_val_split_is_dropped_not_written_as_nan():
    """Some split families have no val at all. A row of NaN there would later read as a
    failed cell, which is a different thing from 'this family has no val'."""
    assert sw.splits_wanted(_args("."), _P()) == ["train", "val", "test"]
    assert sw.splits_wanted(_args("."), _P(n_va=0)) == ["train", "test"]
    assert sw.splits_wanted(_args(".", score_train=False), _P()) == ["val", "test"]


def test_each_split_is_scored_against_its_own_labels():
    """The metric battery is written against the test keys, and `_score_split` swaps the
    split's own labels in. If that swap were wrong every val number in the dump would be
    a test number wearing a val label -- and nothing downstream could tell."""
    P = _P()
    perfect_val = np.asarray(P["y_va"], dtype=np.float32)
    got = sw._score_split(P, perfect_val, "regression", "val")
    assert got["R2"] == pytest.approx(1.0, abs=1e-6)
    # the same predictions scored as if they were test must NOT come out perfect
    P2 = _P(n_te=len(P["y_va"]))
    other = sw._score_split(P2, perfect_val, "regression", "test")
    assert other["R2"] < 0.99


def test_the_split_keys_cover_every_split_the_sweep_scores():
    assert set(sw.SPLIT_KEYS) == set(sw.SPLITS)


# ---------------------------------------------------------------------- the provenance

def test_provenance_is_stamped_and_stable_within_a_process():
    a = sw.provenance(_args("."))
    b = sw.provenance(_args("."))
    assert set(a) == {"commit", "host", "started"}
    assert a == b, "the run's own timestamp must not move between cells"


def test_an_empty_split_becomes_a_zero_row_matrix_not_an_exception():
    """`_fold_prep` now builds val feature matrices too, and `np.stack([])` raises. A
    family that ships no val must reach `splits_wanted` and be dropped there, not blow
    up fold preparation for every arm at once."""
    emb = {"a": np.zeros(7, np.float32), "b": np.ones(7, np.float32)}
    assert sw._mat(emb, np.array([], dtype=object)).shape == (0, 7)
    assert sw._mat(emb, ["a", "b"]).shape == (2, 7)


# ------------------------------------------------------- extending a run with seeds

def test_a_finished_seed_42_run_is_extended_not_recomputed(tmp_path):
    """The exact move being made on v9_seeded: five folds at seed 42 are on disk, and
    the rerun asks for five seeds. Only the four new ones may be planned -- recomputing
    seed 42 would burn a fifth of the grid AND, with --seed-graph, quietly replace rows
    other numbers already rest on."""
    m, _, r = sw.sibling_paths("hc", "transductive", _args(tmp_path))
    m.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in (1, 2, 3, 4, 5):
        for arm, alpha in (("boost_full", np.nan), ("naive", np.nan),
                           ("graph_legacy", np.nan), ("gate", 0.0), ("gate", 1.0)):
            for split in ("train", "val", "test"):
                rows.append(dict(arm=arm, alpha=alpha, fold=fold, seed=42,
                                 split=split, status="ok", R2=0.5))
    pd.DataFrame(rows).to_csv(r, index=False)

    _, done = sw.load_done(r, m, _args(tmp_path))
    assert len(done) == 5 * 5                      # 5 folds x 5 arms, test rows only

    A = argparse.Namespace(seeds=[42, 43, 44, 45, 46], alphas=[0.0, 1.0],
                           legacy=True, gate=True, baselines_only=False)
    jobs = sw.plan([1, 2, 3, 4, 5], A, done)
    assert {j[3] for j in jobs} == {43, 44, 45, 46}, "seed 42 must not be replanned"
    # per new seed x fold: one baselines job + legacy + two gate arms
    assert len(jobs) == 4 * 5 * 4
    # and seed-major, so a partial run leaves whole seeds rather than a ragged slice
    assert [j[3] for j in jobs] == sorted(j[3] for j in jobs)


def test_resuming_with_the_wrong_seeding_flag_is_refused(tmp_path):
    """--seed-graph is the only axis NOT in the filename, so a resume with the flag
    flipped would append rows whose seed column means something different from the ones
    already there, under the same name, with nothing to tell them apart."""
    m, _, r = sw.sibling_paths("hc", "transductive", _args(tmp_path))
    m.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(arm="gate", alpha=1.0, fold=1, seed=42, split="test", seeded_graph=True),
        dict(arm="boost_full", alpha=np.nan, fold=1, seed=42, split="test"),
    ]).to_csv(r, index=False)
    with pytest.raises(SystemExit, match="two different series"):
        sw.load_done(r, m, _args(tmp_path, seed_graph=False))
    _, cells = sw.load_done(r, m, _args(tmp_path, seed_graph=True))
    assert len(cells) == 2      # the reference arms carry no flag and must not trip it
