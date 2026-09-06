"""Printing the three paper tables out of TWO result trees.

The v8 grid is safe on its own: one sweep, one set of folds, one head, paired by
construction. The risk is entirely in the second tree. `results/ensemble_logs/` holds
runs that may be on another split family, another fold set, another task or another
head, and a number lifted out of it into the same table looks exactly like a number
that belongs there.

So what is pinned here is the matching and the refusals: a run is found by its SOURCE
SPEC and never by its combo name, it is admitted only for the (dataset, regime) its
config actually describes, it is differenced only over folds both sides really ran, and
every departure from that comes back as a note rather than as a quietly plausible cell.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import alpha_grid as ag
from scripts.analysis import paper_tables as pt


def _grid(alphas=(0.4, 1.0), folds=(1, 2, 3, 4, 5), ds="m2or",
          regime="transductive", boost=0.80, gate=0.85):
    metric = ag.OF_RECORD[ag.TASK[ds]]
    rows = []
    for f in folds:
        base = dict(dataset=ds, regime=regime, mol_source="chemberta", fold=f, seed=42,
                    status="ok", nodes="onehot",
                    variant_tag="q99greedy" if ds == "m2or" else "q0cov")
        rows.append(base | {"arm": "boost_full", "alpha": np.nan, metric: boost})
        rows.append(base | {"arm": "graph_legacy", "alpha": np.nan, metric: gate - 0.005})
        for a in alphas:
            rows.append(base | {"arm": "gate", "alpha": a, metric: gate})
    return ag.add_series(pd.DataFrame(rows))


def _run(tmp, pool, name, cfg, repeats, metric="AUROC", level=0.7, combos=("cls+mol",)):
    d = tmp / pool / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    rows = [dict(repeat=r, kind="combo", name=c, **{metric: level})
            for r in repeats for c in combos]
    pd.DataFrame(rows).to_csv(d / "metrics.csv", index=False)
    return d


M2OR_TRANS = dict(regime="full_full", full_full_mode="transductive",
                  task="classification", tune_boost=False,
                  sources=["cls=hladis:prot.npz:1:20000", "mol=gin:mol.npz"])


# ------------------------------------------------------------------- finding the run

def test_a_run_is_found_by_its_source_spec_not_by_its_combo_name(tmp_path):
    """Every cls source produces a combo called "cls+mol". Matching on that name would
    hand the Hladis row whichever cls extractor happened to run last -- the project's
    own "one combo name is not one construction" trap, as a silent wrong number."""
    _run(tmp_path, "p1", "r1", M2OR_TRANS, [1, 2, 3, 4, 5])
    _run(tmp_path, "p2", "r2", M2OR_TRANS | {"sources": ["cls=prosmith:x", "mol=gin:y"]},
         [1, 2, 3, 4, 5])
    assert len(pt.find_runs(tmp_path, "hladis")) == 1
    assert len(pt.find_runs(tmp_path, "prosmith")) == 1
    assert pt.find_runs(tmp_path, "molor") == []


def test_a_run_outside_the_six_cells_is_not_offered_at_all(tmp_path):
    """`curated_full`, or an upstream split family these tables do not use. It is a
    real run, it just is not a row of any of the three tables."""
    _run(tmp_path, "p", "r", dict(regime="curated_full", split="stratified",
                                  task="classification",
                                  sources=["cls=hladis:x"]), [1, 2])
    assert pt.find_runs(tmp_path, "hladis") == []
    _run(tmp_path, "p2", "r2", dict(regime="ofm", dataset="cc", split_family="scaf",
                                    task="regression", sources=["cls=hladis:x"]), [1])
    assert pt.find_runs(tmp_path, "hladis") == []


def test_the_scope_comes_from_the_config_not_the_folder_name(tmp_path):
    """Run folders are named by hand and get renamed; config.json is written from the
    parsed args."""
    _run(tmp_path, "totally-misleading", "cc-rand-whatever",
         dict(regime="full_full", full_full_mode="inductive_molecule_v5",
              task="classification", sources=["cls=hladis:x"]), [42, 43])
    got = pt.find_runs(tmp_path, "hladis")
    assert (got[0]["dataset"], got[0]["regime"]) == ("m2or", "inductive")


def test_a_run_whose_task_disagrees_is_not_put_in_the_table(tmp_path):
    """A regression run cannot supply a row of a classification table; its columns are
    different quantities with the same names nowhere in common."""
    runs = pt.find_runs(tmp_path, "hladis")
    _run(tmp_path, "p", "r", M2OR_TRANS | {"task": "regression"}, [1, 2, 3, 4, 5])
    runs = pt.find_runs(tmp_path, "hladis")
    assert runs and runs[0]["task"] == "regression"
    assert pt.baseline_row(runs, "m2or", "transductive", ["AUROC"],
                           "classification") is None


# ----------------------------------------------------------------------- the pairing

def test_matching_folds_pair_and_the_delta_is_exact(tmp_path):
    df = _grid(boost=0.80, gate=0.85)
    _run(tmp_path, "p", "r", M2OR_TRANS, [1, 2, 3, 4, 5], level=0.72)
    got = pt.baseline_row(pt.find_runs(tmp_path, "hladis"), "m2or", "transductive",
                          ["AUROC"], "classification")
    boost = df[df.arm == "boost_full"].set_index(ag.SPLIT)
    mu, hw, won, n = pt.cross_tree_delta(got["frame"], boost, "AUROC")
    assert n == 5 and won == 0
    assert mu == pytest.approx(0.72 - 0.80)


def test_disjoint_split_ids_refuse_to_be_differenced(tmp_path):
    """The transductive folds are 1-5 and the cold-molecule seeds are 42-46. Two rows
    on those are not two measurements of the same thing, and n=0 is the refusal."""
    df = _grid(folds=(1, 2, 3, 4, 5))
    _run(tmp_path, "p", "r", M2OR_TRANS, [42, 43, 44, 45, 46])
    got = pt.baseline_row(pt.find_runs(tmp_path, "hladis"), "m2or", "transductive",
                          ["AUROC"], "classification")
    boost = df[df.arm == "boost_full"].set_index(ag.SPLIT)
    assert pt.cross_tree_delta(got["frame"], boost, "AUROC")[3] == 0


def test_a_partial_overlap_pairs_on_the_shared_folds_only(tmp_path):
    """And the count says so. This is the quiet case: three folds against five still
    produces a plausible-looking number, and the row's own n is the only hint unless
    the note spells it out."""
    df = _grid(folds=(1, 2, 3, 4, 5))
    _run(tmp_path, "p", "r", M2OR_TRANS, [1, 2, 3])
    got = pt.baseline_row(pt.find_runs(tmp_path, "hladis"), "m2or", "transductive",
                          ["AUROC"], "classification")
    boost = df[df.arm == "boost_full"].set_index(ag.SPLIT)
    assert pt.cross_tree_delta(got["frame"], boost, "AUROC")[3] == 3


def test_the_grid_rows_pair_on_fold_and_seed_together(tmp_path):
    """Inside one tree both the split and the model draw are shared, so both are
    removed. This is a stronger pairing than the cross-tree one and must not silently
    degrade to it."""
    df = _grid()
    two = pd.concat([df, df.assign(seed=43)], ignore_index=True)
    rows, cells = pt.grid_rows(ag.add_series(two), "m2or", "transductive",
                              [1.0], ["AUROC"])
    d = pt.paired_delta(cells["GNN alpha=1"], cells["boost"], "AUROC")
    assert d[3] == 10                      # 5 folds x 2 seeds, not 5


# ------------------------------------------------------------------ what gets printed

def _args(tmp_path, **kw):
    import argparse
    return argparse.Namespace(**(dict(ensemble_root=str(tmp_path), alphas=[1.0],
                                      metrics=None, baseline=["hladis"], combo=None,
                                      delta=False) | kw))


def test_a_missing_baseline_is_a_dash_and_a_note_not_a_vanished_row(tmp_path):
    """A row that disappears reads as "we did not compare"; a dash reads as "we looked
    and there is nothing there", which is what is true."""
    df = _grid(ds="hc", regime="transductive")
    tables = pt.build(df, _args(tmp_path))
    (_, blocks, _, _, notes) = tables[0]
    rows = blocks[0][1]
    assert "Hladis" in [r["model"] for r in rows]
    assert [r for r in rows if r["model"] == "Hladis"][0]["n"] == 0
    assert any("no run under" in n for n in notes)


# ------------------------------------------------------------------- the place column

def test_the_place_is_a_rank_within_each_split_not_a_rank_of_the_means():
    """Folds differ wildly in difficulty -- a cold-molecule fold moves every model by
    0.1 -- so ranking the means would report which folds a model happened to be
    measured on. Here one model wins three folds and loses two by a landslide: its
    MEAN is worst, its PLACE is best."""
    cells = {
        "A": pd.DataFrame({"R2": [0.9, 0.9, 0.9, 0.0, 0.0]},
                          index=pd.Index([1, 2, 3, 4, 5], name="fold")),
        "B": pd.DataFrame({"R2": [0.8, 0.8, 0.8, 9.0, 9.0]},
                          index=pd.Index([1, 2, 3, 4, 5], name="fold")),
    }
    places, dropped, folds = pt.mean_place(cells, {}, "R2")
    assert len(folds) == 5 and not dropped
    assert places["A"] == pytest.approx(1.4)      # 1,1,1,2,2
    assert places["B"] == pytest.approx(1.6)
    assert cells["B"]["R2"].mean() > cells["A"]["R2"].mean()   # the mean disagrees


def test_a_model_on_disjoint_splits_is_excluded_from_the_ranking(tmp_path):
    """A place is a place in ONE competition. A row measured on other folds cannot
    hold one, and inventing it from its own folds would be a rank against nobody."""
    cells = {
        "boost": pd.DataFrame({"R2": [0.5] * 5},
                              index=pd.Index([1, 2, 3, 4, 5], name="fold")),
        "GNN": pd.DataFrame({"R2": [0.6] * 5},
                            index=pd.Index([1, 2, 3, 4, 5], name="fold")),
    }
    base = {"Hladis": pd.DataFrame({"R2": [0.9] * 5}, index=[42, 43, 44, 45, 46])}
    places, dropped, folds = pt.mean_place(cells, base, "R2")
    assert dropped == ["Hladis"] and "Hladis" not in places
    assert len(folds) == 5 and places["GNN"] == pytest.approx(1.0)


def test_places_are_over_the_folds_every_row_shares(tmp_path):
    """A baseline on three of five folds does not shrink the others' competition to
    its own three by accident -- it shrinks it on purpose, and the block header says
    over how many splits the places were taken."""
    cells = {"boost": pd.DataFrame({"R2": [0.5] * 5},
                                   index=pd.Index([1, 2, 3, 4, 5], name="fold"))}
    base = {"Hladis": pd.DataFrame({"R2": [0.9] * 3}, index=[1, 2, 3])}
    places, dropped, folds = pt.mean_place(cells, base, "R2")
    assert folds == [1, 2, 3] and not dropped
    assert places["Hladis"] == pytest.approx(1.0)


def test_the_delta_column_is_still_available_on_request(tmp_path):
    df = _grid(ds="hc", regime="transductive", boost=0.5, gate=0.6)
    tables = pt.build(df, _args(tmp_path, delta=True))
    rows = tables[0][1][0][1]
    gnn = [r for r in rows if r["model"] == "GNN alpha=1"][0]
    assert "place" not in gnn and gnn["delta"][0] == pytest.approx(0.1)


def test_the_combo_matching_the_graph_rows_construction_is_taken(tmp_path):
    """A run with `--combos "1 2 12 123"` wrote several heads under one roof, and
    `cls+prot+mol` usually scores highest -- because it also gets the RAW ESM VECTOR,
    which is most of what `boost` is. Taking the best combo would put a row in the
    table that beats boost partly by containing it. The sweep's graph rows are
    cls+mol, so the baseline row must be cls+mol too."""
    d = tmp_path / "p" / "r"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(M2OR_TRANS), encoding="utf-8")
    rows = [dict(repeat=r, kind="combo", name=n, AUROC=v)
            for r in (1, 2, 3, 4, 5)
            for n, v in (("cls", 0.69), ("cls+mol", 0.73), ("cls+prot+mol", 0.88))]
    pd.DataFrame(rows).to_csv(d / "metrics.csv", index=False)
    got = pt.baseline_row(pt.find_runs(tmp_path, "hladis"), "m2or", "transductive",
                          ["AUROC"], "classification")
    assert got["combo"] == "cls+mol"
    assert float(got["frame"]["AUROC"].iloc[0]) == pytest.approx(0.73)
    assert got["offered"] == ["cls", "cls+mol", "cls+prot+mol"]
    assert got["contaminated"] is False
    named = pt.baseline_row(pt.find_runs(tmp_path, "hladis"), "m2or", "transductive",
                            ["AUROC"], "classification", combo="cls")
    assert named["combo"] == "cls"


def test_a_prot_bearing_combo_is_flagged_when_it_is_all_there_is(tmp_path):
    """Sometimes the only head that ran is cls+prot+mol. Print it -- hiding it is
    worse -- but say what it is, because the number will look like a win."""
    d = tmp_path / "p" / "r"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(M2OR_TRANS), encoding="utf-8")
    pd.DataFrame([dict(repeat=r, kind="combo", name="cls+prot+mol", AUROC=0.90)
                  for r in (1, 2, 3, 4, 5)]).to_csv(d / "metrics.csv", index=False)
    got = pt.baseline_row(pt.find_runs(tmp_path, "hladis"), "m2or", "transductive",
                          ["AUROC"], "classification")
    assert got["combo"] == "cls+prot+mol" and got["contaminated"] is True
    notes = pt._baseline_notes("Hladis", "m2or", "transductive", got, None, "AUROC")
    assert any("RAW PROTEIN VECTOR" in n for n in notes)


def test_a_timing_pool_is_never_a_results_row(tmp_path):
    """`_timing/` holds one-fold stopwatch runs for the params/time table. They are
    newer than the real runs by definition -- they were measured last -- so a
    newest-wins rule hands the baseline row a single fold of a benchmark."""
    _run(tmp_path, "_timing", "t_Hladis", M2OR_TRANS, [1], level=0.93)
    _run(tmp_path, "m2or-trans", "real", M2OR_TRANS, [1, 2, 3, 4, 5], level=0.72)
    runs = pt.find_runs(tmp_path, "hladis")
    assert [r["pool"] for r in runs] == ["m2or-trans"]


def test_the_run_with_the_most_splits_wins_not_the_newest(tmp_path):
    """A rerun of one fold is a normal thing to have on disk, and it is always the
    newest file in the pool."""
    import os
    import time
    _run(tmp_path, "p1", "full", M2OR_TRANS, [1, 2, 3, 4, 5], level=0.72)
    one = _run(tmp_path, "p2", "just_fold3", M2OR_TRANS, [3], level=0.95)
    t = time.time() + 10_000
    os.utime(one / "metrics.csv", (t, t))
    got = pt.baseline_row(pt.find_runs(tmp_path, "hladis"), "m2or", "transductive",
                          ["AUROC"], "classification")
    assert got["frame"].index.nunique() == 5
    assert got["n_runs"] == 2


def test_only_combo_rows_are_read(tmp_path):
    """metrics.csv also carries `ensemble` and `naive` rows. An ensemble row is a
    stacker over several heads and is not the baseline; a naive row is the floor."""
    d = tmp_path / "p" / "r"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(M2OR_TRANS), encoding="utf-8")
    pd.DataFrame([dict(repeat=1, kind="combo", name="cls+mol", AUROC=0.72),
                  dict(repeat=1, kind="ensemble", name="ensemble[logreg]", AUROC=0.99),
                  dict(repeat=1, kind="naive", name="naive[train-mean]", AUROC=0.50)]
                 ).to_csv(d / "metrics.csv", index=False)
    got = pt.baseline_row(pt.find_runs(tmp_path, "hladis"), "m2or", "transductive",
                          ["AUROC"], "classification")
    assert len(got["frame"]) == 1
    assert float(got["frame"]["AUROC"].iloc[0]) == pytest.approx(0.72)


def test_a_tuned_head_is_reported_as_such(tmp_path):
    """`--tune-boost` gives the baseline a per-combo optuna search the sweep's rows
    never had. The number is still worth printing; pretending the two heads are the same
    is not."""
    _run(tmp_path, "p", "r", M2OR_TRANS | {"tune_boost": True}, [1, 2, 3, 4, 5])
    assert pt.find_runs(tmp_path, "hladis")[0]["tuned"] is True
