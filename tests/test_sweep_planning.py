"""The bookkeeping around the alpha sweep, which is where a silent wrong answer lives.

None of this trains anything. It pins the three things that decide WHICH numbers a
reader ends up looking at:

1. the output filename -- two runs that differ in molecule source, node features or
   edge variant must not share a file;
2. the resume key -- a cell already on disk is skipped, so anything that changes what
   is computed and is not in the filename has to be in the key. The seed was not, which
   meant a second seed silently re-reported the first one's rows;
3. `parse_name` in the reader -- it has to invert every name `out_path` can emit, or a
   whole series is dropped from the table with one skipped-file line.

The reader is then run end to end on fabricated CSVs, because its means, its paired
deltas and its consistency checks are what the headline table IS.
"""
import argparse
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(relpath, name):
    """Import a script by path, stubbing torch_geometric -- the only dependency of the
    training modules that is not installed everywhere. Nothing under test touches it."""
    try:                                    # never shadow a real install -- on the
        import torch_geometric               # server this import succeeds and the
        _ = torch_geometric                  # stub below is not built at all
    except ImportError:
        import torch
        tg = types.ModuleType("torch_geometric")
        nn = types.ModuleType("torch_geometric.nn")
        for cls in ("HeteroConv", "MessagePassing", "SAGEConv"):
            setattr(nn, cls, type(cls, (torch.nn.Module,), {}))
        tg.nn = nn
        sys.modules["torch_geometric"], sys.modules["torch_geometric.nn"] = tg, nn
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sweep():
    return _load("scripts/modeling/train/run_alpha_gate_sweep.py", "ag_sweep")


@pytest.fixture(scope="module")
def table():
    return _load("scripts/analysis/headline_table.py", "ag_table")


def A(**kw):
    """Sweep args with the defaults the parser would give."""
    base = dict(nodes="esm", mol_source="chemberta", variant=None, out=None,
                seeds=[42], folds=None, n_repeats=None, baselines_only=False,
                legacy=True, gate=True,
                alphas=[0.0, 1.0], force=False, prot_embeddings=None,
                mol_embeddings=None)
    return argparse.Namespace(**(base | kw))


# --------------------------------------------------------------------------- naming

@pytest.mark.parametrize("kw,ds,regime,expect", [
    # GIN stays untagged: the files written before --mol-source existed keep their names
    (dict(mol_source="gin"), "cc", "transductive", "metrics_cc_rand.csv"),
    (dict(mol_source="gin"), "hc", "inductive", "metrics_hc_our_inductive.csv"),
    (dict(mol_source="gin", nodes="onehot"), "cc", "transductive",
     "metrics_cc_rand_onehot.csv"),
    (dict(mol_source="gin"), "m2or", "transductive",
     "metrics_m2or_transductive_q99greedy.csv"),
    (dict(mol_source="gin", variant="q0cov", nodes="onehot"), "m2or", "inductive",
     "metrics_m2or_inductive_molecule_v5_q0cov_onehot.csv"),
    # ChemBERTa is tagged, and the order is variant, source, nodes
    (dict(), "cc", "transductive", "metrics_cc_rand_chemberta.csv"),
    (dict(nodes="onehot"), "hc", "inductive",
     "metrics_hc_our_inductive_chemberta_onehot.csv"),
    (dict(nodes="onehot"), "m2or", "inductive",
     "metrics_m2or_inductive_molecule_v5_q99greedy_chemberta_onehot.csv"),
])
def test_output_names_separate_every_axis(sweep, kw, ds, regime, expect):
    assert sweep.out_path(ds, regime, A(**kw)).name == expect


def test_every_emitted_name_is_readable_back(sweep, table):
    """The reader's parser must invert the writer. A name it cannot parse is not an
    error anywhere -- the file is just quietly absent from the table."""
    seen = set()
    for ds, regimes in (("cc", ["transductive", "inductive"]),
                        ("hc", ["transductive", "inductive"]),
                        ("m2or", ["transductive", "inductive"])):
        for regime in regimes:
            for mol in ("gin", "chemberta"):
                for nodes in ("esm", "onehot"):
                    for variant in (None, "q0cov", "q99greedy"):
                        args = A(mol_source=mol, nodes=nodes, variant=variant)
                        name = sweep.out_path(ds, regime, args).stem
                        got_ds, family, got_nodes, got_mol, _ = table.parse_name(name)
                        assert got_ds == ds, name
                        assert got_nodes == nodes, name
                        assert got_mol == mol, name
                        assert family in table.REGIME_OF, name
                        assert table.REGIME_OF[family] == regime, name
                        seen.add(name)
    assert len(seen) > 30


def test_paths_pick_the_right_file_per_dataset(sweep):
    for ds in ("cc", "hc"):
        assert sweep.paths(ds, A(mol_source="gin"))[1].endswith(
            f"gin_supervised_contextpred_{ds}.npz")
        assert sweep.paths(ds, A())[1].endswith(f"chemberta_77m_{ds}.npz")
        assert sweep.paths(ds, A())[0].endswith(f"esm1b_650m_mean_{ds}.npz")
    # m2or's GIN file is the one name that breaks the {ds} template; its protein file
    # carries no suffix at all
    assert sweep.paths("m2or", A(mol_source="gin"))[1].endswith(
        "gin_supervised_contextpred_all_m2or.npz")
    assert sweep.paths("m2or", A())[1].endswith("chemberta_77m_m2or.npz")
    assert sweep.paths("m2or", A())[0].endswith("esm1b_650m_mean.npz")


def test_n_repeats_slices_each_datasets_own_repeat_list(sweep):
    """The trap `--n-repeats` exists to close: m2or/inductive's repeats are the
    cold-molecule seeds 42-46, so a literal `--folds 1 2 3` meant as "run it cheaper"
    asks it for splits that do not exist, while the same flag is correct everywhere
    else in the same invocation."""
    assert sweep.repeats("cc", "inductive", A()) == [1, 2, 3, 4, 5]
    assert sweep.repeats("m2or", "inductive", A()) == [42, 43, 44, 45, 46]
    assert sweep.repeats("cc", "inductive", A(n_repeats=3)) == [1, 2, 3]
    assert sweep.repeats("m2or", "inductive", A(n_repeats=3)) == [42, 43, 44]
    assert sweep.repeats("m2or", "transductive", A(n_repeats=3)) == [1, 2, 3]
    # an explicit --folds still wins, for the single-dataset case it is meant for
    assert sweep.repeats("m2or", "inductive", A(folds=[42, 46], n_repeats=3)) == [42, 46]


# ------------------------------------------------------------------------- planning

def test_plan_counts_folds_times_seeds(sweep):
    jobs = sweep.plan([1, 2, 3], A(seeds=[42, 43], alphas=[0.0, 0.5, 1.0]), set())
    # per (fold, seed): 1 baselines + 1 legacy + 3 gates
    assert len(jobs) == 3 * 2 * 5
    assert {j[3] for j in jobs} == {42, 43}
    assert {j[2] for j in jobs} == {1, 2, 3}
    # seed-major, so an interrupted sweep leaves whole seeds rather than a ragged slice
    assert [j[3] for j in jobs] == sorted([j[3] for j in jobs])


def test_no_gate_and_baselines_only_trim_the_grid(sweep):
    assert {j[0] for j in sweep.plan([1], A(gate=False), set())} == {"baselines",
                                                                    "graph_legacy"}
    assert {j[0] for j in sweep.plan([1], A(baselines_only=True), set())} == {"baselines"}
    assert {j[0] for j in sweep.plan([1], A(legacy=False), set())} == {"baselines", "gate"}


def test_a_finished_cell_is_skipped_but_a_new_seed_is_not(sweep):
    done = {sweep.key("gate", 1.0, 1, 42), sweep.key("graph_legacy", None, 1, 42),
            sweep.key("boost_full", None, 1, 42), sweep.key("naive", None, 1, 42)}
    assert sweep.plan([1], A(seeds=[42], alphas=[1.0]), done) == []
    jobs = sweep.plan([1], A(seeds=[42, 43], alphas=[1.0]), done)
    assert {j[3] for j in jobs} == {43}
    assert len(jobs) == 3


def test_a_seedless_csv_is_read_as_seed_42(sweep, tmp_path):
    """The regression test for the bug this refactor exists to fix. Every CSV already on
    the server predates the seed column and holds exactly seed 42 -- the ensembler's own
    default. Read any other way, a five-seed rerun either recomputes everything or, far
    worse, skips the four new seeds and reports seed 42's numbers five times."""
    old = pd.DataFrame([{"arm": "boost_full", "alpha": np.nan, "fold": f, "status": "ok",
                         "R2": 0.5} for f in (1, 2)]
                       + [{"arm": "gate", "alpha": 1.0, "fold": f, "status": "ok",
                           "R2": 0.6} for f in (1, 2)]
                       + [{"arm": "graph_legacy", "alpha": np.nan, "fold": f,
                           "status": "ok", "R2": 0.55} for f in (1, 2)]
                       + [{"arm": "naive", "alpha": np.nan, "fold": f, "status": "ok",
                           "R2": 0.0} for f in (1, 2)])
    p = tmp_path / "metrics_cc_rand_chemberta.csv"
    old.to_csv(p, index=False)

    rows, done = sweep.load_done(p)
    assert len(rows) == 8
    assert all(k[3] == 42 for k in done)
    assert sweep.plan([1, 2], A(seeds=[42], alphas=[1.0]), done) == []
    jobs = sweep.plan([1, 2], A(seeds=[42, 43, 44], alphas=[1.0]), done)
    assert {j[3] for j in jobs} == {43, 44}
    assert len(jobs) == 2 * 2 * 3          # 2 new seeds x 2 folds x (baselines+legacy+gate)
    # --force ignores the file entirely
    assert sweep.load_done(p, force=True) == ([], set())


# --------------------------------------------------------------------------- reader

def _fake_run(path, arms, folds=(1, 2), seeds=(42, 43), task="regression", **extra):
    """A metrics CSV shaped like the sweep's own, with a fixed value per arm."""
    met = "R2" if task == "regression" else "AUROC"
    rows = []
    for (arm, alpha, val) in arms:
        for f in folds:
            for s in seeds:
                rows.append({"arm": arm, "alpha": alpha, "fold": f, "seed": s,
                             "status": "ok", met: val + 0.01 * f + 0.001 * (s - 42),
                             **extra})
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_table(table, root, **kw):
    args = argparse.Namespace(root=str(root), dataset=None, regime=None,
                              mol_source=None, metric=None, all_metrics=False,
                              compact=False, csv=None, **kw)
    return table.build(table.load(root, args), args), table.load(root, args), args


def test_table_picks_the_three_models_from_the_two_node_runs(table, tmp_path):
    _fake_run(tmp_path / "metrics_cc_rand_chemberta.csv",
              [("boost_full", np.nan, 0.50), ("naive", np.nan, 0.0),
               ("graph_legacy", np.nan, 0.60)])
    _fake_run(tmp_path / "metrics_cc_rand_chemberta_onehot.csv",
              [("boost_full", np.nan, 0.50), ("naive", np.nan, 0.0),
               ("gate", 1.0, 0.58)])
    tab, df, args = _run_table(table, tmp_path)
    assert set(tab.model) == {"boost", "GNN old", "GNN new"}
    got = {r.model: (r.mean, r.n, r.folds, r.seeds) for r in tab.itertuples()}
    for model, base in (("boost", 0.50), ("GNN old", 0.60), ("GNN new", 0.58)):
        mean, n, folds, seeds = got[model]
        # the fixture's cell value is base + 0.01*fold + 0.001*(seed-42)
        assert mean == pytest.approx(base + 0.015 + 0.0005)
        assert (n, folds, seeds) == (4, 2, 2)
    # paired against boost on the cells they share, not a difference of two means
    d = {r.model: r.delta for r in tab.itertuples()}
    assert d["GNN old"] == pytest.approx(0.10)
    assert d["GNN new"] == pytest.approx(0.08)
    assert {r.won for r in tab.itertuples() if r.model == "GNN old"} == {4}
    assert not table.checks(df, tab)


def test_table_falls_back_to_legacy_for_the_onehot_graph_and_says_so(table, tmp_path):
    _fake_run(tmp_path / "metrics_cc_rand_chemberta.csv",
              [("boost_full", np.nan, 0.50), ("graph_legacy", np.nan, 0.60)])
    _fake_run(tmp_path / "metrics_cc_rand_chemberta_onehot.csv",
              [("boost_full", np.nan, 0.50), ("graph_legacy", np.nan, 0.58)])
    tab, df, _ = _run_table(table, tmp_path)
    new = tab[tab.model == "GNN new"].iloc[0]
    assert new.n == 4 and new.fallback == "legacy"
    assert any("SUBST" in m for m in table.checks(df, tab))


def test_table_flags_a_single_seed_and_a_missing_arm(table, tmp_path):
    _fake_run(tmp_path / "metrics_hc_our_inductive_chemberta.csv",
              [("boost_full", np.nan, 0.4), ("graph_legacy", np.nan, 0.45)],
              seeds=(42,))
    tab, df, _ = _run_table(table, tmp_path)
    msgs = table.checks(df, tab)
    assert any("1 SEED" in m for m in msgs)
    assert any("ABSENT" in m and "GNN new" in m for m in msgs)
    assert tab[tab.model == "GNN new"].iloc[0].n == 0


def test_table_catches_a_boost_that_disagrees_across_node_runs(table, tmp_path):
    """boost never sees the node features, so its two copies are the same computation.
    If they differ, the two files are not the pair the table assumes they are -- a
    different molecule source, a different pool fold, a stale file."""
    _fake_run(tmp_path / "metrics_cc_rand_chemberta.csv",
              [("boost_full", np.nan, 0.50), ("graph_legacy", np.nan, 0.60)])
    _fake_run(tmp_path / "metrics_cc_rand_chemberta_onehot.csv",
              [("boost_full", np.nan, 0.53), ("gate", 1.0, 0.58)])
    tab, df, _ = _run_table(table, tmp_path)
    assert any("MISMATCH" in m for m in table.checks(df, tab))


def test_table_separates_sources_regimes_and_variants(table, tmp_path):
    for name in ("metrics_cc_rand_chemberta.csv", "metrics_cc_rand.csv",
                 "metrics_cc_our_inductive_chemberta.csv"):
        _fake_run(tmp_path / name, [("boost_full", np.nan, 0.5),
                                    ("graph_legacy", np.nan, 0.6)])
    for name in ("metrics_m2or_transductive_q99greedy_chemberta.csv",
                 "metrics_m2or_transductive_q0cov_chemberta.csv"):
        _fake_run(tmp_path / name, [("boost_full", np.nan, 0.8),
                                    ("graph_legacy", np.nan, 0.85)], task="classification")
    tab, df, _ = _run_table(table, tmp_path)
    keys = {(r.dataset, r.regime, r.mol_source, r.variant_tag) for r in tab.itertuples()}
    assert ("cc", "transductive", "chemberta", "") in keys
    assert ("cc", "transductive", "gin", "") in keys
    assert ("cc", "inductive", "chemberta", "") in keys
    assert ("m2or", "transductive", "chemberta", "q99greedy") in keys
    assert ("m2or", "transductive", "chemberta", "q0cov") in keys
    # the metric of record follows the task family, per group
    assert set(tab[tab.dataset == "m2or"].metric) == {"AUROC"}
    assert set(tab[tab.dataset == "cc"].metric) == {"R2"}


def test_failed_cells_are_excluded_and_reported(table, tmp_path):
    p = tmp_path / "metrics_cc_rand_chemberta.csv"
    _fake_run(p, [("boost_full", np.nan, 0.50), ("graph_legacy", np.nan, 0.60)])
    df = pd.read_csv(p)
    df.loc[(df.arm == "graph_legacy") & (df.fold == 1), ["status", "R2"]] = \
        ["failed: boom", np.nan]
    df.to_csv(p, index=False)
    tab, full, _ = _run_table(table, tmp_path)
    assert tab[tab.model == "GNN old"].iloc[0].n == 2      # fold 2 only, both seeds
    assert any("FAILED" in m for m in table.checks(full, tab))
