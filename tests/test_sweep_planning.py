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
    for ds in sweep.FAMILY:                        # every dataset the writer knows: a
                                                   # new one must extend this test, and
                                                   # `cc_shrinked` is why -- its name
                                                   # carries the separator, so the old
                                                   # parser read it as ds=cc and family
                                                   # "shrinked_rand" and dropped a
                                                   # finished sweep on the floor
        for regime in ("transductive", "inductive"):
            for mol in sweep.MOL_SOURCES:          # not a hardcoded pair: a new
                                                   # source must extend this test
                for dial in ("gate", "nodes"):     # `nodes` is what writes _nodedial,
                                                   # the tag every v9/v10 run carries
                    for nodes in ("esm", "onehot"):
                        for variant in (None, "q0cov", "q99greedy"):
                            args = A(mol_source=mol, nodes=nodes, variant=variant,
                                     dial=dial)
                            name = sweep.out_path(ds, regime, args).stem
                            got_ds, family, got_nodes, got_mol, _ = table.parse_name(name)
                            assert got_ds == ds, name
                            assert got_mol == mol, name
                            assert family in table.REGIME_OF, name
                            assert table.REGIME_OF[family] == regime, name
                            if dial == "gate":
                                assert got_nodes == nodes, name
                            else:
                                assert got_nodes == "nodedial", name
                            seen.add(name)
    assert len(seen) > 30


def test_the_reader_knows_every_source_the_writer_can_tag(sweep, table):
    """The one failure mode that costs a whole GPU run and reports nothing.

    `out_path` tags the filename with the molecule source (all but the untagged
    legacy one), and `parse_name` peels that tag off. A source the writer can emit
    and the reader has never heard of does not raise: the family comes back as
    "rand_ecfp", misses REGIME_OF, and the file is dropped with a single counted
    line -- so the cells are computed, written, and never looked at."""
    tagged = set(sweep.MOL_SOURCES) - {sweep.UNTAGGED_MOL}
    assert tagged <= table.KNOWN_MOL, (
        f"the sweep can write {sorted(tagged - table.KNOWN_MOL)} into a filename that "
        f"headline_table.parse_name cannot read back")


def test_paths_pick_the_right_file_per_dataset(sweep):
    for ds in ("cc", "hc"):
        assert sweep.paths(ds, A(mol_source="gin"))[1].endswith(
            f"gin_supervised_contextpred_{ds}.npz")
        assert sweep.paths(ds, A())[1].endswith(f"chemberta_77m_{ds}.npz")
        assert sweep.paths(ds, A())[0].endswith(f"esm3_{ds}.npz")
    # m2or's GIN file is the one name that breaks the {ds} template
    assert sweep.paths("m2or", A(mol_source="gin"))[1].endswith(
        "gin_supervised_contextpred_all_m2or.npz")
    assert sweep.paths("m2or", A())[1].endswith("chemberta_77m_m2or.npz")
    assert sweep.paths("m2or", A())[0].endswith("esm3_m2or.npz")


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
    # a run asking only for the head those rows carry has nothing left at seed 42
    assert sweep.plan([1], A(seeds=[42], alphas=[1.0], combos=["cls+mol"]), done) == []
    jobs = sweep.plan([1], A(seeds=[42, 43], alphas=[1.0], combos=["cls+mol"]), done)
    assert {j[3] for j in jobs} == {43}
    assert len(jobs) == 3
    # asking for BOTH heads backfills seed 42 -- one job per graph cell, the missing
    # head only, flagged as a fill -- and runs seed 43 in full
    jobs = sweep.plan([1], A(seeds=[42, 43], alphas=[1.0]), done)
    s42 = [j for j in jobs if j[3] == 42]
    assert [(j[0], j[4]["combos"], j[4]["fill"]) for j in s42] == [
        ("graph_legacy", ("cls+prot+mol",), True), ("gate", ("cls+prot+mol",), True)]
    assert not any(j[4].get("fill") for j in jobs if j[3] == 43)


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
    # resume now reads the RECORDS file -- it is the only one carrying every split
    m = tmp_path / "metrics_cc_rand_chemberta.csv"
    r = tmp_path / "records_cc_rand_chemberta.csv"
    old.to_csv(r, index=False)

    rows, done = sweep.load_done(r, m, A())
    assert len(rows) == 8
    assert all(k[3] == 42 for k in done)
    assert sweep.plan([1, 2], A(seeds=[42], alphas=[1.0], combos=["cls+mol"]),
                      done) == []
    # the old rows are cls+mol, so a run that wants both heads only BACKFILLS them
    old = sweep.plan([1, 2], A(seeds=[42], alphas=[1.0]), done)
    assert old and all(j[4]["fill"] and j[4]["combos"] == ("cls+prot+mol",)
                       for j in old)
    jobs = sweep.plan([1, 2], A(seeds=[42, 43, 44], alphas=[1.0], combos=["cls+mol"]),
                      done)
    assert {j[3] for j in jobs} == {43, 44}
    assert len(jobs) == 2 * 2 * 3          # 2 new seeds x 2 folds x (baselines+legacy+gate)
    # --force ignores the file entirely
    assert sweep.load_done(r, m, A(force=True)) == ([], set())


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
                             "nodes": "onehot", **extra})
    pd.DataFrame(rows).to_csv(path, index=False)


def _grid(path, boost=0.50, a0=0.50, a1=0.58, **kw):
    """One file in the v8 grid shape: baselines plus the gate at both ends."""
    _fake_run(path, [("boost_full", np.nan, boost), ("naive", np.nan, 0.0),
                     ("gate", 0.0, a0), ("gate", 1.0, a1),
                     ("graph_legacy", np.nan, a1)], **kw)


def _run_table(table, root, **kw):
    """The table as `main` would build it. `all_seeds` defaults to True here so a test
    can use two seeds without tripping the seed-42 view; the default view has its own
    test below."""
    base = dict(dataset=None, regime=None, mol_source=None, metric=None, nodes=None,
                all_metrics=False, compact=False, csv=None, seed=42, all_seeds=True,
                variant=None, all_variants=True, at_alpha=None, legacy=False)
    args = argparse.Namespace(root=str(root), **(base | kw))
    df = table.load(root, args)
    return table.build(df, args), df, args


# --------------------------------------------------------------------------- reader

def test_the_columns_are_the_two_ends_of_the_dial(table, tmp_path):
    """Under the separated design one file holds every column: receptor nodes are
    one-hot throughout, so ESM enters only through the frozen branch and alpha is the
    protein-embedding axis. alpha=0 is structure alone, alpha=1 function alone."""
    _grid(tmp_path / "metrics_cc_rand_chemberta_onehot.csv",
          boost=0.50, a0=0.50, a1=0.58)
    tab, df, args = _run_table(table, tmp_path)
    assert list(tab.model) == ["boost", "alpha=0", "alpha=1"]
    got = {r.model: (r.mean, r.n, r.folds, r.seeds) for r in tab.itertuples()}
    for model, base in (("boost", 0.50), ("alpha=0", 0.50), ("alpha=1", 0.58)):
        mean, n, folds, seeds = got[model]
        assert mean == pytest.approx(base + 0.015 + 0.0005)
        assert (n, folds, seeds) == (4, 2, 2)
    assert {r.model: r.delta for r in tab.itertuples()}["alpha=1"] == pytest.approx(0.08)
    assert not table.checks(df, tab, args)


def test_at_alpha_and_legacy_add_columns_in_the_right_places(table, tmp_path):
    """An intermediate alpha belongs BETWEEN the ends; legacy is not a point on the
    dial at all -- it is the pre-gate model -- so it goes last."""
    _fake_run(tmp_path / "metrics_cc_rand_chemberta_onehot.csv",
              [("boost_full", np.nan, 0.50), ("gate", 0.0, 0.50), ("gate", 0.5, 0.55),
               ("gate", 1.0, 0.58), ("graph_legacy", np.nan, 0.575)])
    tab, _, _ = _run_table(table, tmp_path, at_alpha=[0.5], legacy=True)
    assert list(tab.model) == ["boost", "alpha=0", "alpha=0.5", "alpha=1", "legacy"]
    tab2, _, _ = _run_table(table, tmp_path, at_alpha=[0.0, 1.0])
    assert list(tab2.model) == ["boost", "alpha=0", "alpha=1"], "an end was duplicated"


def test_the_anchor_check_fires_when_alpha_zero_leaves_boost(table, tmp_path):
    """The invariant that replaces the old cross-node boost check, and a sharper one:
    at alpha=0 the receptor vector is a frozen rank-k rotation of the same ESM the
    baseline reads raw, so the two see one body of information through two readers. A
    wide gap means the anchor is not what it claims -- wrong rank, a centred
    projection, or one-hot vectors having reached it by mistake."""
    _grid(tmp_path / "metrics_cc_rand_chemberta_onehot.csv", boost=0.50, a0=0.50)
    tab, df, args = _run_table(table, tmp_path)
    assert not any("ANCHOR" in m for m in table.checks(df, tab, args))
    _grid(tmp_path / "metrics_hc_rand_chemberta_onehot.csv", boost=0.50, a0=0.28)
    tab, df, args = _run_table(table, tmp_path)
    msgs = table.checks(df, tab, args)
    assert any("ANCHOR" in m and "hc" in m for m in msgs)
    assert not any("ANCHOR" in m and "cc" in m for m in msgs)


def test_table_flags_a_missing_arm(table, tmp_path):
    _fake_run(tmp_path / "metrics_hc_our_inductive_chemberta_onehot.csv",
              [("boost_full", np.nan, 0.4), ("gate", 0.0, 0.4)], seeds=(42,))
    tab, df, args = _run_table(table, tmp_path)
    assert any("ABSENT" in m and "alpha=1" in m for m in table.checks(df, tab, args))
    assert tab[tab.model == "alpha=1"].iloc[0].n == 0


def test_table_separates_sources_regimes_and_variants(table, tmp_path):
    for name in ("metrics_cc_rand_chemberta_onehot.csv", "metrics_cc_rand_onehot.csv",
                 "metrics_cc_our_inductive_chemberta_onehot.csv"):
        _grid(tmp_path / name)
    for name in ("metrics_m2or_transductive_q99greedy_chemberta_onehot.csv",
                 "metrics_m2or_transductive_q0cov_chemberta_onehot.csv"):
        _grid(tmp_path / name, boost=0.8, a0=0.8, a1=0.85, task="classification")
    tab, _, _ = _run_table(table, tmp_path)
    keys = {(r.dataset, r.regime, r.mol_source, r.variant_tag) for r in tab.itertuples()}
    assert ("cc", "transductive", "chemberta", "q0cov") in keys
    assert ("cc", "transductive", "gin", "q0cov") in keys
    assert ("cc", "inductive", "chemberta", "q0cov") in keys
    assert ("m2or", "transductive", "chemberta", "q99greedy") in keys
    assert ("m2or", "transductive", "chemberta", "q0cov") in keys
    assert set(tab[tab.dataset == "m2or"].metric) == {"AUROC"}
    assert set(tab[tab.dataset == "cc"].metric) == {"R2"}


def test_the_default_view_is_seed_42_and_the_canonical_edge_variant(table, tmp_path):
    """Two conventions the table leans on, and both are about COMPARABILITY.

    Seed: every reported number was produced at seed 42, so a five-seed series and a
    one-seed one only sit in the same table if the extra seeds are set aside -- pooled,
    a 25-cell row and a 5-cell row are not comparable line for line.

    Variant: m2or has two live edge variants and only q99greedy is of record. q0cov is
    hidden rather than deleted -- it is the evidence that what breaks m2or transductive
    is the edge set, not the protein embedding."""
    _grid(tmp_path / "metrics_cc_rand_chemberta_onehot.csv", seeds=(42, 43, 44))
    for v, val in (("q99greedy", 0.85), ("q0cov", 0.70)):
        _grid(tmp_path / f"metrics_m2or_transductive_{v}_chemberta_onehot.csv",
              boost=0.80, a0=0.80, a1=val, task="classification",
              seeds=(42, 43, 44), folds=(1, 2))
    tab, _, _ = _run_table(table, tmp_path, all_seeds=False, all_variants=False)
    assert set(tab.seeds) <= {0, 1}, "the default view must rest on one seed"
    assert set(tab[tab.dataset == "cc"].n) == {2}             # 2 folds x 1 seed
    assert set(tab.variant_tag) == {"q0cov", "q99greedy"}     # cc's own, and m2or's
    assert (tab[tab.dataset == "m2or"].variant_tag == "q99greedy").all()
    # and both are recoverable
    wide, _, _ = _run_table(table, tmp_path, all_seeds=True, all_variants=True)
    assert set(wide.seeds) <= {0, 3}
    assert set(wide[wide.dataset == "m2or"].variant_tag) == {"q99greedy", "q0cov"}


def test_a_blank_variant_is_the_datasets_own_default_not_a_second_one(table, tmp_path):
    """The insect files written before the `variant` column existed carry a blank, and
    a blank read literally splits one dataset into two rows for the same edge set."""
    _grid(tmp_path / "metrics_cc_rand_onehot.csv", seeds=(42,))       # old, no variant
    _grid(tmp_path / "metrics_cc_rand_chemberta_onehot.csv",          # new, records it
          seeds=(42,), variant="q0cov")
    tab, _, _ = _run_table(table, tmp_path, all_seeds=False, all_variants=False)
    assert set(tab.variant_tag) == {"q0cov"}
    assert set(tab.mol_source) == {"gin", "chemberta"}


def test_failed_cells_are_excluded_and_reported(table, tmp_path):
    p = tmp_path / "metrics_cc_rand_chemberta_onehot.csv"
    _grid(p)
    df = pd.read_csv(p)
    hit = (df.arm == "gate") & np.isclose(df.alpha.fillna(-1), 1.0) & (df.fold == 1)
    df.loc[hit, ["status", "R2"]] = ["failed: boom", np.nan]
    df.to_csv(p, index=False)
    tab, full, args = _run_table(table, tmp_path)
    assert tab[tab.model == "alpha=1"].iloc[0].n == 2      # fold 2 only, both seeds
    assert any("FAILED" in m for m in table.checks(full, tab, args))
