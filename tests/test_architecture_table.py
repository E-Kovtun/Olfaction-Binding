"""The architecture sweep and its six-number table.

What is load-bearing here, and what these tests pin:

* the graph must NOT move when the operator does. `VARIANT` is the construction the
  main tables use, per dataset, and it is read from the module rather than from the
  command line;
* the unit of evidence: seeds averaged inside a fold, interval over folds. Four
  operators sit close together, so a halved interval is exactly how this table would
  manufacture a winner;
* the anchor is not a competitor. The boosting base is the ground the graphs are read
  against and must never be marked as the best OPERATOR;
* the resume key includes the width, because two widths of one operator are two models.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


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
    return _spec("_archsweep", "scripts/article_sweeps/s5_run_architecture.py")


@pytest.fixture(scope="module")
def reader():
    return _spec("_archtable", "scripts/article_tables/s5_architecture.py")


# ----------------------------------------------------------------- fixtures

ARCHS = ["sage:paper", "gat:paper", "graphconv:paper", "gin:paper", "sage"]


def frame(dataset="m2or", metric="AUROC", base=0.80, step=0.01, folds=(1, 2, 3, 4, 5),
          seeds=(42, 43), archs=ARCHS, hidden=256, spread=0.0):
    """A cell's CSV: every operator on every fold and seed, plus the anchor.

    Operator i scores base + i*step. With `spread` the folds disagree by that much,
    the same way for every operator -- which is what gives the rows a non-zero
    interval without changing their order.
    """
    rows = []
    for fold in folds:
        wob = spread * ((fold % 2) * 2 - 1)
        for seed in seeds:
            rows.append(dict(dataset=dataset, regime="transductive", conv="boost_full",
                             hidden=np.nan, arch="boost_full", fold=fold, seed=seed,
                             combo="prot+mol", split="test", status="ok",
                             **{metric: base + wob}))
            for i, a in enumerate(archs):
                rows.append(dict(dataset=dataset, regime="transductive", conv=a,
                                 hidden=hidden, arch=a, fold=fold, seed=seed,
                                 combo="cls+mol", split="test", status="ok",
                                 **{metric: base + (i + 1) * step + wob}))
                # a val row that must never reach the table
                rows.append(dict(dataset=dataset, regime="transductive", conv=a,
                                 hidden=hidden, arch=a, fold=fold, seed=seed,
                                 combo="cls+mol", split="val", status="ok",
                                 **{metric: 0.99}))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- the sweep

def test_the_graph_is_pinned_per_dataset(sweep):
    """The construction the main tables use. If this drifts, every row of the table
    compares an operator on one graph with an operator on another."""
    assert sweep.VARIANT["m2or"] == dict(q=0.99, criterion="greedy_pair_cover",
                                         k_mode="coverage_quantile")
    for ds in ("cc", "hc"):
        assert sweep.VARIANT[ds]["q"] == 0.0
        assert sweep.VARIANT[ds]["criterion"] == "coverage"


def test_the_construction_is_not_a_command_line_knob(sweep):
    """It is pinned in the module on purpose: an operator comparison run on two
    different graphs is not one."""
    flags = {a.dest for a in sweep.parser()._actions}
    assert not ({"q", "quantiles", "criterion", "criteria"} & flags)


def test_every_operator_is_offered(sweep):
    """Every operator is a default row. `none` is in CONVS but is not an operator --
    it is the no-message-passing control, and it is run on request, not by default."""
    from orbind.gnn_extractor import CONVS
    specs = sweep.parser().parse_args([]).conv
    assert [s.split(":")[0] for s in specs] == [c for c in CONVS if c != "none"]
    assert all(s.endswith(":paper") for s in specs), (
        "every default row trains in the regime the project reports; an un-suffixed "
        "spec is the historical encoder and does not belong in the table")
    assert specs[0] == "sage:paper", "ours leads the list, as it leads the table"


def test_a_spec_is_parsed_and_a_bad_one_is_refused(sweep):
    plain = dict(paper=False, edges="signed", layers=2)
    assert sweep.parse_spec("sage") == ("sage", plain)
    assert sweep.parse_spec("gat:paper") == ("gat", dict(plain, paper=True))
    assert sweep.parse_spec("sage:paper:pos") == (
        "sage", dict(paper=True, edges="positive", layers=2))
    assert sweep.parse_spec("sage:paper:1layer") == (
        "sage", dict(paper=True, edges="signed", layers=1))
    with pytest.raises(SystemExit, match="unknown operator"):
        sweep.parse_spec("gcn")
    with pytest.raises(SystemExit, match="unknown suffix"):
        sweep.parse_spec("sage:sampled")


def test_a_spec_cannot_ablate_twice_or_ablate_nothing(sweep):
    """Two edge modes in one row is not a row, and `none` has no layers to thin."""
    with pytest.raises(SystemExit, match="sets 'edges' twice"):
        sweep.parse_spec("sage:pos:unsigned")
    with pytest.raises(SystemExit, match="takes no suffix"):
        sweep.parse_spec("none:paper")


def test_the_ablation_specs_are_not_run_by_default(sweep):
    """They answer a different question from 'which operator', they cost a run each,
    and an old command must not quietly start training them."""
    assert set(sweep.ABLATION_SPECS).isdisjoint(sweep.DEFAULT_SPECS)
    assert "none" in sweep.ABLATION_SPECS


def test_the_paper_rows_add_sampling_and_normalisation(sweep):
    """Both, together, and nothing else -- the two things our encoder took from
    neither GraphSAGE nor GAT."""
    assert sweep.PAPER == dict(fanout=(25, 10), normalize_layers=True)
    assert sweep.parser().parse_args([]).fanout == [25, 10]


def test_a_paper_row_never_shares_a_name_with_its_plain_row(sweep):
    paper = sweep.parse_spec("sage:paper")[1]
    plain = sweep.parse_spec("sage")[1]
    assert sweep.arch_label("sage", 256, paper) == "sage:paper"
    assert sweep.arch_label("sage", 256, plain) == "sage"
    assert sweep.arch_label("gat", 512, sweep.parse_spec("gat:paper")[1]) \
        == "gat@512:paper"


def test_a_row_has_one_name_however_the_spec_was_spelled(sweep):
    """The suffix order in a label is canonical, so `sage:pos:paper` and
    `sage:paper:pos` are one cached row and not two."""
    a = sweep.arch_label("sage", 256, sweep.parse_spec("sage:paper:pos")[1])
    b = sweep.arch_label("sage", 256, sweep.parse_spec("sage:pos:paper")[1])
    assert a == b == "sage:paper:pos"


def test_the_width_is_in_the_row_name_only_when_it_is_not_the_reported_one(sweep):
    plain = sweep.parse_spec("gat")[1]
    assert sweep.arch_label("gat", 256, plain) == "gat"
    assert sweep.arch_label("gat", 512, plain) == "gat@512"


def test_the_resume_key_separates_widths_heads_and_regimes(sweep):
    base = dict(conv="gat", arch="gat", hidden=256, fold=1, seed=42,
                combo="cls+mol", split="test")
    assert sweep.cell_key(base) != sweep.cell_key({**base, "arch": "gat:paper"})
    k = sweep.cell_key(base)
    assert k != sweep.cell_key({**base, "hidden": 512, "arch": "gat@512"})
    assert k != sweep.cell_key({**base, "combo": "cls+prot+mol"})
    assert k != sweep.cell_key({**base, "split": "val"})
    assert k == sweep.cell_key(dict(base))


def test_the_anchor_key_has_no_width(sweep):
    assert sweep.cell_key(dict(conv="boost_full", arch="boost_full", hidden=np.nan,
                               fold=1, seed=42, combo="prot+mol",
                               split="test"))[1] is None


def test_planning_skips_what_is_already_done(sweep):
    args = sweep.parser().parse_args(["--conv", "sage", "gat", "--seeds", "42"])
    done = {("sage", 256, 1, 42, "cls+mol", "test"),
            ("boost_full", None, 1, 42, "prot+mol", "test")}
    jobs = sweep.plan([1], done, args)
    assert jobs == [("gnn", 1, 42, "gat", 256)]


def test_a_paper_row_is_planned_separately_from_its_plain_row(sweep):
    """Having trained `sage` must not mark `sage:paper` as done."""
    args = sweep.parser().parse_args(["--conv", "sage", "sage:paper", "--seeds", "42",
                                      "--no-boost-full"])
    done = {("sage", 256, 1, 42, "cls+mol", "test")}
    assert sweep.plan([1], done, args) == [("gnn", 1, 42, "sage:paper", 256)]


def test_planning_covers_the_width_grid(sweep):
    args = sweep.parser().parse_args(["--conv", "sage", "--hidden", "128", "256",
                                      "--seeds", "42", "--no-boost-full"])
    jobs = sweep.plan([1], set(), args)
    assert sorted(j[4] for j in jobs) == [128, 256]


def test_an_old_csv_is_refused_rather_than_resumed_onto(sweep, tmp_path):
    """A file without the operator columns predates this script; resuming onto it
    would mix two row formats and nothing would say so."""
    p = tmp_path / "metrics_m2or_transductive__chemberta.csv"
    pd.DataFrame([dict(fold=1, seed=42, combo="cls+mol", split="test")]).to_csv(
        p, index=False)
    with pytest.raises(SystemExit, match="predates this script"):
        sweep.load_done(p)


# ----------------------------------------------------------------- the table

def test_the_table_reads_test_only(reader):
    """The val rows in the fixture score 0.99; a table that let them in would say so
    loudly and be wrong quietly."""
    t, metric = reader.cell_table(frame(), "m2or", "cls+mol")
    assert metric == "AUROC"
    assert t["value"].max() < 0.9


def test_seeds_are_averaged_inside_a_fold(reader):
    """Five folds and two seeds is five observations, not ten. With ten the interval
    would come out about 1/sqrt(2) as wide, and this table lives or dies on widths."""
    t, _ = reader.cell_table(frame(), "m2or", "cls+mol")
    assert (t["folds"] == 5).all()
    assert (t["seeds"] == 2).all()


def test_the_anchor_survives_the_head_filter(reader):
    """It carries `prot+mol` and has no `cls` to concatenate, so filtering by combo
    alone would silently drop the row the whole table is read against."""
    t, _ = reader.cell_table(frame(), "m2or", "cls+mol")
    assert reader.ANCHOR in set(t["arch"])


def test_rows_are_ordered_base_then_ours_then_the_rest(reader):
    t, _ = reader.cell_table(frame(), "m2or", "cls+mol")
    assert list(t["arch"])[:2] == ["boost_full", "sage:paper"]


def test_the_anchor_is_never_marked_as_the_best_operator(reader):
    """Make the base the highest number in the column: it still must not be starred,
    because it is the ground and not a competitor."""
    df = frame(base=0.99, step=-0.01)
    t, metric = reader.cell_table(df, "m2or", "cls+mol")
    rows, cols, wide, labels, ranks = reader.combine([("m2or", "transductive",
                                                       metric, t)])
    assert reader.best_in_column(rows, cols[0], wide) != reader.ANCHOR


def test_the_best_is_direction_aware(reader):
    """On an error metric the winner is the SMALLEST. Carey reports R2 by default, so
    this asks the question directly of the helper."""
    from scripts.analysis import alpha_grid as ag
    assert "RMSE" in ag.LOWER_IS_BETTER
    wide = {"sage": {("cc", "transductive", "RMSE"): (0.5, 0.01)},
            "gat": {("cc", "transductive", "RMSE"): (0.9, 0.01)}}
    assert reader.best_in_column(["sage", "gat"], ("cc", "transductive", "RMSE"),
                                 wide) == "sage"


def test_a_cell_missing_an_operator_keeps_the_row_visible(reader):
    """`--` means not trained here; it must not be confused with trained and bad."""
    a, ma = reader.cell_table(frame(), "m2or", "cls+mol")
    b, mb = reader.cell_table(frame(archs=["sage", "gat"]), "m2or", "cls+mol")
    rows, cols, wide, labels, ranks = reader.combine(
        [("m2or", "transductive", ma, a), ("m2or", "inductive", mb, b)])
    assert "gin:paper" in rows
    assert reader.fmt(wide["gin:paper"].get(cols[1])) == "--"


def test_overlap_is_computed_and_not_asserted(reader):
    col = ("m2or", "transductive", "AUROC")
    wide = {"sage": {col: (0.80, 0.02)}, "gat": {col: (0.81, 0.02)},
            "gin": {col: (0.90, 0.01)}}
    assert reader.overlaps(wide, col, "sage", "gat") is True
    assert reader.overlaps(wide, col, "sage", "gin") is False
    assert reader.overlaps(wide, col, "sage", "absent") is None


def test_the_text_block_names_the_columns_that_separate_nothing(reader, capsys):
    """When ours and the marked operator overlap, the table says so in words rather
    than leaving a reader to compare two intervals by eye."""
    t, metric = reader.cell_table(frame(base=0.80, step=0.001, spread=0.02),
                                  "m2or", "cls+mol")
    rows, cols, wide, labels, ranks = reader.combine([("m2or", "transductive",
                                                       metric, t)])
    block = reader.text(rows, cols, wide, labels, ranks)
    assert "separate nothing" in block


def test_the_width_row_is_labelled_as_a_width(reader):
    assert reader.label("gat@512") == "GAT, width 512 (full neighbourhood, un-normalised)"
    assert reader.label("sage") == "GraphSAGE (full neighbourhood, un-normalised)"


def test_an_ablation_row_says_what_it_ablated(reader):
    assert reader.label("sage:paper:pos") == "GraphSAGE, positive edges only"
    assert reader.label("sage:paper:unsigned") == "GraphSAGE, unsigned edges"
    assert reader.label("sage:paper:1layer") == "GraphSAGE, one layer"
    assert reader.label("none") == "No message passing"


def test_the_base_is_ranked_and_the_ablations_are_not(reader):
    """The base is one of the answers to "which would you rather have", so it takes
    part. An encoder ablation is a part of our own row, and folding it in would move
    every operator's number each time another ablation is added."""
    assert reader.is_ranked("sage:paper") and reader.is_ranked("gin:paper")
    assert reader.is_ranked("boost_full")
    assert not reader.is_ranked("none")
    assert not reader.is_ranked("sage:paper:pos")


def test_only_the_operators_are_operators(reader):
    assert reader.is_operator("sage:paper") and reader.is_operator("gin:paper")
    assert not reader.is_operator("boost_full")
    assert not reader.is_operator("none")
    assert not reader.is_operator("sage:paper:pos")


def test_the_blocks_read_as_the_argument(reader):
    """Base, us, the alternatives, the parts of us, no graph at all."""
    rows = ["none", "gin:paper", "sage:paper:pos", "boost_full", "sage:paper"]
    assert sorted(rows, key=reader.row_order) == [
        "boost_full", "sage:paper", "gin:paper", "sage:paper:pos", "none"]


def test_the_control_survives_the_reported_regime_filter(reader):
    """`none` carries no `:paper` because it has nothing to sample or normalise, and
    dropping it as 'historical' would delete the control."""
    import pandas as pd
    d = pd.DataFrame({"arch": ["sage:paper", "sage", "none", "boost_full",
                               "sage:paper:pos"]})
    assert set(reader.reported_only(d)["arch"]) == {
        "sage:paper", "none", "boost_full", "sage:paper:pos"}


def test_the_reported_regime_is_not_announced_in_the_label(reader):
    """It is the only regime in the table, so naming it on every row would be noise --
    and calling it "ours" would claim a variant this project no longer has."""
    assert reader.label("sage:paper") == "GraphSAGE"
    assert reader.label("gat@512:paper") == "GAT, width 512"


def test_a_historical_row_is_marked_if_one_turns_up(reader):
    """The un-suffixed rows are a different encoder. They are not in the default table,
    but a CSV can still hold them, and an unmarked row would make the table read as a
    comparison of operators when it was partly a comparison of regimes."""
    assert reader.label("sage") == "GraphSAGE (full neighbourhood, un-normalised)"


def test_nothing_on_disk_prints_the_command_and_fails(reader, tmp_path, capsys):
    rc = reader.main(["--root", str(tmp_path), "--out", str(tmp_path / "out"),
                      "--dataset", "m2or", "--regime", "transductive"])
    assert rc == 1
    assert "s5_run_architecture.py" in capsys.readouterr().out


def test_the_end_to_end_render_writes_its_three_files(reader, tmp_path):
    root = tmp_path / "sweep"
    root.mkdir()
    frame().to_csv(root / "metrics_m2or_transductive__chemberta.csv", index=False)
    out = tmp_path / "out"
    assert reader.main(["--root", str(root), "--out", str(out),
                        "--dataset", "m2or", "--regime", "transductive"]) == 0
    for name in ("architecture_long.csv", "architecture.tex", "architecture.txt"):
        assert (out / name).exists()
    tex = (out / "architecture.tex").read_text(encoding="utf-8")
    assert r"\label{tab:arch}" in tex
    assert "GraphSAGE" in tex
    assert "ours" not in tex
