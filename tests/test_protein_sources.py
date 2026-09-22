"""The protein-representation table (A2) and the GNN rows that feed it.

Two things here are load-bearing and neither is arithmetic:

* our rows must be TRAINED IN THIS SCRIPT'S FOLDS. The reader therefore never
  imports a receptor vector from another run, and the compute script's spec parser
  is what pins which graph is built (source, dial position, edge variant);
* the unit of evidence is the fold. Both the boost seed and the graph seed are
  averaged inside a fold before any interval is taken.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, _root / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T = _mod("_protsrc", "scripts/article_tables/07_protein_sources.py")


@pytest.fixture(scope="module")
def sweep():
    """The compute half. It imports torch only inside `gnn_rows`, so this is cheap."""
    return _mod("_protfloor", "scripts/modeling/analysis/prot_floor_sweep.py")


METRICS = ["AUROC", "AUPRC", "MCC", "F1"]


def frame(with_gnn=True, folds=(1, 2, 3, 4, 5), seeds=(42, 43)):
    rows = []
    plain = {"esm3": 0.86, "prott5": 0.85, "esm1b": 0.85, "aac": 0.80, "ctd": 0.79,
             "onehot": 0.79, "onehot_only": 0.70, "mol_only": 0.68}
    ours = {"GNN[esm3]@a1": 0.88, "GNN[esm3]@a0": 0.87, "GNN[prott5]@a1": 0.875}
    table = dict(plain, **ours) if with_gnn else plain
    for name, base in table.items():
        for f in folds:
            for s in seeds:
                v = base + 0.01 * f + (0.004 if s == 43 else -0.004)
                rec = {"prot": name, "fold": f, "seed": s, "pdim": 128,
                       "dim": 512, "n_train": 1000,
                       **{m: v for m in METRICS}}
                if name.startswith("GNN["):
                    rec |= {"gnn_seed": 42, "t_graph": 12.3}
                rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- the reader

def test_the_unit_of_evidence_is_the_fold_not_the_seed():
    """Two boost seeds straddling the fold mean must not become two observations:
    n counts folds, and the +-0.004 pair cancels inside each one."""
    t = T.cell_table(frame(), "m2or", METRICS)
    r = t[t.prot == "esm3"].iloc[0]
    assert r["AUROC_n"] == 5
    assert r["AUROC"] == pytest.approx(0.86 + 0.01 * 3, abs=1e-9)   # folds 1..5


def test_our_rows_come_first_and_are_labelled_readably():
    t = T.cell_table(frame(), "m2or", METRICS)
    assert t.iloc[0]["family"] == "ours"
    labels = set(t[t.family == "ours"]["label"])
    assert "ours, ESM3 nodes" in labels
    assert "ours, identity nodes (alpha=0)" in labels
    assert "ours, ProtT5 nodes" in labels


def test_families_are_grouped_in_a_fixed_order():
    t = T.cell_table(frame(), "m2or", METRICS)
    seen = list(dict.fromkeys(t["family"]))
    assert seen == [f for f in T.FAMILY_ORDER if f in set(seen)]


def test_an_unknown_descriptor_is_shown_not_dropped():
    df = pd.concat([frame(), frame().assign(prot="some_new_descriptor")])
    t = T.cell_table(df, "m2or", METRICS)
    assert "some_new_descriptor" in set(t["prot"])
    assert t[t.prot == "some_new_descriptor"].iloc[0]["family"] == "other"


@pytest.mark.parametrize("metric,lower", [("AUROC", False), ("RMSE", True)])
def test_the_bolded_row_follows_the_metric_direction(metric, lower):
    """RMSE is won by being smaller; a table that bolded its maximum would
    advertise the worst row as the best."""
    df = frame()
    df[metric] = np.linspace(0.1, 0.9, len(df))
    t = T.cell_table(df, "m2or", [metric])
    i = T.best_of(t, metric)
    v = t[metric].to_numpy()
    assert v[i] == (v.min() if lower else v.max())


def test_a_table_without_our_rows_still_prints_and_says_what_to_run(tmp_path, capsys):
    d = tmp_path / "tables"
    d.mkdir()
    frame(with_gnn=False).to_csv(d / "prot_floor_m2or_transductive.csv", index=False)
    rc = T.main(["--dataset", "m2or", "--regime", "transductive",
                 "--root", str(d), "--out", str(tmp_path / "out")])
    assert rc == 0
    o = capsys.readouterr().out
    assert "no GNN rows" in o and "--gnn esm3@1 esm3@0 prott5@1" in o


def test_a_missing_cell_is_a_note_not_a_crash(tmp_path, capsys):
    rc = T.main(["--dataset", "m2or", "--regime", "transductive",
                 "--root", str(tmp_path), "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "nothing on disk" in capsys.readouterr().out


def test_latex_bolds_and_labels(tmp_path):
    d = tmp_path / "tables"
    d.mkdir()
    frame().to_csv(d / "prot_floor_m2or_inductive.csv", index=False)
    T.main(["--dataset", "m2or", "--regime", "inductive", "--root", str(d),
            "--out", str(tmp_path / "out")])
    tex = (tmp_path / "out" / "protein_sources.tex").read_text(encoding="utf-8")
    assert r"\label{tab:protsrcm2orindu}" in tex
    assert r"\textbf{" in tex
    assert "identity nodes" in tex


# ----------------------------------------------------------------- the compute half

def test_the_spec_parser_pins_source_and_dial(sweep):
    plms = {"esm3": "data/embeddings/proteins/esm3_m2or.npz",
            "prott5": "data/embeddings/proteins/prott5_m2or.npz"}
    name, path, alpha = sweep.parse_gnn_spec("esm3@1", plms)
    assert (name, alpha) == ("GNN[esm3]@a1", 1.0)
    assert path.endswith("esm3_m2or.npz")
    assert sweep.parse_gnn_spec("prott5@0.5", plms)[0] == "GNN[prott5]@a0.5"


@pytest.mark.parametrize("bad", ["esm3", "esm3@x", "esm3@2", "nosuch@1"])
def test_a_bad_spec_fails_loudly(sweep, bad):
    """A typo must not silently become a different row: the table's whole point is
    that the reader can tell which graph produced which line."""
    with pytest.raises(SystemExit):
        sweep.parse_gnn_spec(bad, {"esm3": "x.npz"})


def test_the_default_gnn_specs_are_the_three_the_table_asks_for(sweep):
    assert sweep.DEFAULT_GNN == ["esm3@1", "esm3@0", "prott5@1"]


def test_every_dataset_has_an_edge_variant_recorded(sweep):
    """The construction must match the one the paper reports, per dataset -- M2OR's
    hub core, the insects' complete matrix."""
    assert sweep.GNN_VARIANT["m2or"]["criterion"] == "greedy_pair_cover"
    assert sweep.GNN_VARIANT["m2or"]["q"] == 0.99
    for ds in ("cc", "hc"):
        assert sweep.GNN_VARIANT[ds]["criterion"] == "coverage"
        assert sweep.GNN_VARIANT[ds]["q"] == 0.0


# ----------------------------------------------------------------- resuming

def _args(**kw):
    import argparse
    base = dict(force=False, trust_existing=False, dataset="m2or",
                regime="transductive", mol=None, folds=[1, 2, 3, 4, 5],
                seeds=[42, 43], gnn_seeds=[42])
    return argparse.Namespace(**(base | kw))


def test_a_cell_is_keyed_by_graph_seed_too(sweep):
    """Two GNN rows differing only by the graph seed are two trainings; a descriptor
    row has no graph at all and must never collide with one."""
    base = {"prot": "esm3", "fold": 1, "seed": 42}
    assert sweep.cell_key(base) == sweep.cell_key(base | {"gnn_seed": float("nan")})
    g1 = sweep.cell_key({"prot": "GNN[esm3]@a1", "fold": 1, "seed": 42, "gnn_seed": 42})
    g2 = sweep.cell_key({"prot": "GNN[esm3]@a1", "fold": 1, "seed": 42, "gnn_seed": 43})
    assert g1 != g2 and g1 != sweep.cell_key(base)


def test_existing_rows_are_reused_when_provenance_is_there(sweep, tmp_path):
    out = tmp_path / "prot_floor_m2or_transductive.csv"
    frame(with_gnn=False).to_csv(out, index=False)
    out.with_suffix(".json").write_text(
        '{"dataset": "m2or", "regime": "transductive", "mol": null}', encoding="utf-8")
    rows, done = sweep.load_existing(out, _args())
    assert len(rows) == len(frame(with_gnn=False))
    assert ("esm3", 1, 42, -1) in done


def test_a_csv_without_provenance_is_refused_by_default(sweep, tmp_path):
    """It may predate the fix that stopped folding val into train, and rows fitted on
    more data than everyone else saw must not quietly join fresh ones."""
    out = tmp_path / "prot_floor_m2or_transductive.csv"
    frame(with_gnn=False).to_csv(out, index=False)
    with pytest.raises(SystemExit, match="sidecar"):
        sweep.load_existing(out, _args())
    rows, done = sweep.load_existing(out, _args(trust_existing=True))
    assert rows and done


def test_force_ignores_what_is_on_disk(sweep, tmp_path):
    out = tmp_path / "prot_floor_m2or_transductive.csv"
    frame().to_csv(out, index=False)
    out.with_suffix(".json").write_text('{"dataset": "m2or"}', encoding="utf-8")
    assert sweep.load_existing(out, _args(force=True)) == ([], set())


def test_a_sidecar_from_another_cell_is_an_error_not_a_merge(sweep, tmp_path):
    out = tmp_path / "prot_floor_m2or_transductive.csv"
    frame().to_csv(out, index=False)
    out.with_suffix(".json").write_text(
        '{"dataset": "cc", "regime": "transductive", "mol": null}', encoding="utf-8")
    with pytest.raises(SystemExit, match="different table"):
        sweep.load_existing(out, _args())


def test_a_csv_in_the_old_row_format_is_refused_with_a_reason(sweep, tmp_path):
    """Some prot_floor CSVs on disk predate the seed column. Resuming from one would
    crash inside the key function; saying so up front is the whole difference."""
    out = tmp_path / "prot_floor_m2or_transductive.csv"
    pd.DataFrame([{"prot": "esm1b", "AUROC": 0.9}]).to_csv(out, index=False)
    out.with_suffix(".json").write_text('{"dataset": "m2or"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="predates the current row format"):
        sweep.load_existing(out, _args())
