"""Comparing two table runs cell by cell.

The one thing that must not silently go wrong: the sign. RMSE going down is an
improvement and AUROC going down is not, and a table that gets this backwards reads
as the opposite conclusion while looking perfectly plausible.
"""
import importlib.util
import pathlib
import sys

import pandas as pd
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location(
    "_cmp", _root / "scripts/legacy/04_compare_runs.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _long(rows):
    return pd.DataFrame([
        dict(dataset=d, regime=g, method=me, metric=mt, mean=v, std=0.01, rank=rk)
        for d, g, me, mt, v, rk in rows])


def _write(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir(parents=True)
    _long(rows).to_csv(d / "main_long.csv", index=False)
    return d


@pytest.mark.parametrize("metric,a,b,expected_sign", [
    ("AUROC", 0.80, 0.85, +1),      # up is better
    ("AUROC", 0.85, 0.80, -1),
    ("RMSE", 0.70, 0.60, +1),       # down is better
    ("RMSE", 0.60, 0.70, -1),
    ("R2", 0.40, 0.50, +1),
])
def test_the_sign_follows_the_metrics_own_direction(metric, a, b, expected_sign):
    row = pd.Series({"metric": metric, "mean_a": a, "mean_b": b})
    assert (C.improvement(row) > 0) == (expected_sign > 0)


def test_matched_cells_are_joined_on_all_four_keys(tmp_path, capsys):
    rows_a = [("m2or", "transductive", "Our graph", "AUROC", 0.882, 2.0),
              ("m2or", "cold_molecule", "Our graph", "AUROC", 0.840, 1.0)]
    rows_b = [("m2or", "transductive", "Our graph", "AUROC", 0.863, 3.0),
              ("m2or", "cold_molecule", "Our graph", "AUROC", 0.820, 2.0)]
    A, B = _write(tmp_path, "a", rows_a), _write(tmp_path, "b", rows_b)
    C.main(["--a", str(A), "--b", str(B), "--a-label", "1b", "--b-label", "3"])
    out = capsys.readouterr().out
    assert "matched: 2" in out
    # the same method in two regimes must NOT collapse into one row
    assert out.count("Our graph") >= 2
    assert "worse" in out


def test_a_cell_present_on_only_one_side_is_reported_not_dropped_silently(tmp_path, capsys):
    A = _write(tmp_path, "a", [("m2or", "transductive", "LORAX", "AUROC", 0.9, 1.0),
                               ("m2or", "transductive", "Ghost", "AUROC", 0.5, 2.0)])
    B = _write(tmp_path, "b", [("m2or", "transductive", "LORAX", "AUROC", 0.9, 1.0)])
    C.main(["--a", str(A), "--b", str(B)])
    out = capsys.readouterr().out
    assert "1 row(s)" in out and "no counterpart" in out


def test_place_changes_are_printed_even_when_the_value_barely_moves(tmp_path, capsys):
    A = _write(tmp_path, "a", [("cc", "transductive", "Base", "R2", 0.6610, 2.20)])
    B = _write(tmp_path, "b", [("cc", "transductive", "Base", "R2", 0.6560, 1.00)])
    C.main(["--a", str(A), "--b", str(B)])
    out = capsys.readouterr().out
    assert "2.20 -> 1.00" in out


def test_min_delta_hides_the_cells_that_did_not_move(tmp_path, capsys):
    A = _write(tmp_path, "a", [("m2or", "transductive", "Still", "AUROC", 0.8940, 1.0),
                               ("m2or", "transductive", "Moved", "AUROC", 0.8820, 2.0)])
    B = _write(tmp_path, "b", [("m2or", "transductive", "Still", "AUROC", 0.8960, 1.0),
                               ("m2or", "transductive", "Moved", "AUROC", 0.8630, 2.0)])
    C.main(["--a", str(A), "--b", str(B), "--min-delta", "0.01"])
    out = capsys.readouterr().out
    assert "Moved" in out and "Still" not in out.split("=== ")[1]


def test_an_old_csv_without_the_panel_columns_is_refused(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    pd.DataFrame([dict(method="x", metric="AUROC", mean=0.5, std=0.0, rank=1.0)]).to_csv(
        d / "main_long.csv", index=False)
    with pytest.raises(SystemExit) as e:
        C.load(d)
    assert "regenerate" in str(e.value)
