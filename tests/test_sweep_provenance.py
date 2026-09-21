"""Reading a sweep root's flags back out of its CSVs.

Two roots go into the same sentence only if they were produced the same way. The
script exists because shell history is not evidence and `seeded_graph` is exactly
the kind of flag whose absence looks like `False`.
"""
import importlib.util
import pathlib
import sys

import pandas as pd

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location(
    "_prov", _root / "scripts/analysis/sweep_provenance.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def _cell(root, name, **cols):
    root.mkdir(parents=True, exist_ok=True)
    base = dict(alpha=[1.0, 1.0], seed=[42, 43], fold=[1, 1], seeded_graph=[True, True])
    base.update(cols)
    pd.DataFrame(base).to_csv(root / name, index=False)


def test_the_recorded_flags_are_reported(tmp_path):
    _cell(tmp_path, "metrics_m2or_transductive.csv")
    s = P.summarise(tmp_path)
    assert s["cells"] == 1 and s["rows"] == 2
    assert s["alpha"] == [1.0] and s["seed"] == [42, 43] and s["seeded_graph"] == [True]


def test_a_column_the_series_never_had_is_omitted_not_invented(tmp_path):
    # An older series has no seeded_graph at all. Printing False would assert
    # something the file does not say.
    _cell(tmp_path, "metrics_old.csv")
    pd.read_csv(tmp_path / "metrics_old.csv").drop(columns=["seeded_graph"]).to_csv(
        tmp_path / "metrics_old.csv", index=False)
    assert "seeded_graph" not in P.summarise(tmp_path)


def test_an_empty_root_says_so_rather_than_crashing(tmp_path):
    assert P.summarise(tmp_path)["cells"] == 0


def test_two_roots_are_diffed_field_by_field(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _cell(a, "metrics_m2or.csv")
    _cell(b, "metrics_m2or.csv", seeded_graph=[False, False])
    P.main(["--root", str(a), "--root", str(b)])
    out = capsys.readouterr().out
    assert "seeded_graph" in out.split("vs")[-1]
    assert "[True]  !=  [False]" in out


def test_identical_roots_are_called_comparable(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _cell(a, "metrics_m2or.csv")
    _cell(b, "metrics_m2or.csv")
    P.main(["--root", str(a), "--root", str(b)])
    assert "comparable" in capsys.readouterr().out


def test_a_long_value_list_is_abbreviated(tmp_path):
    _cell(tmp_path, "metrics_grid.csv",
          alpha=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
          seed=[42] * 11, fold=[1] * 11, seeded_graph=[True] * 11)
    assert P.summarise(tmp_path)["alpha"] == [0.0, "...", 1.0]
