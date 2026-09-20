"""The run inventory: one line per run, and above all the right ENV on each line.

The env column is the point of the script — it answers "which runs did this broken
environment produce". Getting it from a recorded interpreter is easy; getting it
right for the runs written before recording existed is what these tests pin.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location("_rundates", _root / "scripts/analysis/run_dates.py")
RD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RD)


def _run(root, pool, name, sources, repeats=5, **extra):
    d = root / pool / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"sources": sources, **extra}), encoding="utf-8")
    rows = "\n".join(f"{r},cls,0.1" for r in range(1, repeats + 1))
    (d / "metrics.csv").write_text("repeat,combo,R2\n" + rows + "\n", encoding="utf-8")
    return d


@pytest.mark.parametrize("kind,env", [("molor", ".venv-molor"),
                                      ("lorax", ".venv-controls"),
                                      ("prosmith", ".venv-controls"),
                                      ("hladis", ".venv"),
                                      ("gnn_signed", ".venv")])
def test_an_old_run_gets_the_env_its_method_requires(kind, env):
    assert RD.env_of({"sources": [f"cls={kind}:p.npz", "prot=esm:p.npz"]}) == env


def test_a_recorded_interpreter_wins_over_the_inference():
    # If a run says which python wrote it, believe it -- inference is the fallback,
    # not a second opinion.
    config = {"sources": ["cls=molor:p.npz"], "python": "/abs/path/.venv/bin/python"}
    assert RD.env_of(config) == ".venv"


def test_a_run_with_no_pair_level_source_is_called_boost():
    config = {"sources": ["prot=esm:p.npz", "mol=gin:m.npz"]}
    assert RD.method_of(config) == "boost"
    assert RD.env_of(config) == "?"


def test_the_finish_time_is_read_from_metrics_not_from_config(tmp_path):
    # A run that started before an upgrade can finish after it, so start-up time
    # would put it in the wrong bucket.
    d = _run(tmp_path, "pool", "run", ["cls=hladis:p.npz"])
    (d / "config.json").touch()
    early = 1_000_000_000
    import os
    os.utime(d / "metrics.csv", (early, early))
    assert RD.when_of(d).year == 2001


def test_a_run_without_metrics_counts_zero_repeats(tmp_path):
    d = tmp_path / "pool" / "dead"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"sources": ["cls=lorax:p.npz"]}), encoding="utf-8")
    assert RD.n_repeats(d) == 0
    assert RD.when_of(d) is not None      # falls back to config.json's own mtime


def test_every_run_prints_exactly_one_line(tmp_path, capsys):
    _run(tmp_path, "p1", "a", ["cls=lorax:p.npz"])
    _run(tmp_path, "p1", "b", ["cls=molor:p.npz"], repeats=2)
    _run(tmp_path, "p2", "c", ["cls=hladis:p.npz"], python=".venv/bin/python", xgboost="2.1.4")
    RD.main(["--root", str(tmp_path)])
    body = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len([l for l in body if "/" in l and l[0].isdigit()]) == 3


def test_filters_narrow_to_one_environment(tmp_path, capsys):
    _run(tmp_path, "p", "lorax_run", ["cls=lorax:p.npz"])
    _run(tmp_path, "p", "hladis_run", ["cls=hladis:p.npz"])
    RD.main(["--root", str(tmp_path), "--env", ".venv-controls"])
    out = capsys.readouterr().out
    assert "lorax_run" in out and "hladis_run" not in out


def test_a_missing_root_is_not_an_error(tmp_path, capsys):
    RD.main(["--root", str(tmp_path / "nope")])
    assert "no runs matched" in capsys.readouterr().out
