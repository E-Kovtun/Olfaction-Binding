"""Rebuilding a run's command line from the config.json it wrote at start-up.

The property that matters is a round trip: whatever the relauncher prints must
parse back into the same arguments the dead run had. A command that merely *looks*
right is how you rerun a fold with a silently different combo order -- the trap
that has already produced one wrong comparison here.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _root / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("_relaunch", "scripts/modeling/train/relaunch_incomplete.py")

CONFIG = {
    "regime": "ofm", "task": "regression",
    "sources": ["cls=molor:p.npz:m.npz", "prot=esm:p.npz", "mol=gin:m.npz"],
    "combos": "1 12 123", "weight_method": "both", "on_missing": "drop",
    "tune_boost": False, "n_trials": 30, "out_dir": "results/x/hc-rand",
    "run_name": "hc_rand_molor", "skip_checkpoints": False,
    "max_parallel": 2, "gpus": [2, 3],
    "pairs": "data/processed/pairs_curated.csv", "split": "stratified",
    "seeds": [42], "test_size": 0.2, "val_size": 0.2,
    "full_full_mode": "transductive", "repeats": [1, 2, 3, 4, 5],
    "dataset": "hc", "split_family": "rand",
    "python": ".venv-molor/bin/python", "xgboost": "3.2.0",
}


def _write_run(root, pool, run, config, done_repeats=()):
    d = root / pool / run
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if done_repeats:
        rows = "\n".join(f"{r},cls,0.1" for r in done_repeats)
        (d / "metrics.csv").write_text("repeat,combo,R2\n" + rows + "\n", encoding="utf-8")
    return d


def test_the_rebuilt_command_parses_back_into_the_same_arguments():
    parser = R._teb.build_parser()
    argv = R.rebuild(CONFIG, parser, "py", "train.py")
    reparsed = vars(parser.parse_args(argv[2:]))
    for key, value in CONFIG.items():
        if key in R.NON_CLI_KEYS:
            continue
        assert reparsed[key] == value, key


def test_values_left_at_their_default_are_not_spelled_out():
    parser = R._teb.build_parser()
    argv = R.rebuild(CONFIG, parser, "py", "train.py")
    # split_family="rand" and n_trials=30 are the parser's own defaults; printing
    # them would bury the three flags that actually distinguish this run.
    assert "--split-family" not in argv
    assert "--n-trials" not in argv
    # ...and a repeatable flag stays repeatable rather than collapsing to one.
    assert argv.count("--source") == 3


def test_provenance_keys_are_read_but_never_re_emitted():
    parser = R._teb.build_parser()
    argv = R.rebuild(CONFIG, parser, "py", "train.py")
    assert "--python" not in argv and "--xgboost" not in argv
    assert "3.2.0" not in argv


def test_a_run_with_every_repeat_is_not_relaunched(tmp_path, capsys):
    _write_run(tmp_path, "pool", "full", CONFIG, done_repeats=(1, 2, 3, 4, 5))
    R.main(["--root", str(tmp_path)])
    assert "nothing to relaunch" in capsys.readouterr().out


@pytest.mark.parametrize("done", [(), (1, 2)])
def test_a_run_missing_repeats_is_relaunched_with_its_own_interpreter(tmp_path, capsys, done):
    _write_run(tmp_path, "pool", "partial", CONFIG, done_repeats=done)
    R.main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert f"({len(done)}/5 repeats present)" in out
    assert ".venv-molor/bin/python" in out


def test_the_interpreter_can_be_overridden_for_a_whole_block(tmp_path, capsys):
    # The point of the fix: the env that recorded itself is the broken one.
    _write_run(tmp_path, "pool", "partial", CONFIG)
    R.main(["--root", str(tmp_path), "--python", ".venv/bin/python"])
    out = capsys.readouterr().out
    assert ".venv/bin/python scripts/" in out
    assert ".venv-molor" not in out


def test_an_empty_metrics_file_counts_as_nothing_done(tmp_path):
    d = _write_run(tmp_path, "pool", "empty", CONFIG)
    (d / "metrics.csv").write_text("", encoding="utf-8")
    assert R.done_repeats(d) == set()


def test_an_old_config_without_a_recorded_interpreter_gets_the_env_its_source_needs(tmp_path, capsys):
    old = {k: v for k, v in CONFIG.items() if k not in R.NON_CLI_KEYS}
    _write_run(tmp_path, "pool", "old_molor", old)
    R.main(["--root", str(tmp_path)])
    assert ".venv-molor/bin/python" in capsys.readouterr().out


@pytest.mark.parametrize("kind,env", [("hladis", ".venv/bin/python"),
                                      ("lorax", ".venv-controls/bin/python"),
                                      ("prosmith", ".venv-controls/bin/python"),
                                      ("molor", ".venv-molor/bin/python")])
def test_each_pair_level_source_maps_to_its_own_env(kind, env):
    assert R.infer_python({"sources": [f"cls={kind}:p.npz", "prot=esm:p.npz"]}) == env


def test_an_entity_only_run_names_no_env_and_falls_back(tmp_path, capsys):
    # prot+mol has no pair-level source, so any env with xgboost will do.
    assert R.infer_python({"sources": ["prot=esm:p.npz", "mol=gin:m.npz"]}) is None
