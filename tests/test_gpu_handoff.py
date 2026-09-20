"""The two guards around sharing one GPU between torch and XGBoost.

Both pin behaviour that only ever showed up on the server, and both fail SILENTLY
when broken -- which is why they are worth a test rather than a comment:

  * the extractors must all be done before the first boosting head starts, so torch's
    caching allocator has already handed the card back. When they interleave, the
    boost head quietly falls back to CPU (different `hist` splits, so numbers drift
    between repeats for no reason a seed explains) and the process can die outright
    in a CUDA destructor.
  * a worker that dies that way never answers its queue. The parent used to block on
    it forever while metrics.csv stayed short -- and a short metrics.csv is read
    downstream as a perfectly legitimate run with fewer folds.
"""
import importlib.util
import pathlib
import queue
import sys

import numpy as np
import pandas as pd
import pytest

import orbind.ensemble as ens

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------- extract, then boost

class _Ext:
    """Minimal extractor: records when it was asked for features."""

    def __init__(self, name, log, dim=2):
        self.name, self.log, self.dim = name, log, dim
        # run_ensemble refuses a self-training source that doesn't declare which task
        # its head optimises, rather than silently fitting BCE against a continuous
        # target. Declaring it is part of being a plausible extractor.
        self.task = "classification"

    def covered(self, pairs, idx):
        return np.ones(len(idx), dtype=bool)

    def fit_transform(self, pairs, tr, va, te, seed, checkpoint_dir=None):
        self.log.append(("extract", self.name))
        rng = np.random.default_rng(len(self.name) + ord(self.name[0]))
        return tuple(rng.normal(size=(len(i), self.dim)).astype(np.float32)
                     for i in (tr, va, te))


def test_every_extractor_finishes_before_the_first_boosting_head(monkeypatch):
    """The ordering IS the fix: lazily pulling features inside the combo loop puts a
    live torch allocator and XGBoost's device vectors on the same card at the same
    moment, which is what killed a cc LORAX run."""
    log = []
    monkeypatch.setattr(ens, "_release_gpu", lambda: log.append(("release", None)))
    monkeypatch.setattr(ens, "fit_boost",
                        lambda Xtr, ytr, seed=42, task="classification":
                        log.append(("boost", Xtr.shape[1])) or "clf")
    monkeypatch.setattr(ens, "predict_scores",
                        lambda model, X, task="classification": X.sum(axis=1).astype(float))

    n = 30
    pairs = pd.DataFrame({"receptor": [f"r{i % 5}" for i in range(n)],
                          "inchikey": [f"m{i}" for i in range(n)],
                          "label": np.linspace(-1.0, 1.0, n)})
    idx = np.arange(n)
    extractors = {"a": _Ext("a", log), "b": _Ext("b", log)}

    ens.run_ensemble(pairs, extractors, "1 2 12",
                     train_idx=idx[:18], val_idx=idx[18:24], test_idx=idx[24:],
                     weight_method="simplex", task="regression")

    kinds = [k for k, _ in log]
    assert kinds.count("extract") == 2, "each extractor is still built exactly once"
    assert kinds.count("boost") == 3, "one head per combo"
    last_extract = max(i for i, (k, _) in enumerate(log) if k == "extract")
    first_boost = min(i for i, (k, _) in enumerate(log) if k == "boost")
    release = kinds.index("release")
    assert last_extract < release < first_boost, (
        f"torch and XGBoost overlap on the GPU: {log}")


def test_the_shared_features_are_still_built_once_each(monkeypatch):
    """Priming the cache must not become a second, separate extraction pass: combo
    'a+b' has to reuse what combos 'a' and 'b' already computed."""
    log = []
    monkeypatch.setattr(ens, "_release_gpu", lambda: None)
    monkeypatch.setattr(ens, "fit_boost",
                        lambda Xtr, ytr, seed=42, task="classification": "clf")
    monkeypatch.setattr(ens, "predict_scores",
                        lambda model, X, task="classification": np.zeros(len(X)))

    n = 24
    pairs = pd.DataFrame({"receptor": [f"r{i % 4}" for i in range(n)],
                          "inchikey": [f"m{i}" for i in range(n)],
                          "label": np.linspace(0.0, 1.0, n)})
    idx = np.arange(n)
    ens.run_ensemble(pairs, {"a": _Ext("a", log), "b": _Ext("b", log)}, "1 12 2",
                     train_idx=idx[:14], val_idx=idx[14:19], test_idx=idx[19:],
                     weight_method="simplex", task="regression")
    assert [name for k, name in log if k == "extract"] == ["a", "b"]


def test_release_gpu_is_a_noop_without_cuda():
    """It runs on every box, including the laptop this suite is developed on."""
    ens._release_gpu()


# ------------------------------------------------- the repeat pool and dead workers

teb = _module("scripts/modeling/train/train_ensemble_boost.py", "_train_ensemble_boost_gpu")


class _Worker:
    """One fake repeat, playing both the process and its queue.

    It answers after `ticks` polls. `dies` exits without answering (SIGABRT in a CUDA
    teardown); `late` reports itself dead BEFORE its result is readable -- the
    feeder-thread race -- and only a grace read then finds the result."""

    def __init__(self, repeat, ticks, dies=False, late=False):
        self.repeat, self.ticks, self.dies, self.late = repeat, ticks, dies, late
        self.taken, self.exitcode = False, None

    def is_alive(self):
        if self.dies or self.late:
            return self.ticks > 0
        return not self.taken

    def join(self):
        self.exitcode = -6 if self.dies else 0

    def get_nowait(self):
        self.ticks -= 1
        if self.dies or self.late or self.ticks > 0:
            raise queue.Empty
        self.taken = True
        return self.repeat, {"repeat": self.repeat}

    def get(self, timeout=None):
        if self.dies:
            raise queue.Empty
        self.taken = True
        return self.repeat, {"repeat": self.repeat}


def _pool(ticks, n_slots=2, gpus=(0, 1), dies=(), late=()):
    """Run the real scheduler over fake workers; return what it did."""
    events, collected, running = [], [], {}

    def launch(repeat, gpu):
        w = _Worker(repeat, ticks[repeat], dies=repeat in dies, late=repeat in late)
        running[repeat] = gpu
        events.append(("start", repeat, gpu, dict(running)))
        return w, w

    def collect(repeat, result):
        running.pop(repeat)
        collected.append(repeat)

    failed = teb._run_repeat_pool(list(ticks), n_slots, list(gpus), launch, collect,
                                  poll=0, grace=0, log=lambda *a, **k: None)
    for r, _ in failed:
        running.pop(r, None)
    return events, collected, failed


def test_a_freed_slot_is_refilled_before_the_slow_repeat_finishes():
    """The whole point: repeat 1 trains for hours, repeat 2 loads a checkpoint. Under
    the chunked loop repeat 3 waited for repeat 1; here it takes repeat 2's card."""
    events, collected, failed = _pool({1: 50, 2: 2, 3: 2})
    assert failed == []
    assert collected == [2, 3, 1]
    starts = [(r, g) for _, r, g, _ in events]
    assert starts == [(1, 0), (2, 1), (3, 1)], "the refill lands on the card just freed"
    assert 1 in events[-1][3], "repeat 3 started while repeat 1 was still running"


def test_never_more_workers_than_slots_and_never_two_on_one_card():
    events, collected, failed = _pool({1: 9, 2: 3, 3: 7, 4: 1, 5: 4, 6: 2, 7: 5})
    assert sorted(collected) == [1, 2, 3, 4, 5, 6, 7] and failed == []
    for _, _, _, running in events:
        assert len(running) <= 2
        assert len(set(running.values())) == len(running), f"two workers on one GPU: {running}"


def test_a_dead_worker_is_reported_and_the_others_are_still_written():
    """A CUDA teardown abort in one repeat used to raise at once and throw away the
    hours of training still running next to it."""
    events, collected, failed = _pool({1: 6, 2: 2, 3: 3}, dies={2})
    assert failed == [(2, -6)]
    assert sorted(collected) == [1, 3]


def test_a_result_flushed_as_the_worker_exits_is_still_collected():
    """`put` then exit is a race, not a crash: the feeder thread may still be
    flushing when is_alive() first says False. Believing that would throw away a
    finished repeat."""
    events, collected, failed = _pool({1: 2, 2: 3}, late={1})
    assert failed == [] and sorted(collected) == [1, 2]


def test_without_gpus_the_slots_still_bound_concurrency():
    events, collected, failed = _pool({1: 3, 2: 1, 3: 2}, n_slots=2, gpus=())
    assert sorted(collected) == [1, 2, 3]
    assert all(g is None for _, _, g, _ in events)
    assert max(len(r) for *_, r in events) <= 2


# --------------------------------------------------------------------------- #
# The third guard: which xgboost is installed.
#
# A whole ESM3 baseline block died on 2026-09-20 because `.venv-controls` and
# `.venv-molor` had xgboost 3.x while `.venv` had 2.1.4. 3.x allocates device
# vectors through CUDA virtual memory and aborts on two of the server's cards --
# and the abort comes out of a C++ destructor, so no `except` sees it. The
# version check exists to turn three lost hours into one line at start-up.
# --------------------------------------------------------------------------- #
from orbind.baselines import check_xgboost_version


def test_the_matching_major_is_accepted_and_the_version_returned():
    # Parametrised by what is installed rather than by the pin, so the test says
    # the same thing in an env that has not been fixed yet.
    import xgboost as xgb
    assert check_xgboost_version(major=int(xgb.__version__.split(".")[0])) == xgb.__version__


def test_a_different_major_is_refused_before_anything_runs():
    import pytest
    with pytest.raises(RuntimeError) as e:
        check_xgboost_version(major=99)
    msg = str(e.value)
    # The message has to be actionable on a server at 3am: what is wrong, and the
    # exact command that fixes it.
    assert "xgboost" in msg and "uv pip install" in msg
