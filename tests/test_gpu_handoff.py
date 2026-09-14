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


# ------------------------------------------------------------------ the dead worker

teb = _module("scripts/modeling/train/train_ensemble_boost.py", "_train_ensemble_boost_gpu")


class _Proc:
    def __init__(self, alive, exitcode=None):
        self._alive, self.exitcode = alive, exitcode

    def is_alive(self):
        return self._alive


class _Queue:
    """Empty for the first `empties` reads, then hands over `item`."""

    def __init__(self, item=None, empties=0):
        self.item, self.left, self.reads = item, empties, 0

    def get(self, timeout=None):
        self.reads += 1
        if self.left > 0:
            self.left -= 1
            raise queue.Empty
        if self.item is None:
            raise queue.Empty
        return self.item


def test_a_worker_killed_in_a_cuda_teardown_raises_instead_of_hanging():
    q = _Queue(item=None)
    with pytest.raises(RuntimeError, match=r"repeat 3 died \(exit code -6\)"):
        teb._await_repeat(3, _Proc(alive=False, exitcode=-6), q, poll=0.01, grace=0.01)


def test_a_result_flushed_as_the_worker_exits_is_still_collected():
    """`put` then exit is a race, not a crash: the feeder thread may still be
    flushing when is_alive() first says False. Believing that would throw away a
    finished repeat."""
    q = _Queue(item=(4, {"combos": {}}), empties=1)
    assert teb._await_repeat(4, _Proc(alive=False, exitcode=0), q,
                             poll=0.01, grace=0.01) == (4, {"combos": {}})


def test_a_slow_but_living_worker_is_waited_on():
    q = _Queue(item=(2, "ok"), empties=3)
    assert teb._await_repeat(2, _Proc(alive=True), q, poll=0.001, grace=0.001) == (2, "ok")
    assert q.reads == 4, "it kept polling rather than declaring the worker dead"
