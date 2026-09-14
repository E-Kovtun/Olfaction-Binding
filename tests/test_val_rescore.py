"""The val rescore's backfill of per-row predictions.

The metrics it writes are exercised by the grid's own readers; what is pinned here is the
part that TOUCHES A FILE THE RECORDED NUMBERS REST ON. A backfill that replaced a key
instead of adding one would detach every row already on disk from the cloud it was
computed from, and nothing would raise.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


vr = _module("scripts/analysis/val_rescore.py", "_val_rescore_under_test")


def _V(n=3):
    return {"y": np.arange(n, dtype=np.float32),
            "rec": np.array([f"r{i}" for i in range(n)], dtype=object),
            "mol": np.array([f"m{i}" for i in range(n)], dtype=object)}


def test_a_backfill_adds_keys_and_keeps_every_one_already_there(tmp_path):
    f = tmp_path / "cell.npz"
    np.savez_compressed(f, z_prot=np.zeros((2, 2), np.float32),
                        pred=np.array([1.0, 2.0], np.float32))
    vr._merge_npz(f, pred=np.array([9.9], np.float32),
                  pred_va=np.array([0.3, 0.4], np.float32))
    with np.load(f) as z:
        assert set(z.files) == {"z_prot", "pred", "pred_va"}
        assert z["pred"].tolist() == [1.0, 2.0], "the recorded prediction was replaced"
        assert z["pred_va"].tolist() == pytest.approx([0.3, 0.4])
    assert not list(f.parent.glob("*.tmp.npz"))


@pytest.mark.parametrize("arm, alpha, combo, key, name", [
    ("gate", 1.0, "cls+mol", "pred_va", "gate_a1_f2_s43.npz"),
    ("gate", 1.0, "cls+prot+mol", "pred_va_prot", "gate_a1_f2_s43.npz"),
    ("gate", 1.0, None, "pred_va", "gate_a1_f2_s43.npz"),        # written before `combo`
    ("boost_full", float("nan"), "prot+mol", "pred_va", "boost_full_aNone_f2_s43.npz"),
])
def test_each_head_writes_under_its_own_key(tmp_path, arm, alpha, combo, key, name):
    """Two heads share one npz. Under one key they would overwrite each other and the
    table would read one head's threshold against the other head's scores."""
    row = {"arm": arm, "alpha": alpha, "fold": 2, "seed": 43, "combo": combo}
    V = _V()
    out = vr.dump_val(tmp_path, row, np.array([0.1, 0.2, 0.3], np.float32), V)
    assert out.name == name
    with np.load(out) as z:
        assert key in z.files
        assert z[key].tolist() == pytest.approx([0.1, 0.2, 0.3])
        assert z["y_val"].tolist() == [0.0, 1.0, 2.0]
        assert z["receptor_val"].tolist() == ["r0", "r1", "r2"]


def test_the_constant_arm_gets_no_operating_point(tmp_path):
    row = {"arm": "naive", "alpha": float("nan"), "fold": 1, "seed": 42, "combo": "const"}
    assert vr.dump_val(tmp_path, row, np.zeros(3, np.float32), _V()) is None
    assert not list(tmp_path.glob("*.npz"))
