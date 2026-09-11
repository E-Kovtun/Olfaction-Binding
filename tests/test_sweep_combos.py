"""Two boosting heads on one trained graph, and the backfill that adds the second head to
a run that only ever fitted the first.

The expensive part of a cell is the graph; a head is seconds. So the sweep fits
`cls+mol` = [z_prot || molecule] and `cls+prot+mol` = [z_prot || raw ESM || molecule] on
the same receptor cloud. A directory finished before that existed has `cls+mol` rows and
a dumped cloud per cell, and the missing head must be added WITHOUT retraining when the
dump can be trusted -- and by retraining when it cannot.

The graph itself is never trained here: `_train_graph` is replaced by a stub that hands
back a fixed cloud, so every assertion is about the bookkeeping around it. The boosting
head is real XGBoost, because "the dump reproduces the recorded number" is a claim about
that head and a fake one would make the test pass by construction.
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
    """Import a script by path, stubbing torch_geometric where it is not installed --
    nothing under test touches it, but the extractor module imports it at load."""
    try:
        import torch_geometric                        # noqa: F401
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
def sw():
    return _load("scripts/modeling/train/run_alpha_gate_sweep.py", "combo_sweep")


def A(**kw):
    return argparse.Namespace(**(dict(seeds=[42], alphas=[1.0], legacy=False, gate=True,
                                      baselines_only=False) | kw))


def _gargs(out, **kw):
    """What `_graph_rows` reads off the parsed args, after `sweep` has set the private
    per-file fields."""
    return argparse.Namespace(**(dict(
        out=str(out), variant=None, mol_source="chemberta", nodes="esm", dial="nodes",
        mix_seed=0, seed_graph=True, score_train=True, n_perm=0, dump_embeddings=True,
        dump_predictions=True, reuse_dumps=True, combos=None,
        _variant="q0cov", _regime="transductive", _ds="hc") | kw))


# ---------------------------------------------------------------------- identity

def test_a_row_without_a_combo_is_the_head_the_sweep_always_fitted(sw):
    """Every row on disk today predates the column. Read any other way, a resume would
    either recompute every finished cell or skip the head it was asked to add."""
    assert sw.key("gate", 1.0, 1, 42) == sw.key("gate", 1.0, 1, 42, "cls+mol")
    assert sw.key("gate", 1.0, 1, 42, np.nan) == sw.key("gate", 1.0, 1, 42, "cls+mol")
    assert sw.key("gate", 1.0, 1, 42, "cls+prot+mol") != sw.key("gate", 1.0, 1, 42)
    assert sw.key("boost_full", None, 1, 42)[4] == "prot+mol"


def test_the_second_head_sees_raw_esm_between_graph_and_molecule(sw):
    P = {"Xp_tr": np.full((3, 5), 2.0, np.float32), "Xm_tr": np.full((3, 4), 3.0, np.float32)}
    Z = np.ones((3, 2), np.float32)
    a = sw._head_features("cls+mol", Z, P, "tr")
    b = sw._head_features("cls+prot+mol", Z, P, "tr")
    assert a.shape == (3, 6) and b.shape == (3, 11)
    assert (b[:, :2] == 1).all() and (b[:, 2:7] == 2).all() and (b[:, 7:] == 3).all()
    with pytest.raises(ValueError):
        sw._head_features("prot+mol", Z, P, "tr")      # boost's own; never a graph head


# ---------------------------------------------------------------------- planning

def test_a_fresh_cell_trains_its_graph_once_for_both_heads(sw):
    jobs = sw.plan([1], A(), set())
    graph = [j for j in jobs if j[0] == "gate"]
    assert len(graph) == 1, "two heads must not mean two graph trainings"
    assert graph[0][4] == {"combos": ("cls+mol", "cls+prot+mol"), "fill": False}
    assert sw.n_heads(jobs) == 2


def test_a_finished_cls_mol_run_is_backfilled_not_recomputed(sw, tmp_path):
    rows = [dict(arm="boost_full", alpha=np.nan, fold=1, seed=42, split="test",
                 status="ok", R2=0.5),
            dict(arm="naive", alpha=np.nan, fold=1, seed=42, split="test",
                 status="ok", R2=0.0),
            dict(arm="gate", alpha=1.0, fold=1, seed=42, split="val", status="ok",
                 R2=0.55),
            dict(arm="gate", alpha=1.0, fold=1, seed=42, split="test", status="ok",
                 R2=0.61, rsa_esm=0.2, rsa_esm_z=3.0, k_pca=0)]
    rec = tmp_path / "records_hc_rand_chemberta_nodedial.csv"
    pd.DataFrame(rows).to_csv(rec, index=False)
    prev, done = sw.load_done(rec, tmp_path / "metrics_hc_rand_chemberta_nodedial.csv",
                              argparse.Namespace(force=False, seed_graph=True))
    jobs = sw.plan([1], A(), done, sw.known_cells(prev))
    assert [j[:4] for j in jobs] == [("gate", 1.0, 1, 42)]
    spec = jobs[0][4]
    assert spec["combos"] == ("cls+prot+mol",) and spec["fill"] is True
    assert spec["expect"] == {"cls+mol": {"R2": pytest.approx(0.61)}}
    assert spec["inherit"]["rsa_esm"] == pytest.approx(0.2)
    assert "R2" not in spec["inherit"], "a head's score is not a property of the cell"
    # a run that only ever wanted cls+mol has nothing left to do here
    assert sw.plan([1], A(combos=["cls+mol"]), done, sw.known_cells(prev)) == []


def test_a_failed_job_owes_a_row_per_head(sw):
    rows = sw._failed(("gate", 1.0, 1, 42, {"combos": ("cls+mol", "cls+prot+mol")}),
                      RuntimeError("oom"))
    assert [r["combo"] for r in rows] == ["cls+mol", "cls+prot+mol"]
    assert all(r["status"].startswith("failed") for r in rows)


# ---------------------------------------------------------------------- the dump

def test_a_backfill_adds_to_a_dump_and_never_replaces_what_is_in_it(sw, tmp_path):
    args = _gargs(tmp_path)
    sw._dump(args, "hc", "transductive", "gate", 1.0, 1, 42,
             z_prot=np.zeros((2, 3), np.float32), pred=np.array([1.0, 2.0], np.float32))
    sw._dump(args, "hc", "transductive", "gate", 1.0, 1, 42, merge=True,
             z_prot=np.ones((2, 3), np.float32), pred_prot=np.array([5.0], np.float32))
    f = sw.dump_file(args, "hc", "transductive", "gate", 1.0, 1, 42)
    with np.load(f) as z:
        assert set(z.files) == {"z_prot", "pred", "pred_prot"}
        assert (z["z_prot"] == 0).all(), "the recorded rows' cloud was overwritten"
    assert not list(f.parent.glob("*.tmp.npz"))


def test_the_dump_is_rebuilt_per_pair_and_refuses_a_stranger(sw, tmp_path):
    f = tmp_path / "cell.npz"
    np.savez_compressed(f, z_prot=np.arange(6, dtype=np.float32).reshape(3, 2),
                        receptors=np.array(["a", "b", "c"]))
    P = {"rec_tr": np.array(["c", "a", "a"]), "rec_va": np.array(["b"]),
         "rec_te": np.array([], dtype=object)}
    Zs, _ = sw._z_from_dump(f, P)
    assert Zs["tr"].tolist() == [[4, 5], [0, 1], [0, 1]]
    assert Zs["va"].tolist() == [[2, 3]] and Zs["te"].shape == (0, 2)
    Zs, why = sw._z_from_dump(f, P | {"rec_va": np.array(["zzz"])})
    assert Zs is None and "not in it" in why
    assert sw._z_from_dump(tmp_path / "absent.npz", P)[0] is None


# ---------------------------------------------------------------------- end to end

@pytest.fixture
def fold():
    """A small complete regression fold: 4 receptors x 15 molecules, a fixed 'graph'
    cloud the labels depend on, raw ESM and molecule blocks beside it."""
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    recs = [f"r{i}" for i in range(4)]
    mols = [f"m{j}" for j in range(15)]
    prot = {r: rng.normal(size=6).astype(np.float32) for r in recs}
    mol = {m: rng.normal(size=5).astype(np.float32) for m in mols}
    cloud = {r: rng.normal(size=3).astype(np.float32) for r in recs}
    pairs = [(r, m) for r in recs for m in mols]
    y = np.array([cloud[r].sum() + mol[m][0] + rng.normal(0, 0.1) for r, m in pairs],
                 np.float32)
    idx = rng.permutation(len(pairs))
    P, Zs = {"order": recs}, {}
    for k, ix in (("tr", idx[:36]), ("va", idx[36:48]), ("te", idx[48:])):
        P[k] = ix
        P[f"rec_{k}"] = np.array([pairs[i][0] for i in ix])
        P[f"mol_{k}"] = np.array([pairs[i][1] for i in ix])
        P[f"Xp_{k}"] = np.stack([prot[pairs[i][0]] for i in ix])
        P[f"Xm_{k}"] = np.stack([mol[pairs[i][1]] for i in ix])
        P[f"y_{k}"] = y[ix]
        Zs[k] = np.stack([cloud[pairs[i][0]] for i in ix])
    return P, Zs


@pytest.fixture
def stub_graph(sw, monkeypatch):
    """`_train_graph` replaced by a counter that hands back the fold's fixed cloud, and
    the geometry nulls by a constant -- neither is what these tests are about."""
    calls = []

    def install(Zs):
        def fake(*a, **k):
            calls.append(a[:4])
            return {s: v.copy() for s, v in Zs.items()}, None, 1.5
        monkeypatch.setattr(sw, "_train_graph", fake)
        monkeypatch.setattr(sw, "_geometry", lambda Z, P, n: {"rsa_esm": 0.123})
        return calls
    return install


def _original_run(sw, P, args):
    """The run on disk today: the graph trained, only cls+mol fitted, the cloud dumped."""
    rows = sw._graph_rows("gate", 1.0, 1, 42, "hc", P, args,
                          {"combos": ("cls+mol",), "fill": False})
    return next(r for r in rows if r["split"] == "test")


def test_a_fresh_job_fits_both_heads_on_one_training(sw, tmp_path, fold, stub_graph):
    P, Zs = fold
    calls = stub_graph(Zs)
    args = _gargs(tmp_path)
    rows = sw._graph_rows("gate", 1.0, 1, 42, "hc", P, args,
                          {"combos": ("cls+mol", "cls+prot+mol"), "fill": False})
    assert len(calls) == 1
    assert sorted((r["combo"], r["split"]) for r in rows) == sorted(
        (c, s) for c in ("cls+mol", "cls+prot+mol") for s in ("train", "val", "test"))
    assert {r["z_source"] for r in rows} == {"trained"}
    t = {r["combo"]: r for r in rows if r["split"] == "test"}
    assert t["cls+mol"]["R2"] != t["cls+prot+mol"]["R2"], "the two heads are one model"
    with np.load(sw.dump_file(args, "hc", "transductive", "gate", 1.0, 1, 42)) as z:
        assert {"z_prot", "receptors", "pred", "pred_prot"} <= set(z.files)


def test_backfill_fits_the_missing_head_on_a_verified_dump_without_training(
        sw, tmp_path, fold, stub_graph):
    P, Zs = fold
    calls = stub_graph(Zs)
    args = _gargs(tmp_path)
    recorded = _original_run(sw, P, args)
    assert len(calls) == 1

    spec = {"combos": ("cls+prot+mol",), "fill": True,
            "expect": {"cls+mol": {"R2": recorded["R2"]}},
            "inherit": {"rsa_esm": 0.456, "k_pca": 0}}
    rows = sw._graph_rows("gate", 1.0, 1, 42, "hc", P, args, spec)
    assert len(calls) == 1, "a dump that reproduces the recorded head must be reused"
    assert {r["combo"] for r in rows} == {"cls+prot+mol"}
    assert {r["z_source"] for r in rows} == {"dump"}
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert all(r["rsa_esm"] == 0.456 and r["t_graph"] == 0.0 for r in rows), \
        "geometry comes from the recorded cell, and no graph time was spent"
    with np.load(sw.dump_file(args, "hc", "transductive", "gate", 1.0, 1, 42)) as z:
        assert {"z_prot", "pred", "pred_prot"} <= set(z.files)


@pytest.mark.parametrize("break_it", ["wrong number", "no dump", "reuse off"])
def test_backfill_retrains_when_the_dump_cannot_be_trusted(sw, tmp_path, fold,
                                                           stub_graph, break_it):
    P, Zs = fold
    calls = stub_graph(Zs)
    args = _gargs(tmp_path)
    recorded = _original_run(sw, P, args)
    f = sw.dump_file(args, "hc", "transductive", "gate", 1.0, 1, 42)

    expect = recorded["R2"]
    if break_it == "wrong number":
        expect += 0.05                      # the dump is not the cloud behind that row
    elif break_it == "no dump":
        f.unlink()
    else:
        args.reuse_dumps = False
    spec = {"combos": ("cls+prot+mol",), "fill": True,
            "expect": {"cls+mol": {"R2": expect}}, "inherit": {"rsa_esm": 0.456}}
    rows = sw._graph_rows("gate", 1.0, 1, 42, "hc", P, args, spec)
    assert len(calls) == 2, f"{break_it}: the graph had to be trained again"
    assert {r["combo"] for r in rows} == {"cls+prot+mol"}
    assert {r["z_source"] for r in rows} == {"retrained"}


def _sargs(out, **kw):
    """The parsed CLI a real invocation would hand `sweep`, serial, one fold, one alpha."""
    return argparse.Namespace(**(dict(
        out=str(out), variant=None, mol_source="chemberta", nodes="esm", dial="nodes",
        mix_seed=0, no_mix_renorm=False, seed_graph=True, score_train=True, n_perm=0,
        dump_embeddings=True, dump_predictions=True, reuse_dumps=True,
        combos=list(("cls+mol", "cls+prot+mol")), seeds=[42], alphas=[1.0],
        legacy=False, gate=True, baselines_only=False, folds=[1], n_repeats=None,
        force=False, gpus=None, per_gpu=None, max_parallel=1, n_models=1, epochs=1,
        prot_embeddings=None, mol_embeddings=None, pool_fold=1) | kw))


def test_a_finished_directory_gets_the_missing_head_on_the_next_invocation(
        sw, tmp_path, fold, stub_graph, monkeypatch):
    """THE scenario: a directory produced when the sweep fitted cls+mol only, reopened
    by the sweep that fits both. The parent's bookkeeping -- resume, the recorded row the
    dump is checked against, dedup, the three files -- is exercised for real; only the
    data loading and the graph training are stubbed."""
    P, Zs = fold
    calls = stub_graph(Zs)
    monkeypatch.setattr(sw, "_prepare", lambda ds, args: None)
    monkeypatch.setattr(sw, "_fold_prep", lambda ds, regime, f, args, data: P)
    monkeypatch.setattr(sw, "visible_gpus", lambda: [])

    sw.sweep("hc", "transductive", _sargs(tmp_path, combos=["cls+mol"]))
    assert len(calls) == 1
    m, v, r = sw.sibling_paths("hc", "transductive", _sargs(tmp_path))
    before = pd.read_csv(r)
    old = before[before.arm == "gate"].sort_values("split").reset_index(drop=True)
    assert set(old.combo) == {"cls+mol"}

    sw.sweep("hc", "transductive", _sargs(tmp_path))
    assert len(calls) == 1, "the missing head must come from the verified dump"
    rec = pd.read_csv(r)
    g = rec[rec.arm == "gate"]
    assert g.groupby("combo").size().to_dict() == {"cls+mol": 3, "cls+prot+mol": 3}
    assert set(g[g.combo == "cls+prot+mol"].z_source) == {"dump"}
    kept = g[g.combo == "cls+mol"].sort_values("split").reset_index(drop=True)
    assert kept.R2.tolist() == old.R2.tolist(), "the recorded head was rewritten"
    assert len(rec[rec.arm == "boost_full"]) == 3, "the baselines were recomputed"
    # the test and val views carry both heads too, so a reader MUST choose one
    assert set(pd.read_csv(m).query("arm == 'gate'").combo) == {"cls+mol",
                                                                "cls+prot+mol"}
    assert set(pd.read_csv(v).query("arm == 'gate'").combo) == {"cls+mol",
                                                                "cls+prot+mol"}

    sw.sweep("hc", "transductive", _sargs(tmp_path))
    assert len(calls) == 1 and len(pd.read_csv(r)) == len(rec), \
        "a complete directory must have nothing left to do"


# ---------------------------------------------------------------------- readers

def _grid_file(path, with_combo=True):
    rows = []
    for f in (1, 2):
        base = dict(fold=f, seed=42, status="ok")
        rows.append(base | dict(arm="boost_full", alpha=np.nan, R2=0.5))
        rows.append(base | dict(arm="gate", alpha=1.0, R2=0.6))
        if with_combo:
            rows.append(base | dict(arm="gate", alpha=1.0, R2=0.7, combo="cls+prot+mol"))
    d = pd.DataFrame(rows)
    if with_combo:
        d.loc[d.arm == "boost_full", "combo"] = "prot+mol"
        d.loc[d.arm.eq("gate") & d.combo.isna(), "combo"] = "cls+mol"
    d.to_csv(path, index=False)


def test_readers_keep_to_one_head(tmp_path):
    """The trap the column opens: two heads under one arm/alpha/fold/seed, averaged into
    one number by any reader that does not choose."""
    from scripts.analysis import alpha_grid as ag
    _grid_file(tmp_path / "metrics_hc_rand_chemberta_nodedial.csv")
    d = ag.load(root=tmp_path, nodes="nodedial")
    assert set(d[d.arm == "gate"].R2) == {0.6}
    assert len(d[d.arm == "boost_full"]) == 2
    d = ag.load(root=tmp_path, nodes="nodedial", combo="cls+prot+mol")
    assert set(d[d.arm == "gate"].R2) == {0.7}
    assert len(d[d.arm == "boost_full"]) == 2, "the reference arm belongs to both views"


def test_a_file_from_before_the_column_reads_as_cls_mol(tmp_path):
    from scripts.analysis import alpha_grid as ag
    _grid_file(tmp_path / "metrics_hc_rand_chemberta_nodedial.csv", with_combo=False)
    d = ag.load(root=tmp_path, nodes="nodedial")
    assert set(d[d.arm == "gate"].R2) == {0.6}
    d = ag.load(root=tmp_path, nodes="nodedial", combo="cls+prot+mol")
    assert d[d.arm == "gate"].empty and len(d[d.arm == "boost_full"]) == 2
