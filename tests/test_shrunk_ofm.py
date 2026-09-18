"""The mask that cuts a complete insect panel down to M2OR's sparsity.

What is pinned here is the part that a wrong answer would not announce. The mask decides
which measurements the whole downstream experiment is allowed to see, and every failure
mode is silent: a density that drifts off the one being matched turns "we imposed M2OR's
shape" into "we kept more data than we said", a receptor emptied to zero cells leaves a
graph node with no edges (absent, not under-measured), and a marginal mask that does not
actually skew is indistinguishable in a results table from the random control it is
supposed to be contrasted with.

An earlier version of the repair step added cells to satisfy the per-row floor without
taking any back and overshot the requested 0.063 by half on the fly panel. Nothing in
the pipeline would have flagged that.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sh = _module("scripts/preprocessing/04_build_shrunk_ofm.py", "_shrunk_ofm_under_test")


@pytest.fixture
def prof():
    """A heavy-tailed profile in the same shape M2OR's is: most entities barely
    measured, a few measured almost everywhere."""
    rng = np.random.default_rng(0)
    return {"row": np.clip(rng.lognormal(-2.5, 1.2, 400), 0, 1),
            "col": np.clip(rng.lognormal(-3.0, 1.5, 400), 0, 1),
            "density": 0.0632, "shape": (400, 400)}


@pytest.mark.parametrize("kind", ["marginal", "random"])
@pytest.mark.parametrize("n_rec, n_mol, density", [(50, 110, 0.0632), (24, 110, 0.12),
                                                    (50, 110, 0.25)])
def test_the_density_is_exactly_what_was_asked_for(prof, kind, n_rec, n_mol, density):
    keep = sh.mask(n_rec, n_mol, density, prof, kind, seed=0)
    assert int(keep.sum()) == round(density * n_rec * n_mol)


@pytest.mark.parametrize("kind", ["marginal", "random"])
def test_no_receptor_and_no_odorant_is_emptied(prof, kind):
    keep = sh.mask(50, 110, 0.0632, prof, kind, seed=3, min_row=2, min_col=1)
    assert keep.sum(1).min() >= 2, "a receptor fell below the floor"
    assert keep.sum(0).min() >= 1, "an odorant vanished from the panel"
    assert int(keep.sum()) == round(0.0632 * 50 * 110), "the floor moved the density"


def test_an_impossible_floor_is_refused_rather_than_absorbed(prof):
    """The fly panel at M2OR's density has 167 cells to spend; two per odorant alone
    costs 220. The old code silently kept 329 and reported its own floor as a profile."""
    with pytest.raises(SystemExit, match="floor"):
        sh.mask(24, 110, 0.0632, prof, "marginal", seed=0, min_row=1, min_col=2)


def test_the_marginal_mask_skews_and_the_random_control_does_not(prof):
    """The whole point of the pair. If both came out flat there would be nothing to
    contrast, and the experiment would silently be a data-volume ablation."""
    kw = dict(n_rec=50, n_mol=110, density=0.25, prof=prof, seed=1)
    marg = sh.mask(kind="marginal", **kw).sum(1)
    rand = sh.mask(kind="random", **kw).sum(1)
    spread = lambda v: np.percentile(v, 90) / max(np.percentile(v, 50), 1)
    assert spread(marg) > 2 * spread(rand), (
        f"marginal p90/median {spread(marg):.2f} is not clearly above random "
        f"{spread(rand):.2f}")
    assert spread(rand) < 1.6, "the uniform control should be nearly flat"


def test_the_same_seed_gives_the_same_panel_and_a_different_one_does_not(prof):
    a = sh.mask(50, 110, 0.12, prof, "marginal", seed=7)
    b = sh.mask(50, 110, 0.12, prof, "marginal", seed=7)
    c = sh.mask(50, 110, 0.12, prof, "marginal", seed=8)
    assert np.array_equal(a, b), "the mask is not reproducible from its seed"
    assert not np.array_equal(a, c), "two seeds produced the same panel"


def test_ipf_hits_its_margins_when_they_are_reachable():
    """Away from saturation the fit should be tight -- this is the regime the bulk of
    the profile lives in, and a mask whose margins drifted here would not be matching
    M2OR's shape at all."""
    rng = np.random.default_rng(0)
    row_t = rng.uniform(1, 10, 30)          # 10 of 40 columns: far from the clip
    col_t = rng.uniform(1, 8, 40)
    col_t *= row_t.sum() / col_t.sum()
    P = sh._ipf(row_t, col_t)
    assert P.min() >= 0.0 and P.max() <= 1.0
    assert np.allclose(P.sum(1), row_t, rtol=0.02, atol=0.1)
    assert np.allclose(P.sum(0), col_t, rtol=0.02, atol=0.1)


def test_ipf_stays_a_probability_when_the_margins_are_not_reachable():
    """A row asking for nearly every column can only be served by saturating cells at 1,
    and the leftover mass has nowhere to go. The margins then miss -- which is a real
    limitation of the mask's head, documented in `_ipf` -- but the result must remain a
    probability, because `mask` samples from it."""
    rng = np.random.default_rng(0)
    row_t = rng.uniform(1, 40, 30)          # up to 40 of 40 columns: at the clip
    col_t = rng.uniform(1, 30, 40)
    col_t *= row_t.sum() / col_t.sum()
    P = sh._ipf(row_t, col_t)
    assert np.isfinite(P).all()
    assert P.min() >= 0.0 and P.max() <= 1.0
    assert np.isclose(P.max(), 1.0), "this fixture is meant to saturate; it no longer does"


# ------------------------------------------------- the mask constrains training only

class _Args:
    def __init__(self, **kw):
        self.density, self.mask, self.seed = 0.3, "marginal", 0
        self.min_per_receptor = self.min_per_molecule = 1
        self.dry_run, self.overwrite = False, True
        self.__dict__.update(kw)


@pytest.fixture
def panel(tmp_path, monkeypatch):
    """A complete 6x10 parent panel with both split families, wired so `build` writes
    into tmp_path instead of the repo."""
    recs, mols = [f"R{i}" for i in range(6)], [f"M{j}" for j in range(10)]
    pool = pd.DataFrame([{"SMILES": m, "Protein sequence": r, "output": float(10 * i + j)}
                         for i, r in enumerate(recs) for j, m in enumerate(mols)])
    ofm, mol_dir = tmp_path / "ofm", tmp_path / "mols"
    (ofm / "PARENT" / "raw").mkdir(parents=True)
    mol_dir.mkdir()
    pd.DataFrame({"smiles": mols, "inchikey": [f"K{j}" for j in range(10)]}).to_csv(
        mol_dir / "molecule_smiles_parent.csv", index=False)

    rng = np.random.default_rng(0)
    for f in sh.FOLDS:
        perm = rng.permutation(len(pool))
        d = ofm / "PARENT" / "rand_splits" / f"rand_split_{f}"
        d.mkdir(parents=True)
        for s, idx in (("train", perm[:36]), ("val", perm[36:48]), ("test", perm[48:])):
            pool.iloc[sorted(idx)].to_csv(d / f"{s}_df.csv", index=False)
        te = {mols[(2 * f) % 10], mols[(2 * f + 1) % 10]}
        va = {mols[(2 * f + 2) % 10]}
        d = ofm / "PARENT" / "our_inductive_splits" / f"our_inductive_split_{f}"
        d.mkdir(parents=True)
        m_te, m_va = pool["SMILES"].isin(te), pool["SMILES"].isin(va)
        pool[~m_te & ~m_va].to_csv(d / "train_df.csv", index=False)
        pool[m_va].to_csv(d / "val_df.csv", index=False)
        pool[m_te].to_csv(d / "test_df.csv", index=False)

    monkeypatch.setattr(sh, "OFM_DIR", ofm)
    monkeypatch.setattr(sh, "MOL_DIR", mol_dir)
    monkeypatch.setattr(sh, "_root", tmp_path)
    monkeypatch.setattr(sh, "ofm_pool", lambda d: pool.copy())
    monkeypatch.setattr(sh, "DATASETS", {
        "parent": {"dir": "PARENT", "families": ("rand", "our_inductive"),
                   "molecules": "molecule_smiles_parent.csv"},
        "kid": {"dir": "KID", "raw": pathlib.Path("KID") / "raw" / "kid_z.csv",
                "families": ("rand", "our_inductive"),
                "molecules": "molecule_smiles_kid.csv", "base": "parent"}})
    return pool, ofm


def _splits(ofm, family, f):
    d = ofm / "KID" / f"{family}_splits" / f"{family}_split_{f}"
    return {s: pd.read_csv(d / f"{s}_df.csv") for s in ("train", "val", "test")}


def _keys(df):
    return set(map(tuple, df[["SMILES", "Protein sequence"]].to_numpy()))


def test_the_written_pool_is_the_parent_complete(panel, prof):
    """The shrunk pool must keep every cell and every response. Writing only the
    measured rows is what made the test set 70 rows instead of 5200."""
    pool, ofm = panel
    sh.build("kid", _Args(), prof)
    got = pd.read_csv(ofm / "KID" / "raw" / "kid_z.csv")
    assert len(got) == len(pool)
    assert _keys(got) == _keys(pool)


@pytest.mark.parametrize("family", ["rand", "our_inductive"])
def test_training_sees_only_measured_cells_and_test_is_everything_else(panel, prof, family):
    pool, ofm = panel
    sh.build("kid", _Args(), prof)
    measured = _keys(pd.read_csv(ofm / "KID" / "raw" / "kid_measured.csv"))
    for f in sh.FOLDS:
        s = _splits(ofm, family, f)
        tr, va, te = (_keys(s[k]) for k in ("train", "val", "test"))
        assert tr <= measured, f"{family}/{f}: a training cell was never measured"
        assert va <= measured, f"{family}/{f}: a validation cell was never measured"
        assert not (tr & va) and not (tr & te) and not (va & te), "splits overlap"
        assert tr | va | te == _keys(pool), (
            f"{family}/{f}: the three splits do not partition the pool -- "
            f"ofm_indices checks exactly this")


@pytest.mark.parametrize("family", ["rand", "our_inductive"])
def test_the_parents_own_test_block_stays_recoverable(panel, prof, family):
    """What makes shrunk-vs-parent a PAIRED comparison. The big test is mostly cells the
    parent trained on; `origin == "upstream_test"` has to be the parent's test block row
    for row, or the two panels are scored on different material and no delta between
    them means anything."""
    pool, ofm = panel
    sh.build("kid", _Args(), prof)
    for f in sh.FOLDS:
        o = pd.read_csv(ofm / "KID" / f"{family}_splits" / f"{family}_split_{f}"
                        / "test_origin.csv")
        parent = _keys(pd.read_csv(ofm / "PARENT" / f"{family}_splits"
                                   / f"{family}_split_{f}" / "test_df.csv"))
        assert _keys(o[o.origin == "upstream_test"]) == parent, (
            f"{family}/{f}: the parent's test block drifted")


def test_effective_coldness_is_wider_than_the_split_and_that_is_deliberate(panel, prof):
    """Two different notions, and conflating them would misread the table. A molecule
    the split held out is cold; so is one whose every measurement the mask happened to
    delete, and the model cannot tell them apart. The second set is strictly larger --
    at M2OR's density on Carey it is 1933 cells against the split's 1100."""
    pool, ofm = panel
    sh.build("kid", _Args(), prof)
    for f in sh.FOLDS:
        o = pd.read_csv(ofm / "KID" / "our_inductive_splits"
                        / f"our_inductive_split_{f}" / "test_origin.csv")
        held_out = _keys(o[o.origin == "upstream_test"])
        cold = _keys(o[o.cold_molecule])
        assert held_out <= cold, (
            f"fold {f}: a held-out odorant was not marked cold -- the flag is computed "
            f"against the MASKED train split and must cover the split's own holdout")


def test_every_test_row_is_labelled_with_where_it_came_from(panel, prof):
    pool, ofm = panel
    sh.build("kid", _Args(), prof)
    for family in ("rand", "our_inductive"):
        for f in sh.FOLDS:
            s = _splits(ofm, family, f)
            o = pd.read_csv(ofm / "KID" / f"{family}_splits" / f"{family}_split_{f}"
                            / "test_origin.csv")
            assert len(o) == len(s["test"])
            assert _keys(o) == _keys(s["test"])
            assert set(o.origin) <= {"upstream_test", "unmeasured_train",
                                     "unmeasured_val"}
