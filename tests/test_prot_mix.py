"""The v9 node dial: receptor node features between identity and ESM.

    x_prot(rho) = mu + rho * centred(ESM) + (1 - rho) * centred(identity vectors)

v8 gated the graph's OUTPUT; this moves its INPUT, and its two ends are two models that
already existed -- the one-hot graph at rho=0 and the legacy graph at rho=1. What was
missing was the road between them, since "half a one-hot" is not a thing. Random
near-orthogonal vectors supply it: they live in ESM's own space, so the two can be mixed
continuously.

Three of the tests below are the design, not decoration:

  the centred-spread match   ESM is ~94% a vector every receptor shares. Matching TOTAL
                             energy hands the random end an order of magnitude more
                             receptor-separating signal and the dial becomes a step --
                             v8 measured that crossover at 0.05. So the test asserts
                             BALANCE at rho=0.5, which total-energy matching fails.
  the renormalisation        two clouds of equal spread mix to sqrt(rho^2+(1-rho)^2),
                             which dips to 0.71 in the middle. Uncorrected, a sag in any
                             curve there is the parameterisation, not a finding.
  determinism by NAME        the same receptor must get the same vector in every fold,
                             worker and dataset ordering. Python's `hash` is salted per
                             process, so this is a real way to get it wrong.
"""
import numpy as np
import pytest

# torch_geometric is stubbed in conftest where it is not installed
from orbind.gnn_extractor import (
    GnnSignedExtractor, _centred_rms, _identity_vectors, mix_protein_features)

DIM = 256


@pytest.fixture
def esm():
    """An ESM-shaped cloud: a large vector every receptor shares plus a small
    receptor-specific part. The share is what the scaling decision turns on."""
    rng = np.random.default_rng(0)
    mu = rng.normal(0, 1.0, DIM) * 3.0
    prot = {f"OR{i}": (mu + rng.normal(0, 0.25, DIM)).astype(np.float32)
            for i in range(40)}
    train = [f"OR{i}" for i in range(30)]
    return prot, train


def _cloud(d):
    return np.stack([d[r] for r in sorted(d)]).astype(np.float64)


def _centred(d, rows=None):
    A = _cloud(d)
    return A - (A if rows is None else A[rows]).mean(0)


# --------------------------------------------------------------------------- the ends

def test_rho_one_is_the_embedding_file_itself(esm):
    """The legacy end of the dial must BE the legacy input, not a rescaling away from
    it -- otherwise "rho=1 reproduces the legacy graph" is a claim rather than an
    identity, and the whole dial is anchored to nothing."""
    prot, train = esm
    got = mix_protein_features(prot, train, 1.0)
    assert set(got) == set(prot)
    for k in prot:
        assert np.array_equal(got[k], prot[k])


def test_rho_zero_keeps_identity_and_destroys_similarity(esm):
    """The other end. Every receptor stays distinguishable -- that is what a one-hot
    does -- but which receptors ESM called similar must leave no trace, or the "function
    alone" end still smuggles structure in."""
    prot, train = esm
    got = mix_protein_features(prot, train, 0.0)
    A, B = _centred(got), _centred(prot)

    def _off_diag_cos(M):
        U = M / np.linalg.norm(M, axis=1, keepdims=True)
        C = U @ U.T
        return C[~np.eye(len(C), dtype=bool)]

    # identity survives: no two receptors collapse onto each other
    assert np.abs(_off_diag_cos(A)).max() < 0.5
    # similarity does not: the two similarity structures are uncorrelated
    r = np.corrcoef(_off_diag_cos(A), _off_diag_cos(B))[0, 1]
    assert abs(r) < 0.15, f"rho=0 still carries ESM's similarity structure (r={r:+.3f})"


# -------------------------------------------------------------- the scaling decisions

def test_the_dial_is_balanced_in_the_middle_because_the_match_is_on_centred_spread(esm):
    """THE decision. ESM here is ~99% common mean, so a total-energy match would scale
    the random cloud to that huge norm and swamp the structural part: at rho=0.5 the
    mixture would be almost entirely random. Matching the CENTRED spread instead puts
    the halfway point actually halfway."""
    prot, train = esm
    A = _centred(mix_protein_features(prot, train, 0.5))
    E = _centred(prot)
    R = _identity_vectors(sorted(prot), DIM)
    R = R - R.mean(0)
    cos = lambda X, Y: float((X * Y).sum() / (np.linalg.norm(X) * np.linalg.norm(Y)))  # noqa: E731
    a_esm, a_rand = cos(A, E), cos(A, R)
    assert a_esm == pytest.approx(0.707, abs=0.08), a_esm
    assert a_rand == pytest.approx(0.707, abs=0.08), a_rand


def test_alignment_with_esm_climbs_monotonically_from_zero_to_one(esm):
    """The dial has to travel, and travel the way its name says."""
    prot, train = esm
    E = _centred(prot)
    got = []
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        A = _centred(mix_protein_features(prot, train, rho))
        got.append(float((A * E).sum() / (np.linalg.norm(A) * np.linalg.norm(E))))
    assert abs(got[0]) < 0.1 and got[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(b > a for a, b in zip(got, got[1:]))


def test_the_mixture_keeps_a_constant_spread_and_sags_without_the_correction(esm):
    """Two independent clouds of equal spread mix to sqrt(rho^2 + (1-rho)^2) -- 0.71 at
    the midpoint. Uncorrected, the middle of every curve is quieter than its ends for a
    reason that has nothing to do with the receptors."""
    prot, train = esm
    tr = [i for i, r in enumerate(sorted(prot)) if r in set(train)]
    spread = lambda d, **kw: _centred_rms(_centred(mix_protein_features(  # noqa: E731
        d, train, kw.pop("rho"), **kw), tr)[tr])
    ends = spread(prot, rho=1.0)
    assert spread(prot, rho=0.0) == pytest.approx(ends, rel=0.02)
    assert spread(prot, rho=0.5) == pytest.approx(ends, rel=0.02)
    sagged = spread(prot, rho=0.5, renorm=False)
    assert sagged == pytest.approx(ends / np.sqrt(2), rel=0.05)


def test_the_statistics_are_fit_on_train_receptors_only(esm):
    """Same discipline as the v8 structural anchor. Moving a held-out receptor must not
    move the transform every other receptor goes through."""
    prot, train = esm
    ref = mix_protein_features(prot, train, 0.5)
    moved = dict(prot)
    for r in set(prot) - set(train):
        moved[r] = moved[r] + 50.0
    got = mix_protein_features(moved, train, 0.5)
    for r in train:
        assert np.allclose(ref[r], got[r], atol=1e-4), r


# ------------------------------------------------------------------- the random vectors

def test_the_identity_vectors_are_near_orthogonal():
    """The property that makes them a one-hot: n random directions in d dimensions have
    expected |cos| ~ 1/sqrt(d), so they carry identity and no similarity."""
    R = _identity_vectors([f"OR{i}" for i in range(60)], 1280)
    C = R @ R.T
    np.fill_diagonal(C, 0.0)
    assert np.allclose(np.linalg.norm(R, axis=1), 1.0)
    assert np.abs(C).mean() < 3 / np.sqrt(1280)
    assert np.abs(C).max() < 0.25


def test_a_receptor_gets_the_same_vector_everywhere(esm):
    """Keyed by NAME, not position. Python's own `hash` is salted per process, so a
    worker would otherwise draw a different identity than the parent -- and the sweep
    runs a dozen workers."""
    prot, train = esm
    names = sorted(prot)
    a = _identity_vectors(names, DIM)
    b = _identity_vectors(list(reversed(names)), DIM)[::-1]
    assert np.array_equal(a, b)
    # and through the full mix, under a shuffled input dict
    shuffled = {k: prot[k] for k in reversed(names)}
    assert np.allclose(mix_protein_features(shuffled, train, 0.3)["OR7"],
                       mix_protein_features(prot, train, 0.3)["OR7"])


def test_the_draw_can_be_changed_but_not_at_the_legacy_end(esm):
    """`mix_seed` is the robustness check -- no conclusion should rest on one lucky set
    of directions -- and it must be inert at rho=1, which contains no random part."""
    prot, train = esm
    assert not np.allclose(mix_protein_features(prot, train, 0.0, seed=0)["OR3"],
                           mix_protein_features(prot, train, 0.0, seed=1)["OR3"])
    assert np.array_equal(mix_protein_features(prot, train, 1.0, seed=0)["OR3"],
                          mix_protein_features(prot, train, 1.0, seed=1)["OR3"])


# ------------------------------------------------------------------------- the contract

def test_the_extractor_refuses_the_two_node_switches_together(tmp_path):
    """`onehot_nodes` and `prot_mix` both replace the receptor node features, and
    prot_mix=0 IS the one-hot arm up to a rotation -- so silently letting one win would
    produce a run whose label says one thing and whose input is another."""
    import numpy as np
    pp, mp = tmp_path / "p.npz", tmp_path / "m.npz"
    np.savez(pp, ids=np.array(["A", "B"], dtype=object),
             emb=np.zeros((2, 4), np.float32))
    np.savez(mp, ids=np.array(["X", "Y"], dtype=object),
             emb=np.zeros((2, 4), np.float32))
    kw = dict(name="cls", protein_path=str(pp), molecule_path=str(mp))
    with pytest.raises(ValueError, match="never both"):
        GnnSignedExtractor(**kw, prot_mix=0.0, onehot_nodes=True)
    with pytest.raises(ValueError, match="prot_mix must be in"):
        GnnSignedExtractor(**kw, prot_mix=1.5)
    ok = GnnSignedExtractor(**kw, prot_mix=0.4)
    assert ok.prot_mix == 0.4 and ok.alpha is None


def test_out_of_range_is_refused_at_the_function_too(esm):
    prot, train = esm
    with pytest.raises(ValueError, match="prot_mix must be in"):
        mix_protein_features(prot, train, -0.1)
