"""The v8 gate must do exactly three things, and these pin all three.

1. `alpha=None` leaves the historical graph untouched -- no branch, no scaling.
2. `alpha=0` puts the receptor cloud ON the ESM geometry, whatever training does,
   because the frozen branch is the only term left.
3. the structural branch is fit on TRAIN receptors and cannot see the others.

Everything here is CPU-only and tiny; the readout is RSA, the same cosine-rank
correlation the mechanism holdout scores with, so "same geometry" means the same
thing it means in the results.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from orbind.gnn_extractor import (  # noqa: E402
    ETYPE, ETYPE_NEG, PROT, RTYPE, RTYPE_NEG, _SignedSage, _structural_anchor)
from scripts.modeling.analysis.mechanism_holdout import GEOMETRY, emb_sim  # noqa: E402


@pytest.fixture
def graph():
    """8 molecules, 6 receptors, a handful of signed edges -- shapes, not science."""
    rng = np.random.default_rng(0)
    x_mol = torch.tensor(rng.normal(size=(8, 12)), dtype=torch.float32)
    x_prot = torch.tensor(rng.normal(size=(6, 20)), dtype=torch.float32)
    e = lambda n: torch.tensor(np.stack([rng.integers(0, 8, n), rng.integers(0, 6, n)]),
                               dtype=torch.long)
    pos, neg = e(14), e(11)
    pos_eidx = {ETYPE: pos, RTYPE: pos.flip(0)}
    neg_eidx = {ETYPE_NEG: neg, RTYPE_NEG: neg.flip(0)}
    return x_mol, x_prot, pos_eidx, neg_eidx


def build(x_prot, hidden=16, alpha=None, s=None):
    torch.manual_seed(7)
    return _SignedSage(12, 20, hidden, 0.0, alpha=alpha, s_prot=s)


def test_alpha_none_is_the_historical_model(graph):
    """No gate attribute set -> no buffer, no normalisation, identical output."""
    x_mol, x_prot, pe, ne = graph
    a = build(x_prot).encode(x_mol, x_prot, pe, ne)
    b = build(x_prot).encode(x_mol, x_prot, pe, ne)
    assert torch.allclose(a[PROT], b[PROT])
    assert build(x_prot).s_prot is None


def test_alpha_zero_reproduces_the_esm_geometry(graph):
    """The endpoint that makes the dial legible: at alpha=0 the receptor cloud is a
    frozen linear image of ESM, so RSA against ESM is ~1 no matter what the graph
    does -- including after the weights are scrambled."""
    x_mol, x_prot, pe, ne = graph
    s, k = _structural_anchor(x_prot, range(6), 16)
    m = build(x_prot, alpha=0.0, s=s)
    z = m.encode(x_mol, x_prot, pe, ne)[PROT].detach().numpy()
    assert k == 6                                   # min(hidden, n_train, dim)
    # The claim is algebraic -- the branch is an isometry on the train span, so the
    # whole cosine matrix survives. RSA is a rank correlation over only 15 pairs
    # here, so float32 rounding can still swap two near-tied ones; the cosines are
    # the exact statement and RSA is what the reported readout will see.
    assert np.allclose(emb_sim(z), emb_sim(x_prot.numpy()), atol=1e-5)
    assert GEOMETRY["rsa"](z, x_prot.numpy()) > 0.99
    with torch.no_grad():                           # "training" cannot move it
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 3.0)
    z2 = m.encode(x_mol, x_prot, pe, ne)[PROT].detach().numpy()
    assert np.allclose(z, z2)


def test_alpha_one_keeps_the_graph_geometry(graph):
    """The other endpoint: alpha=1 is the historical embedding up to one global
    scalar, and no geometry readout here can see a global scalar."""
    x_mol, x_prot, pe, ne = graph
    s, _ = _structural_anchor(x_prot, range(6), 16)
    z_gate = build(x_prot, alpha=1.0, s=s).encode(x_mol, x_prot, pe, ne)[PROT].detach().numpy()
    z_hist = build(x_prot).encode(x_mol, x_prot, pe, ne)[PROT].detach().numpy()
    for g in ("rsa", "cca", "procrustes"):
        assert GEOMETRY[g](z_gate, z_hist) > 0.99


def test_the_anchor_is_fit_on_train_receptors_only():
    """Held-out receptors are projected through the train basis; changing one of
    them must not move the others' coordinates."""
    rng = np.random.default_rng(1)
    X = torch.tensor(rng.normal(size=(9, 20)), dtype=torch.float32)
    train = range(6)
    a, _ = _structural_anchor(X, train, 8)
    X2 = X.clone()
    X2[7] += 50.0                                   # a test receptor moves far away
    b, _ = _structural_anchor(X2, train, 8)
    assert torch.allclose(a[:7], b[:7], atol=1e-4)
    assert not torch.allclose(a[7], b[7])


def test_intermediate_alpha_lies_between_the_two_geometries(graph):
    """The dial has to be monotone to be worth turning: as alpha falls, agreement
    with ESM rises."""
    x_mol, x_prot, pe, ne = graph
    s, _ = _structural_anchor(x_prot, range(6), 16)
    rsa = [GEOMETRY["rsa"](build(x_prot, alpha=a, s=s)
                           .encode(x_mol, x_prot, pe, ne)[PROT].detach().numpy(),
                           x_prot.numpy())
           for a in (1.0, 0.75, 0.5, 0.25, 0.0)]
    # the endpoints are guaranteed by construction; in between we only ask that the
    # dial travels one way -- a mixture of two clouds need not be exactly monotone
    assert rsa[-1] == max(rsa) and rsa[0] == min(rsa), f"endpoints wrong: {rsa}"
    assert rsa[-1] - rsa[0] > 0.5
    from scipy.stats import spearmanr
    assert spearmanr(rsa, [1.0, 0.75, 0.5, 0.25, 0.0]).statistic < -0.89
