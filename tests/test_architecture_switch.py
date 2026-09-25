"""The `conv` switch: which message-passing operator the signed stacks are built from.

The architecture ablation stands on one promise -- a row differs from ours in the
OPERATOR and in nothing else. So these tests are about the switch's edges rather than
about any operator's quality:

* the default is `sage` and its construction path is untouched, or every number in the
  paper silently belongs to a different model;
* an unknown operator is refused at construction, not at epoch 300 on a GPU;
* GAT's head arithmetic has to divide, because layer 1 concatenates the heads back to
  `hidden` and a comparison at a different width is a comparison of widths;
* the magnitude edge weighting is implemented for `sage` alone and must refuse the
  others rather than ignore the weights it was asked to use.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import torch_geometric_is_stubbed  # noqa: E402

from orbind.gnn_extractor import CONVS, GAT_HEADS, GnnSignedExtractor, _make_conv


def npz_pair(tmp_path):
    pp, mp = tmp_path / "p.npz", tmp_path / "m.npz"
    np.savez(pp, ids=np.array(["A", "B"], dtype=object),
             emb=np.zeros((2, 4), np.float32))
    np.savez(mp, ids=np.array(["X", "Y"], dtype=object),
             emb=np.zeros((2, 4), np.float32))
    return dict(name="cls", protein_path=str(pp), molecule_path=str(mp))


# ----------------------------------------------------------------- the list

def test_the_operators_are_the_four_we_argue_about():
    """Four operators, and one entry that is not an operator at all: `none` is the
    control in which no message passing happens, and it lives in the same list because
    the sweep selects a row by this name."""
    assert CONVS == ("sage", "gat", "graphconv", "gin", "none")


def test_plain_gcn_is_not_among_them():
    """`GCNConv` cannot be a row here: its symmetric normalisation and mandatory
    self-loops assume one node set, and this graph has two. `graphconv` is the
    bipartite-defined version of the same idea, and the table must not call it GCN."""
    assert "gcn" not in CONVS


# ----------------------------------------------------------------- the default

def test_the_default_is_ours(tmp_path):
    ext = GnnSignedExtractor(**npz_pair(tmp_path))
    assert ext.conv == "sage"
    assert ext.heads == GAT_HEADS


# ----------------------------------------------------------------- refusals

def test_an_unknown_operator_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError, match="conv must be one of"):
        GnnSignedExtractor(**npz_pair(tmp_path), conv="gcn")


def test_gat_heads_must_divide_the_width(tmp_path):
    kw = npz_pair(tmp_path)
    with pytest.raises(ValueError, match="divide by heads"):
        GnnSignedExtractor(**kw, conv="gat", hidden=100, heads=8)
    ok = GnnSignedExtractor(**kw, conv="gat", hidden=256, heads=8)
    assert ok.conv == "gat"


def test_heads_are_ignored_by_the_others(tmp_path):
    """A width that does not divide by `heads` is only GAT's problem."""
    ext = GnnSignedExtractor(**npz_pair(tmp_path), conv="gin", hidden=100, heads=8)
    assert ext.hidden == 100


def test_weighted_edges_refuse_every_operator_but_sage():
    """`edge_weight_mode='magnitude'` lives in `_WSAGE`. Another operator would drop
    the weights on the floor and report a weighted run that was not one.

    `none` is excluded because it has no operator at all: it refuses earlier, and for
    a different reason."""
    for kind in [c for c in CONVS if c not in ("sage", "none")]:
        with pytest.raises(ValueError, match="implemented for conv='sage'"):
            _make_conv(kind, 16, 1, 4, 0.1, weighted=True)
    with pytest.raises(ValueError, match="has no operator"):
        _make_conv("none", 16, 1, 4, 0.1, weighted=True)


# ----------------------------------------------------------------- the operators

@pytest.mark.skipif(torch_geometric_is_stubbed(),
                    reason="needs a real torch_geometric, not the import stub")
@pytest.mark.parametrize("kind", CONVS)
def test_every_operator_encodes_to_the_same_width(kind):
    """The decoder sees `2 * hidden` whatever the operator is -- GAT's layer 1 splits
    into heads and concatenates them back, and nothing else changes shape."""
    import torch

    from orbind.gnn_extractor import (ETYPE, ETYPE_NEG, MOL, PROT, RTYPE, RTYPE_NEG,
                                      _SignedSage)
    hidden = 16
    x_mol, x_prot = torch.randn(7, 11), torch.randn(5, 13)
    e_pos = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    e_neg = torch.tensor([[4, 5, 6, 0], [4, 4, 3, 1]])
    pos = {ETYPE: e_pos, RTYPE: e_pos.flip(0)}
    neg = {ETYPE_NEG: e_neg, RTYPE_NEG: e_neg.flip(0)}

    torch.manual_seed(0)
    m = _SignedSage(11, 13, hidden, 0.1, conv=kind, heads=4)
    z = m.encode(x_mol, x_prot, pos, neg)
    assert z[MOL].shape == (7, hidden)
    assert z[PROT].shape == (5, hidden)
    out = m.decode(z, torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(torch_geometric_is_stubbed(),
                    reason="needs a real torch_geometric, not the import stub")
def test_the_operators_are_actually_different_models():
    """Same seed, same input, four different answers -- otherwise the table would be
    four copies of one row and nobody would notice."""
    import torch

    from orbind.gnn_extractor import (ETYPE, ETYPE_NEG, PROT, RTYPE, RTYPE_NEG,
                                      _SignedSage)
    x_mol, x_prot = torch.randn(7, 11), torch.randn(5, 13)
    e_pos = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    e_neg = torch.tensor([[4, 5, 6, 0], [4, 4, 3, 1]])
    pos = {ETYPE: e_pos, RTYPE: e_pos.flip(0)}
    neg = {ETYPE_NEG: e_neg, RTYPE_NEG: e_neg.flip(0)}

    seen = []
    for kind in CONVS:
        torch.manual_seed(0)
        m = _SignedSage(11, 13, 16, 0.0, conv=kind, heads=4).eval()
        with torch.no_grad():
            seen.append(m.encode(x_mol, x_prot, pos, neg)[PROT])
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not torch.allclose(seen[i], seen[j])


# ----------------------------------------------------------------- sampling

def test_sampling_caps_the_neighbourhood_and_keeps_thin_ones_whole():
    """A fan-out bounds how much of the graph a node sees per layer. A neighbourhood
    smaller than the fan-out is kept entire -- padding it with duplicates would
    reweight that neighbour inside a mean aggregation."""
    import torch

    from orbind.gnn_extractor import sample_neighbours
    dst = torch.tensor([0] * 10 + [1] * 3 + [2])
    e = torch.stack([torch.arange(14), dst])
    g = torch.Generator(); g.manual_seed(0)
    out = sample_neighbours({"t": e}, 4, g)["t"]
    counts = torch.bincount(out[1], minlength=3).tolist()
    assert counts == [4, 3, 1]


def test_sampling_draws_without_replacement():
    import torch

    from orbind.gnn_extractor import sample_neighbours
    e = torch.stack([torch.arange(10), torch.zeros(10, dtype=torch.long)])
    g = torch.Generator(); g.manual_seed(1)
    for _ in range(20):
        kept = sample_neighbours({"t": e}, 4, g)["t"][0].tolist()
        assert len(set(kept)) == len(kept) == 4


def test_sampling_is_uniform_over_the_neighbourhood():
    """Every neighbour must have the same chance. A sampler that took the first k of
    an unshuffled edge list would bias towards whichever order the graph was built
    in, which on our hub core is the coverage ranking itself."""
    import collections

    import torch

    from orbind.gnn_extractor import sample_neighbours
    e = torch.stack([torch.arange(10), torch.zeros(10, dtype=torch.long)])
    g = torch.Generator(); g.manual_seed(2)
    seen = collections.Counter()
    for _ in range(2000):
        seen.update(sample_neighbours({"t": e}, 4, g)["t"][0].tolist())
    shares = [v / 2000 for v in seen.values()]
    assert len(seen) == 10
    assert max(shares) - min(shares) < 0.06      # expected 0.4 each


def test_sampling_is_off_by_default_and_a_no_op_at_zero():
    import torch

    from orbind.gnn_extractor import sample_neighbours
    e = torch.stack([torch.arange(10), torch.zeros(10, dtype=torch.long)])
    assert sample_neighbours({"t": e}, 0)["t"].shape[1] == 10
    assert sample_neighbours({"t": e}, None)["t"].shape[1] == 10


def test_an_empty_edge_type_survives_sampling():
    import torch

    from orbind.gnn_extractor import sample_neighbours
    e = torch.zeros((2, 0), dtype=torch.long)
    assert sample_neighbours({"t": e}, 5)["t"].shape == (2, 0)


def test_the_extractor_refuses_a_fanout_that_is_not_per_layer(tmp_path):
    """One number would silently mean 'both', and the paper's own setting is two
    different ones. At `layers=1` the same rule gives one entry, not two."""
    kw = npz_pair(tmp_path)
    with pytest.raises(ValueError, match="needs that many entries"):
        GnnSignedExtractor(**kw, fanout=(25,))
    with pytest.raises(ValueError, match="must be positive"):
        GnnSignedExtractor(**kw, fanout=(25, 0))
    ok = GnnSignedExtractor(**kw, fanout=(25, 10))
    assert ok.fanout == (25, 10)
    with pytest.raises(ValueError, match="needs that many entries"):
        GnnSignedExtractor(**kw, layers=1, fanout=(25, 10))
    assert GnnSignedExtractor(**kw, layers=1, fanout=(25,)).fanout == (25,)


# ----------------------------------------------------------------- the ablations

def test_the_edge_modes_are_the_three_we_argue_about():
    from orbind.gnn_extractor import EDGE_MODES
    assert EDGE_MODES == ("signed", "positive", "unsigned")


def test_the_extractor_refuses_a_nonsense_ablation(tmp_path):
    kw = npz_pair(tmp_path)
    with pytest.raises(ValueError, match="edges must be one of"):
        GnnSignedExtractor(**kw, edges="negative")
    with pytest.raises(ValueError, match="layers must be 1 or 2"):
        GnnSignedExtractor(**kw, layers=3)
    with pytest.raises(ValueError, match="cannot be combined"):
        GnnSignedExtractor(**kw, edges="unsigned", edge_weight_mode="magnitude")


def test_no_message_passing_carries_neither_sampling_nor_normalisation(tmp_path):
    """There are no layers, so there is nothing to sample per layer and nothing to
    normalise per layer. Forced at construction rather than ignored at run time, so the
    config written beside the numbers says what actually ran."""
    ext = GnnSignedExtractor(**npz_pair(tmp_path), conv="none",
                             fanout=(25, 10), normalize_layers=True)
    assert ext.fanout == () and ext.normalize_layers is False


@pytest.mark.skipif(torch_geometric_is_stubbed(),
                    reason="needs a real torch_geometric, not the import stub")
def test_positive_and_unsigned_drop_the_negative_stack(tmp_path):
    """The ablation moves the parameter count with it: an idle negative stack would
    leave the comparison flattering the ablated rows."""
    from orbind.gnn_extractor import _SignedSage
    signed = _SignedSage(11, 13, 16, 0.0)
    assert hasattr(signed, "conv1_neg") and hasattr(signed, "conv2_neg")
    for mode in ("positive", "unsigned"):
        m = _SignedSage(11, 13, 16, 0.0, signed=False)
        assert not hasattr(m, "conv1_neg") and not hasattr(m, "conv2_neg"), mode


@pytest.mark.skipif(torch_geometric_is_stubbed(),
                    reason="needs a real torch_geometric, not the import stub")
def test_one_layer_builds_one_layer(tmp_path):
    from orbind.gnn_extractor import _SignedSage
    m = _SignedSage(11, 13, 16, 0.0, layers=1)
    assert hasattr(m, "conv1") and hasattr(m, "conv1_neg")
    assert not hasattr(m, "conv2") and not hasattr(m, "conv2_neg")


@pytest.mark.skipif(torch_geometric_is_stubbed(),
                    reason="needs a real torch_geometric, not the import stub")
def test_no_message_passing_builds_no_convolution_at_all():
    from orbind.gnn_extractor import _SignedSage
    m = _SignedSage(11, 13, 16, 0.0, conv="none")
    assert not any(hasattr(m, f"conv{i}{s}") for i in (1, 2) for s in ("", "_neg"))
    assert hasattr(m, "proj_mol") and hasattr(m, "proj_prot") and hasattr(m, "dec")


@pytest.mark.skipif(torch_geometric_is_stubbed(),
                    reason="needs a real torch_geometric, not the import stub")
def test_every_ablation_still_encodes_to_the_same_width():
    """The decoder sees `hidden` columns whatever was ablated; a row that changed the
    width would be comparing widths."""
    import torch
    from orbind.gnn_extractor import (ETYPE, ETYPE_NEG, MOL, PROT, RTYPE,
                                      RTYPE_NEG, _SignedSage)
    hidden = 16
    e = torch.tensor([[0, 1], [0, 1]])
    pos = {ETYPE: e, RTYPE: e.flip(0)}
    neg = {ETYPE_NEG: e, RTYPE_NEG: e.flip(0)}
    x_mol, x_prot = torch.randn(3, 11), torch.randn(3, 13)
    for kw, p, n in ((dict(), pos, neg),
                     (dict(layers=1), pos, neg),
                     (dict(signed=False), pos, {}),
                     (dict(conv="none"), pos, {})):
        m = _SignedSage(11, 13, hidden, 0.0, **kw).eval()
        with torch.no_grad():
            z = m.encode(x_mol, x_prot, p, n)
        assert z[PROT].shape == (3, hidden), kw
        assert z[MOL].shape == (3, hidden), kw


def test_sampling_and_edge_weights_are_refused_together(tmp_path):
    """The sampler drops edges but does not carry their weights, so the two would
    disagree about which edge a weight belongs to."""
    with pytest.raises(ValueError, match="does not carry edge weights"):
        GnnSignedExtractor(**npz_pair(tmp_path), fanout=(25, 10),
                           edge_weight_mode="magnitude")


def test_both_additions_are_on_by_default(tmp_path):
    """DEFAULT MOVED 23.09.2026. The encoder now trains in GraphSAGE's own regime,
    because the architecture ablation measured the two additions and they won every
    paired comparison it made, on validation. A root trained before that date is a
    different model, which is why this assertion is worth a test of its own."""
    ext = GnnSignedExtractor(**npz_pair(tmp_path))
    assert ext.fanout == (25, 10) and ext.normalize_layers is True


def test_the_historical_encoder_is_still_reachable(tmp_path):
    """The architecture table's un-suffixed rows ARE this configuration; if it could
    not be asked for, that table would become seven copies of one regime."""
    ext = GnnSignedExtractor(**npz_pair(tmp_path), fanout=(), normalize_layers=False)
    assert ext.fanout == () and ext.normalize_layers is False
