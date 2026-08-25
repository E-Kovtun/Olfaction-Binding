"""GAT-based bipartite link-prediction model: molecule <-> protein binding.

Same three MP modes as hetero.py, but GraphSAGE is replaced by Graph Attention:

  pos_only  — GATConv over positive neighbors only.
              Attention weights are softmax-normalised → sum to +1 per node.

  all_edges — GATConv over all neighbors (pos ∪ neg), sign-blind.
              Attention weights are softmax-normalised → sum to +1 per node.

  signed    — Two separate GATConv modules, one per sign:
                positive attention  → softmax over N+(i) → sums to +1
                negative attention  → softmax over N-(i) → sums to +1
              Output: h_i = GATConv_pos(h, pos_edges) − GATConv_neg(h, neg_edges)
              Effective signed weight: positives +α, negatives −α, hence
                ∑_{j∈N+} α+_ij = +1,  ∑_{j∈N-} α-_ij = −1.

Architecture:
  proj          : Linear(-1, hidden)  for each node type  [lazy]
  conv1 (×1|×2): multi-head GAT, heads heads, hidden//heads per head → concat → hidden
  conv2 (×1|×2): single-head GAT, hidden → hidden
  decoder       : MLP(2·hidden → hidden → ReLU → Dropout → 1)

Shared data utilities (build_nodes, make_splits, edge_index_dict, …) are
imported directly from hetero so there is no duplication.
"""
from __future__ import annotations
import torch

from orbind.legacy.hetero import (
    MOL, PROT, ETYPE, RTYPE, ETYPE_NEG, RTYPE_NEG, MP_MODES,
    load_npz_dict, build_nodes, make_splits, edge_index_dict, sup_edges,
)

__all__ = ["HeteroGATLink"]


def _make_gat_conv(hidden: int, heads: int, dropout: float,
                   etype, rtype) -> "HeteroConv":
    """One HeteroConv with two GATConv layers (layer-1 style: multi-head concat)."""
    from torch_geometric.nn import HeteroConv, GATConv
    return HeteroConv({
        etype: GATConv((-1, -1), hidden // heads, heads=heads,
                       dropout=dropout, add_self_loops=False),
        rtype: GATConv((-1, -1), hidden // heads, heads=heads,
                       dropout=dropout, add_self_loops=False),
    }, aggr="sum")


def _make_gat_conv2(hidden: int, dropout: float,
                    etype, rtype) -> "HeteroConv":
    """One HeteroConv with two GATConv layers (layer-2 style: single-head)."""
    from torch_geometric.nn import HeteroConv, GATConv
    return HeteroConv({
        etype: GATConv((-1, -1), hidden, heads=1,
                       dropout=dropout, add_self_loops=False),
        rtype: GATConv((-1, -1), hidden, heads=1,
                       dropout=dropout, add_self_loops=False),
    }, aggr="sum")


class HeteroGATLink(torch.nn.Module):
    """Heterogeneous bipartite GAT link predictor."""

    def __init__(self, hidden: int = 128, heads: int = 4,
                 dropout: float = 0.3, mp_mode: str = "pos_only"):
        super().__init__()
        from torch_geometric.nn import Linear
        assert mp_mode in MP_MODES, f"mp_mode must be one of {MP_MODES}"
        assert hidden % heads == 0, f"hidden={hidden} must be divisible by heads={heads}"
        self.mp_mode = mp_mode

        self.proj = torch.nn.ModuleDict({
            MOL:  Linear(-1, hidden),
            PROT: Linear(-1, hidden),
        })

        # Positive (or all-edge) convolutions — used in every mode
        self.conv1 = _make_gat_conv(hidden, heads, dropout, ETYPE, RTYPE)
        self.conv2 = _make_gat_conv2(hidden, dropout, ETYPE, RTYPE)

        # Negative convolutions — signed mode only
        if mp_mode == "signed":
            self.conv1_neg = _make_gat_conv(hidden, heads, dropout, ETYPE_NEG, RTYPE_NEG)
            self.conv2_neg = _make_gat_conv2(hidden, dropout, ETYPE_NEG, RTYPE_NEG)

        self.dec = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def encode_history(self, x_dict, eidx_dict, depth=2):
        """Concatenate initial + per-layer embeddings per node.

        depth=2 (default): [x0 || x1 || x2]  — through the last (2nd) GAT layer.
        depth=1          : [x0 || x1]        — through the first GAT layer only.
        """
        import torch.nn.functional as F
        x0 = {k: F.elu(self.proj[k](v)) for k, v in x_dict.items()}
        if self.mp_mode == "signed":
            pos_eidx = {ETYPE:     eidx_dict[ETYPE],     RTYPE:     eidx_dict[RTYPE]}
            neg_eidx = {ETYPE_NEG: eidx_dict[ETYPE_NEG], RTYPE_NEG: eidx_dict[RTYPE_NEG]}
            xp = self.conv1(x0, pos_eidx)
            xn = self.conv1_neg(x0, neg_eidx)
            x1 = {k: F.elu(xp[k] - xn.get(k, torch.zeros_like(xp[k]))) for k in xp}
            if depth >= 2:
                xp = self.conv2(x1, pos_eidx)
                xn = self.conv2_neg(x1, neg_eidx)
                x2 = {k: xp[k] - xn.get(k, torch.zeros_like(xp[k])) for k in xp}
        else:
            x1 = {k: F.elu(v) for k, v in self.conv1(x0, eidx_dict).items()}
            if depth >= 2:
                x2 = self.conv2(x1, eidx_dict)
        stages = [x0, x1, x2] if depth >= 2 else [x0, x1]
        return {k: torch.cat([s[k] for s in stages], dim=-1) for k in x1}

    def encode(self, x_dict, eidx_dict):
        import torch.nn.functional as F
        # Project all node features to the shared hidden dimension
        x = {k: F.elu(self.proj[k](v)) for k, v in x_dict.items()}

        if self.mp_mode == "signed":
            pos_eidx = {ETYPE:     eidx_dict[ETYPE],     RTYPE:     eidx_dict[RTYPE]}
            neg_eidx = {ETYPE_NEG: eidx_dict[ETYPE_NEG], RTYPE_NEG: eidx_dict[RTYPE_NEG]}

            # Layer 1: signed attention
            # pos branch: ∑_{j∈N+(i)} α+_ij · W+ h_j   (α+ via softmax → sums to +1)
            # neg branch: ∑_{j∈N-(i)} α-_ij · W- h_j   (α- via softmax → sums to +1)
            # result:  pos − neg  →  effective neg weight sums to −1
            x_p = self.conv1(x, pos_eidx)
            x_n = self.conv1_neg(x, neg_eidx)
            x = {k: F.elu(x_p[k] - x_n.get(k, torch.zeros_like(x_p[k])))
                 for k in x_p}

            # Layer 2: same signed structure, no activation on output
            x_p = self.conv2(x, pos_eidx)
            x_n = self.conv2_neg(x, neg_eidx)
            x = {k: x_p[k] - x_n.get(k, torch.zeros_like(x_p[k]))
                 for k in x_p}
        else:
            # pos_only : eidx_dict has ETYPE/RTYPE with positive edges only
            # all_edges: eidx_dict has ETYPE/RTYPE with pos ∪ neg edges
            # In both cases standard GAT softmax applies → weights sum to +1
            x = {k: F.elu(v) for k, v in self.conv1(x, eidx_dict).items()}
            x = self.conv2(x, eidx_dict)

        return x

    def decode(self, z, label_index):
        zm = z[MOL][label_index[0]]
        zp = z[PROT][label_index[1]]
        return self.dec(torch.cat([zm, zp], dim=-1)).squeeze(-1)

    def forward(self, x_dict, eidx_dict, label_index):
        return self.decode(self.encode(x_dict, eidx_dict), label_index)
