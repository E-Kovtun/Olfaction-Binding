"""Attention heads for curated receptor--molecule interaction experiments."""
from __future__ import annotations

import torch
from torch import nn


SETUPS = ("flat_cross", "flat_self", "site_cross", "site_self")


def masked_mean(x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if padding_mask is None:
        return x.mean(1)
    keep = (~padding_mask).unsqueeze(-1).to(x.dtype)
    return (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def self_attention(self, x, padding_mask=None):
        z, _ = self.attn(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        x = self.norm1(x + z)
        return self.norm2(x + self.ff(x))

    def cross_attention(self, query, context, context_padding_mask=None):
        z, _ = self.attn(query, context, context,
                         key_padding_mask=context_padding_mask, need_weights=False)
        query = self.norm1(query + z)
        return self.norm2(query + self.ff(query))


class InteractionAttention(nn.Module):
    """Four controlled attention setups described in the screening experiment.

    Flat variants treat embedding coordinates as scalar tokens. Site variants use
    residues and atoms as tokens. Cross variants let protein tokens query only the
    molecule domain; self variants allow every token to attend to every other token.
    """
    def __init__(self, setup: str, dim: int = 32, heads: int = 4,
                 layers: int = 1, dropout: float = 0.1,
                 protein_dim: int = 1280, molecule_dim: int = 300):
        super().__init__()
        if setup not in SETUPS:
            raise ValueError(f"unknown setup {setup!r}; expected one of {SETUPS}")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.setup = setup
        self.protein_dim, self.molecule_dim = protein_dim, molecule_dim
        self.domain = nn.Embedding(2, dim)

        if setup.startswith("flat"):
            self.protein_in = nn.Linear(1, dim)
            self.molecule_in = nn.Linear(1, dim)
            self.protein_position = nn.Parameter(torch.empty(1, protein_dim, dim))
            self.molecule_position = nn.Parameter(torch.empty(1, molecule_dim, dim))
            nn.init.normal_(self.protein_position, std=0.02)
            nn.init.normal_(self.molecule_position, std=0.02)
        elif setup == "site_cross":
            # Q and K/V projections naturally bridge unequal 1280/300 input widths.
            self.protein_in = nn.Linear(protein_dim, dim)
            self.molecule_in = nn.Linear(molecule_dim, dim)
        else:
            # Explicit requested GIN 300 -> 1280 MLP before all-to-all attention.
            self.molecule_to_protein = nn.Sequential(
                nn.Linear(molecule_dim, 640), nn.GELU(), nn.LayerNorm(640),
                nn.Linear(640, protein_dim), nn.GELU(), nn.LayerNorm(protein_dim),
            )
            self.shared_in = nn.Linear(protein_dim, dim)

        self.blocks = nn.ModuleList(AttentionBlock(dim, heads, dropout) for _ in range(layers))
        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, 1),
        )

    def _flat_tokens(self, protein, molecule):
        p = self.protein_in(protein.unsqueeze(-1)) + self.protein_position
        m = self.molecule_in(molecule.unsqueeze(-1)) + self.molecule_position
        p = p + self.domain.weight[0]
        m = m + self.domain.weight[1]
        return p, m

    def _site_tokens(self, protein, molecule):
        if self.setup == "site_cross":
            p, m = self.protein_in(protein), self.molecule_in(molecule)
        else:
            p = self.shared_in(protein)
            m = self.shared_in(self.molecule_to_protein(molecule))
        return p + self.domain.weight[0], m + self.domain.weight[1]

    def forward(self, protein, molecule, protein_padding_mask=None, molecule_padding_mask=None):
        if self.setup.startswith("flat"):
            p, m = self._flat_tokens(protein, molecule)
        else:
            p, m = self._site_tokens(protein, molecule)

        if self.setup.endswith("cross"):
            for block in self.blocks:
                p = block.cross_attention(p, m, molecule_padding_mask)
        else:
            x = torch.cat([p, m], dim=1)
            mask = None
            if protein_padding_mask is not None:
                mask = torch.cat([protein_padding_mask, molecule_padding_mask], dim=1)
            for block in self.blocks:
                x = block.self_attention(x, mask)
            p, m = x[:, :p.shape[1]], x[:, p.shape[1]:]

        pooled = torch.cat([
            masked_mean(p, protein_padding_mask),
            masked_mean(m, molecule_padding_mask),
        ], dim=-1)
        return self.head(pooled).squeeze(-1)
