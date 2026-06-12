"""Heterogeneous bipartite link-prediction model: molecule <-> protein binding.

Design choices that encode the task constraints:

  * Graph (message-passing) edges = KNOWN POSITIVE pairs only. Tested negatives
    are NOT edges; unmeasured pairs are neither edges nor supervised — so absence
    of a measurement is IGNORED, never treated as a deterministic "no-bind".
    => no random negative sampling; negatives come only from M2OR responsive=0.

  * Bipartite by construction: the decoder always scores (molecule, protein);
    a molecule can never be scored against another molecule.

  * Two regimes:
      "transductive"        — all nodes known; hold out EDGES (≈ stratified).
      "inductive_molecule"  — hold out whole MOLECULES (≈ group_molecule);
                              new molecules appear at test with features but no
                              message-passing edges (cold start).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

MOL, PROT = "molecule", "protein"
ETYPE = (MOL, "binds", PROT)
RTYPE = (PROT, "rev_binds", MOL)


def load_npz_dict(path):
    d = np.load(path, allow_pickle=True)
    return {k: v for k, v in zip(d["ids"].tolist(), d["emb"])}


def build_nodes(pairs_csv, prot_npz, mol_npz):
    """Node features + known positive/negative (mol_idx, prot_idx) pairs."""
    pairs = pd.read_csv(pairs_csv)
    prot, mol = load_npz_dict(prot_npz), load_npz_dict(mol_npz)
    pairs = pairs[pairs["receptor"].isin(prot) & pairs["inchikey"].isin(mol)].copy()

    mol_ids = sorted(pairs["inchikey"].unique())
    prot_ids = sorted(pairs["receptor"].unique())
    mi = {k: i for i, k in enumerate(mol_ids)}
    pi = {k: i for i, k in enumerate(prot_ids)}
    pairs["m"] = pairs["inchikey"].map(mi)
    pairs["p"] = pairs["receptor"].map(pi)

    Xm = torch.tensor(np.stack([mol[k] for k in mol_ids]), dtype=torch.float)
    Xp = torch.tensor(np.stack([prot[k] for k in prot_ids]), dtype=torch.float)
    pos = pairs.loc[pairs["label"] == 1, ["m", "p"]].to_numpy()
    neg = pairs.loc[pairs["label"] == 0, ["m", "p"]].to_numpy()
    print(f"  nodes: {Xm.shape[0]} molecules, {Xp.shape[0]} proteins | "
          f"edges: {len(pos)} pos, {len(neg)} neg (tested)")
    return Xm, Xp, pos, neg, len(mol_ids)


def _split_idx(n, fracs, rng):
    perm = rng.permutation(n)
    n_te = int(fracs[2] * n); n_va = int(fracs[1] * n)
    return perm[n_va + n_te:], perm[:n_va], perm[n_va:n_va + n_te]   # tr, va, te


def make_splits(pos, neg, n_mol, regime="transductive", fracs=(0.7, 0.15, 0.15), seed=42):
    """Return dict split -> {'pos': [E,2], 'neg': [E,2]} and the train pos edges
    used for message passing (`mp`)."""
    rng = np.random.default_rng(seed)
    out = {}
    if regime == "transductive":
        # split edges (positives and negatives independently keep the ratio)
        ip = _split_idx(len(pos), fracs, rng); ineg = _split_idx(len(neg), fracs, rng)
        for s, a, b in zip(["train", "val", "test"], ip, ineg):
            out[s] = {"pos": pos[a], "neg": neg[b]}
    elif regime == "inductive_molecule":
        # hold out whole molecules: a pair goes to the split of its molecule
        m_tr, m_va, m_te = _split_idx(n_mol, fracs, rng)
        sets = {"train": set(m_tr), "val": set(m_va), "test": set(m_te)}
        for s in sets:
            pm = np.array([m in sets[s] for m in pos[:, 0]])
            nm = np.array([m in sets[s] for m in neg[:, 0]])
            out[s] = {"pos": pos[pm], "neg": neg[nm]}
    else:
        raise ValueError(regime)
    mp = torch.tensor(out["train"]["pos"].T, dtype=torch.long)   # [2, E] (mol, prot)
    return out, mp


def edge_index_dict(mp):
    """Bidirectional message-passing edges (positives only)."""
    return {ETYPE: mp, RTYPE: mp.flip(0)}


def sup_edges(split):
    """Supervision label_index [2, E] and labels for one split (pos=1, neg=0)."""
    pos = torch.tensor(split["pos"].T, dtype=torch.long)
    neg = torch.tensor(split["neg"].T, dtype=torch.long)
    idx = torch.cat([pos, neg], dim=1)
    y = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
    return idx, y


class HeteroLink(torch.nn.Module):
    def __init__(self, hidden=128, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import HeteroConv, SAGEConv, Linear
        self.proj = torch.nn.ModuleDict({
            MOL: Linear(-1, hidden), PROT: Linear(-1, hidden)})
        self.conv1 = HeteroConv({ETYPE: SAGEConv((-1, -1), hidden),
                                 RTYPE: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.conv2 = HeteroConv({ETYPE: SAGEConv((-1, -1), hidden),
                                 RTYPE: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def encode(self, x_dict, eidx_dict):
        import torch.nn.functional as F
        x = {k: F.relu(self.proj[k](v)) for k, v in x_dict.items()}
        x = {k: F.relu(v) for k, v in self.conv1(x, eidx_dict).items()}
        x = self.conv2(x, eidx_dict)
        return x

    def decode(self, z, label_index):
        zm = z[MOL][label_index[0]]
        zp = z[PROT][label_index[1]]            # molecule scored ONLY against proteins
        return self.dec(torch.cat([zm, zp], dim=-1)).squeeze(-1)

    def forward(self, x_dict, eidx_dict, label_index):
        return self.decode(self.encode(x_dict, eidx_dict), label_index)
