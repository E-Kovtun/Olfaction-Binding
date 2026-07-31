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

  * Three message-passing (MP) modes (--mp_mode flag):
      pos_only  — only positive (binding) edges in the MP graph (default).
                  Negatives participate only through the supervision loss.
      all_edges — positive AND negative edges in the MP graph, treated equally.
                  The graph encodes co-occurrence regardless of sign.
      signed    — positive edges contribute +, negative edges contribute −.
                  Separate SAGEConv layers for each sign; explicit subtraction
                  in the update formula nudges representations apart.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

MOL, PROT = "molecule", "protein"
ETYPE     = (MOL,  "binds",         PROT)
RTYPE     = (PROT, "rev_binds",     MOL)
ETYPE_NEG = (MOL,  "no_binds",      PROT)
RTYPE_NEG = (PROT, "rev_no_binds",  MOL)

MP_MODES = ("pos_only", "all_edges", "signed")


def load_npz_dict(path):
    d = np.load(path, allow_pickle=True)
    if "ids" in d.files:
        return {k: v for k, v in zip(d["ids"].tolist(), d["emb"])}
    return {k: d[k] for k in d.files}


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


def quality_mol_mask(pos, neg, n_mol, quantile):
    """Boolean mask [n_mol] — True for molecules above coverage quantile.

    Coverage = total (pos + neg) measurements across the full dataset.
    Intended to filter message-passing edges to well-measured molecules only;
    supervision pairs are left untouched.
    """
    counts = np.bincount(np.concatenate([pos[:, 0], neg[:, 0]]), minlength=n_mol)
    threshold = np.quantile(counts, quantile)
    mask = counts >= threshold
    print(f"  quality filter q={quantile}: {mask.sum()} / {n_mol} molecules "
          f"(min {int(threshold)} measurements)")
    return mask


def _split_idx(n, fracs, rng):
    perm = rng.permutation(n)
    n_te = int(fracs[2] * n); n_va = int(fracs[1] * n)
    return perm[n_va + n_te:], perm[:n_va], perm[n_va:n_va + n_te]   # tr, va, te


def make_splits(pos, neg, n_mol, regime="transductive", fracs=(0.7, 0.15, 0.15), seed=42):
    """Return (splits dict, mp_pos [2,E+], mp_neg [2,E-]) for message passing."""
    rng = np.random.default_rng(seed)
    out = {}
    if regime == "transductive":
        ip = _split_idx(len(pos), fracs, rng); ineg = _split_idx(len(neg), fracs, rng)
        for s, a, b in zip(["train", "val", "test"], ip, ineg):
            out[s] = {"pos": pos[a], "neg": neg[b]}
    elif regime == "inductive_molecule":
        m_tr, m_va, m_te = _split_idx(n_mol, fracs, rng)
        sets = {"train": set(m_tr), "val": set(m_va), "test": set(m_te)}
        for s in sets:
            pm = np.array([m in sets[s] for m in pos[:, 0]])
            nm = np.array([m in sets[s] for m in neg[:, 0]])
            out[s] = {"pos": pos[pm], "neg": neg[nm]}
    else:
        raise ValueError(regime)
    mp_pos = torch.tensor(out["train"]["pos"].T, dtype=torch.long)
    mp_neg = torch.tensor(out["train"]["neg"].T, dtype=torch.long)
    return out, mp_pos, mp_neg


def edge_index_dict(mp_pos, mp_neg=None, mode="pos_only"):
    """Build message-passing edge index dict for the requested MP mode.

    pos_only  : {ETYPE: pos, RTYPE: pos_rev}
    all_edges : {ETYPE: pos+neg, RTYPE: (pos+neg)_rev}   — sign-blind
    signed    : {ETYPE: pos, RTYPE: pos_rev,
                 ETYPE_NEG: neg, RTYPE_NEG: neg_rev}       — two separate channels
    """
    if mode == "pos_only":
        return {ETYPE: mp_pos, RTYPE: mp_pos.flip(0)}
    elif mode == "all_edges":
        mp_all = torch.cat([mp_pos, mp_neg], dim=1)
        return {ETYPE: mp_all, RTYPE: mp_all.flip(0)}
    elif mode == "signed":
        return {ETYPE:     mp_pos,        RTYPE:     mp_pos.flip(0),
                ETYPE_NEG: mp_neg,        RTYPE_NEG: mp_neg.flip(0)}
    else:
        raise ValueError(f"mp_mode must be one of {MP_MODES}, got {mode!r}")


def sup_edges(split):
    """Supervision label_index [2, E] and labels for one split (pos=1, neg=0)."""
    pos = torch.tensor(split["pos"].T, dtype=torch.long)
    neg = torch.tensor(split["neg"].T, dtype=torch.long)
    idx = torch.cat([pos, neg], dim=1)
    y = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
    return idx, y


def split_train_for_probe(train_split, probe_frac=0.5, seed=42):
    """Split train labels into disjoint GNN and downstream-probe subsets.

    Positive and negative arrays are partitioned independently, preserving the
    class balance approximately. Only the GNN subset should become MP edges or
    decoder supervision; the probe subset is reserved for fitting XGBoost/MLP.
    """
    if not 0.0 < probe_frac < 1.0:
        raise ValueError(f"probe_frac must be between 0 and 1, got {probe_frac}")
    rng = np.random.default_rng(seed)
    gnn, probe = {}, {}
    for label in ("pos", "neg"):
        edges = train_split[label]
        if len(edges) < 2:
            raise ValueError(f"need at least 2 {label} train edges for a disjoint split")
        perm = rng.permutation(len(edges))
        n_probe = min(max(int(round(probe_frac * len(edges))), 1), len(edges) - 1)
        probe[label] = edges[perm[:n_probe]]
        gnn[label] = edges[perm[n_probe:]]
    gnn_edges = set(map(tuple, np.concatenate([gnn["pos"], gnn["neg"]], axis=0)))
    probe_edges = set(map(tuple, np.concatenate([probe["pos"], probe["neg"]], axis=0)))
    overlap = gnn_edges & probe_edges
    if overlap:
        raise ValueError(
            f"{len(overlap)} labeled pairs occur in both disjoint subsets; "
            "deduplicate conflicting pair labels before splitting")
    return gnn, probe


class HeteroLink(torch.nn.Module):
    """Heterogeneous bipartite link predictor.

    mp_mode controls how negatives enter the message-passing graph:
      pos_only  — negatives invisible to the encoder; only in the loss.
      all_edges — negatives added as ordinary edges (sign-blind SAGE).
      signed    — separate SAGEConv layers for pos/neg with explicit subtraction
                  so negatives push representations apart.
    """
    def __init__(self, hidden=256, dropout=0.3, mp_mode="pos_only", dec_layers=3):
        super().__init__()
        from torch_geometric.nn import HeteroConv, SAGEConv, Linear
        assert mp_mode in MP_MODES, f"mp_mode must be one of {MP_MODES}"
        self.mp_mode = mp_mode
        self.proj = torch.nn.ModuleDict({
            MOL: Linear(-1, hidden), PROT: Linear(-1, hidden)})
        self.conv1 = HeteroConv({ETYPE: SAGEConv((-1, -1), hidden),
                                 RTYPE: SAGEConv((-1, -1), hidden)}, aggr="sum")
        self.conv2 = HeteroConv({ETYPE: SAGEConv((-1, -1), hidden),
                                 RTYPE: SAGEConv((-1, -1), hidden)}, aggr="sum")
        if mp_mode == "signed":
            # Separate convolutions for negative edges; their outputs are subtracted.
            self.conv1_neg = HeteroConv({ETYPE_NEG: SAGEConv((-1, -1), hidden),
                                         RTYPE_NEG: SAGEConv((-1, -1), hidden)}, aggr="sum")
            self.conv2_neg = HeteroConv({ETYPE_NEG: SAGEConv((-1, -1), hidden),
                                         RTYPE_NEG: SAGEConv((-1, -1), hidden)}, aggr="sum")
        if dec_layers == 3:
            # 3-layer: 2·hidden → hidden → hidden//2 → 1
            self.dec = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden, hidden),      torch.nn.ReLU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden,     hidden // 2), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden // 2, 1))
        else:
            # 2-layer (legacy): 2·hidden → hidden → 1
            self.dec = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden, hidden), torch.nn.ReLU(),
                torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def encode(self, x_dict, eidx_dict, noise_eps=0.0):
        """Encode all nodes. With noise_eps>0, add SimGCL-style uniform noise to
        each message-passing layer's output (see simgcl_noise) — used to build
        the two contrastive views; noise_eps=0 (default) is the clean encoder
        every other caller (decode, the XGBoost probe) relies on."""
        import torch.nn.functional as F
        x = {k: F.relu(self.proj[k](v)) for k, v in x_dict.items()}
        if self.mp_mode == "signed":
            pos_eidx = {ETYPE:     eidx_dict[ETYPE],     RTYPE:     eidx_dict[RTYPE]}
            neg_eidx = {ETYPE_NEG: eidx_dict[ETYPE_NEG], RTYPE_NEG: eidx_dict[RTYPE_NEG]}
            x_p = self.conv1(x, pos_eidx)
            x_n = self.conv1_neg(x, neg_eidx)
            # ReLU after signed difference — negatives push representations away
            x = {k: F.relu(x_p[k] - x_n.get(k, torch.zeros_like(x_p[k]))) for k in x_p}
            if noise_eps > 0:
                x = {k: simgcl_noise(v, noise_eps) for k, v in x.items()}
            x_p = self.conv2(x, pos_eidx)
            x_n = self.conv2_neg(x, neg_eidx)
            x = {k: x_p[k] - x_n.get(k, torch.zeros_like(x_p[k])) for k in x_p}
            if noise_eps > 0:
                x = {k: simgcl_noise(v, noise_eps) for k, v in x.items()}
        else:
            x = {k: F.relu(v) for k, v in self.conv1(x, eidx_dict).items()}
            if noise_eps > 0:
                x = {k: simgcl_noise(v, noise_eps) for k, v in x.items()}
            x = self.conv2(x, eidx_dict)
            if noise_eps > 0:
                x = {k: simgcl_noise(v, noise_eps) for k, v in x.items()}
        return x

    def encode_history(self, x_dict, eidx_dict, depth=2):
        """Concatenate initial + per-layer embeddings per node.

        depth=2 (default): [x0 || x1 || x2]   — through the last (2nd) MP layer.
        depth=1          : [x0 || x1]         — through the first MP layer only.

        Richer protein features for unentangled probing without touching the decoder.
        """
        import torch.nn.functional as F
        x0 = {k: F.relu(self.proj[k](v)) for k, v in x_dict.items()}
        if self.mp_mode == "signed":
            pos_eidx = {ETYPE:     eidx_dict[ETYPE],     RTYPE:     eidx_dict[RTYPE]}
            neg_eidx = {ETYPE_NEG: eidx_dict[ETYPE_NEG], RTYPE_NEG: eidx_dict[RTYPE_NEG]}
            xp = self.conv1(x0, pos_eidx)
            xn = self.conv1_neg(x0, neg_eidx)
            x1 = {k: F.relu(xp[k] - xn.get(k, torch.zeros_like(xp[k]))) for k in xp}
            if depth >= 2:
                xp = self.conv2(x1, pos_eidx)
                xn = self.conv2_neg(x1, neg_eidx)
                x2 = {k: xp[k] - xn.get(k, torch.zeros_like(xp[k])) for k in xp}
        else:
            x1 = {k: F.relu(v) for k, v in self.conv1(x0, eidx_dict).items()}
            if depth >= 2:
                x2 = self.conv2(x1, eidx_dict)
        stages = [x0, x1, x2] if depth >= 2 else [x0, x1]
        return {k: torch.cat([s[k] for s in stages], dim=-1) for k in x1}

    def decode(self, z, label_index):
        zm = z[MOL][label_index[0]]
        zp = z[PROT][label_index[1]]            # molecule scored ONLY against proteins
        return self.dec(torch.cat([zm, zp], dim=-1)).squeeze(-1)

    def forward(self, x_dict, eidx_dict, label_index):
        return self.decode(self.encode(x_dict, eidx_dict), label_index)


# --------------------------------------------------------------------------- SimGCL contrastive add-on

def simgcl_noise(x, eps):
    """SimGCL embedding-space perturbation (Yu et al., SIGIR'22, Eq. 7):
    Delta = eps * normalize(U(0,1)) ⊙ sign(x), so ||Delta||_2 = eps and the
    noise stays in the same hyperoctant as x. Added at each message-passing
    layer to create the two contrastive views (see HeteroLink.encode)."""
    import torch.nn.functional as F
    return x + eps * torch.sign(x) * F.normalize(torch.rand_like(x), dim=-1)


def info_nce(z1, z2, tau=0.2):
    """InfoNCE contrastive loss (SimGCL Eq. 2): cosine similarity of the two
    views, temperature tau, positives on the diagonal. No learnable parameters
    — unlike DGI, SimGCL's contrastive head is parameter-free."""
    import torch.nn.functional as F
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


class HeteroDGI(torch.nn.Module):
    """Deep Graph Infomax auxiliary head for the heterogeneous encoder.

    DGI maximizes mutual information between per-node "patch" embeddings and a
    global graph summary: a bilinear discriminator learns to tell real node
    embeddings (which should agree with the summary) from embeddings produced
    on a corrupted graph (row-shuffled input features, same edges). Used as a
    self-supervised regularizer added to the link-prediction loss.

    Both node types leave HeteroLink.encode already in the same `hidden`-dim
    space, so this head only needs them pooled into one node set — the caller
    does that (scope="shared" = molecules+proteins together, scope="prot" =
    proteins only) and also builds the corrupted embeddings; this module just
    scores. Node embeddings are expected pre-normalized by the caller (our
    signed encoder's final layer has no activation, so its raw output is
    unbounded and would saturate sigmoid(mean(·))).
    """
    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden, hidden))
        torch.nn.init.xavier_uniform_(self.weight)

    def discriminate(self, z, summary):
        return z @ torch.matmul(self.weight, summary)

    def loss(self, z_pos, z_neg):
        import torch.nn.functional as F
        summary = torch.sigmoid(z_pos.mean(dim=0))
        pos = self.discriminate(z_pos, summary)
        neg = self.discriminate(z_neg, summary)
        return (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
