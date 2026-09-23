"""Molecule-ranking criteria for the message-passing quality filter.

Single source of truth for the 7 ways of ranking molecules before the quantile
cut used by the graph pipeline's MP-edge filter and by the quantile-sweep study
(notebooks/graph/alternatives/protein_based_graph.ipynb, now display-only).

The quantile sets **how many** molecules to keep (`quality_K` — the coverage
quantile count, criterion-independent); the criterion picks **which** K to keep
(`keep_mask`). All scores are computed on a split's TRAIN edges only, in whatever
local molecule/protein index space the caller uses (0..n_mol-1 / 0..n_prot-1);
molecules with zero train coverage are never eligible.

Criteria (higher = kept first), ported verbatim from the study notebook:
  coverage          total (pos+neg) train measurements of the molecule
  balance_bits      binary entropy H2(pos_frac) of its pos/neg mix
  entropy_bits      coverage * balance_bits
  disc_pairs        n_pos * n_neg (discriminative pair count)
  idf_coverage      sum of protein IDF over the proteins it touches
  composite         balance_bits * idf_coverage
  greedy_pair_cover greedy max protein-pair-coverage ordering (rank, 0 = best)

And one that is not a criterion at all:
  random            K eligible molecules drawn uniformly, `select_seed` deciding the
                    draw. It is the CONTROL the criterion axis is read against: if a
                    ranked criterion does not beat molecules picked at random, then
                    what the graph buys is message passing per se and not the choice
                    of hubs, which is a different claim from the one we make. Draw it
                    with several seeds -- one draw is an anecdote, not a control.
"""
from __future__ import annotations

import numpy as np

CRITERIA = ["coverage", "balance_bits", "entropy_bits", "disc_pairs",
            "idf_coverage", "composite", "greedy_pair_cover", "random"]

#: `random` is a control, not a ranking: it has no score vector, it needs a seed, and
#: it belongs on a figure as a reference rather than as an eighth competitor.
CONTROL_CRITERIA = ("random",)

# criteria whose per-molecule score is a plain "higher is better" vector
_VECTOR_CRITERIA = ["coverage", "balance_bits", "entropy_bits", "disc_pairs",
                    "idf_coverage", "composite"]


def h2(p):
    """Binary entropy in bits (clipped)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _greedy_pair_cover_order(prof, active, n_prot):
    """Greedy ordering that repeatedly picks the molecule maximising newly
    covered protein PAIRS (matches the notebook's torch loop; runs on GPU when
    available). Returns a list of molecule ids, best first."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_mol = len(prof)
    V = torch.zeros(n_mol, n_prot, device=dev)
    for m in active:
        ps = sorted(prof[int(m)])
        if ps:
            V[int(m), torch.tensor(ps, device=dev)] = 1.0
    n_touch = V.sum(1)
    C = torch.zeros(n_prot, n_prot, device=dev)
    rem = set(int(m) for m in active)
    order = []
    while rem:
        r = torch.tensor(sorted(rem), device=dev)
        Vr = V[r]
        gain = n_touch[r] ** 2 - (Vr @ C * Vr).sum(1)
        j = int(torch.argmax(gain).item())
        m = int(r[j].item())
        v = V[m]
        C = torch.maximum(C, torch.outer(v, v))
        order.append(m)
        rem.discard(m)
    return order


def compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot,
                       need_greedy: bool = True, pos_threshold=None):
    """Per-molecule criterion scores from TRAIN edges.

    `mol_ids`/`prot_ids`/`y` are parallel arrays over train pairs (local indices,
    label 1/0). Returns a dict with `cov` (coverage vector), `SCORE` (dict of the
    six vector criteria) and `g_order` (greedy order list, or None if not needed).

    `pos_threshold` makes the label-based criteria usable on a CONTINUOUS target:
    positive becomes `y > pos_threshold` instead of `y == 1`, and negative its
    complement instead of `y == 0`. Pass the same value the graph uses for edge
    signs (`GnnSignedExtractor.edge_threshold`) so a molecule's "positive mix"
    means the same thing to the ranking and to the message passing. Left None,
    the historical exact-match rule applies and every count is empty on a
    z-scored response -- which is why those criteria used to be refused there.
    """
    mol_ids = np.asarray(mol_ids); prot_ids = np.asarray(prot_ids)
    y = np.asarray(y)
    if pos_threshold is None:
        pos_m = mol_ids[y == 1]
        neg_m = mol_ids[y == 0]
    else:
        is_pos = y > pos_threshold
        pos_m = mol_ids[is_pos]
        neg_m = mol_ids[~is_pos]

    cov = np.bincount(mol_ids, minlength=n_mol)
    npos = np.bincount(pos_m, minlength=n_mol)
    nneg = np.bincount(neg_m, minlength=n_mol)

    prof = [set() for _ in range(n_mol)]
    for m, p in zip(mol_ids, prot_ids):
        prof[int(m)].add(int(p))
    active = np.where(cov > 0)[0]

    pos_frac = np.divide(npos, np.maximum(cov, 1))
    balance = np.where(cov > 0, h2(pos_frac), 0.0)
    entropy_bits = cov * balance
    disc_pairs = npos.astype(float) * nneg.astype(float)

    df = np.zeros(n_prot)
    for s in prof:
        for p in s:
            df[p] += 1
    m_active = len(active)
    idf = np.where(df > 0, np.log2(max(m_active, 1) / np.maximum(df, 1)), 0.0)
    idf_cov = np.array([idf[list(s)].sum() if s else 0.0 for s in prof])
    composite = balance * idf_cov

    SCORE = {"coverage": cov.astype(float), "balance_bits": balance,
             "entropy_bits": entropy_bits, "disc_pairs": disc_pairs,
             "idf_coverage": idf_cov, "composite": composite}
    g_order = _greedy_pair_cover_order(prof, active, n_prot) if need_greedy else None
    return {"cov": cov, "SCORE": SCORE, "g_order": g_order}


def quality_K(cov, q: float) -> int:
    """How many molecules the quantile keeps: the coverage-quantile count
    (`counts >= quantile(counts, q)`), criterion-independent — identical to
    hetero.quality_mol_mask / gnn_extractor._mp_edges. `q` is a fraction in [0,1].

    Thin alias for `resolve_K(cov, q, "coverage_quantile")`, kept because it is
    the name every M2OR-era caller uses."""
    return resolve_K(cov, q, "coverage_quantile")


K_MODES = ("coverage_quantile", "fraction")


def resolve_K(cov, q: float, k_mode: str = "coverage_quantile") -> int:
    """How many molecules to keep, under either reading of `q`.

    `coverage_quantile` (the historical one, M2OR): keep every molecule whose
    coverage reaches the q-th quantile of the coverage distribution. On M2OR
    that distribution is long-tailed, so this isolates a hub core.

    `fraction`: keep the top (1-q) share of the eligible molecules outright.

    The two agree whenever coverage is tie-free, but on the Carey/Hallem
    matrices they diverge completely, because **every** molecule is measured
    against **every** receptor. There the coverage vector is constant on any
    transductive-style train split and near-constant on a cold-molecule one, so
    `np.quantile` returns that same value and `cov >= threshold` keeps
    everything -- q becomes a no-op and the whole sweep collapses to one point.
    Verified: on CC/our_inductive, q=0.99 and q=0 both give K=70 of 70.
    `fraction` is the tie-free reading of the same intent and is what those
    datasets have to use if the x-axis is to mean anything.
    """
    cov = np.asarray(cov)
    eligible = int((cov > 0).sum())
    if k_mode not in K_MODES:
        raise ValueError(f"k_mode must be one of {K_MODES}, got {k_mode!r}")
    if not q or q <= 0:
        return eligible
    if k_mode == "fraction":
        return max(1, int(round((1.0 - q) * eligible)))
    return int((cov >= np.quantile(cov, q)).sum())


def keep_mask(criterion: str, scores: dict, K: int, n_mol: int,
              select_seed: int = 0) -> np.ndarray:
    """Boolean [n_mol] mask of the K molecules to keep for message passing,
    chosen by `criterion` (top-K by its score among coverage>0; greedy uses its
    own order). Ties/coverage handling mirrors the study's `select`.

    `select_seed` is read by `random` only, and every other criterion ignores it --
    they are deterministic given the train edges, which is why no existing number
    moves when this argument appears."""
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}, have {CRITERIA}")
    cov = scores["cov"]
    if criterion == "random":
        # The control: K of the ELIGIBLE molecules, uniformly. Eligibility is the one
        # thing it shares with the ranked criteria -- a molecule with no train
        # coverage carries no edges, so drawing it would quietly shrink K and make the
        # control a smaller graph rather than a differently chosen one.
        eligible = np.flatnonzero(np.asarray(cov) > 0)
        rng = np.random.default_rng(int(select_seed))
        kept = rng.permutation(eligible)[:K]
    elif criterion == "greedy_pair_cover":
        order = scores["g_order"]
        if order is None:
            raise ValueError("greedy_pair_cover needs g_order (need_greedy=True)")
        kept = order[:K]
    else:
        ranked = [int(m) for m in np.argsort(-scores["SCORE"][criterion]) if cov[m] > 0]
        kept = ranked[:K]
    mask = np.zeros(n_mol, dtype=bool)
    mask[np.asarray(kept, dtype=int)] = True
    return mask


def select_keep_mask(criterion, mol_ids, prot_ids, y, n_mol, n_prot, q,
                     k_mode: str = "coverage_quantile", pos_threshold=None,
                     select_seed: int = 0):
    """One-shot convenience: TRAIN edges + (criterion, q) -> boolean keep mask
    [n_mol]. `q` sets K (see `resolve_K` for the two readings); the criterion
    picks which K. With the default k_mode and criterion='coverage' this
    reproduces `counts >= quantile(counts, q)` bit-for-bit.

    K is the SAME for every criterion including `random`, which is what makes the
    control a control: the graphs being compared differ in which molecules carry the
    messages and in nothing else -- not in how many, and not in how many edges."""
    sc = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot,
                            need_greedy=(criterion == "greedy_pair_cover"),
                            pos_threshold=pos_threshold)
    K = resolve_K(sc["cov"], q, k_mode)
    return keep_mask(criterion, sc, K, n_mol, select_seed=select_seed)
