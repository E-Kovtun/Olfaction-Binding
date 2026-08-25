"""The molecule-ranking filter: `q` -> how many, criterion -> which ones.

These are pure functions with no I/O, and they carry two pieces of semantics that
are easy to break silently and expensive to notice: the two readings of the
quantile, and what "positive" means when the target is continuous. Both changed
under us once already.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbind.mol_selection import (CRITERIA, K_MODES, compute_mol_scores, h2,
                                   keep_mask, quality_K, resolve_K,
                                   select_keep_mask)


# --------------------------------------------------------------------- fixtures

def binary_edges():
    """A small bipartite train set, 4 molecules x 3 proteins.

    mol0: 3 proteins (2 pos, 1 neg)   mol2: 1 protein (0 pos, 1 neg)
    mol1: 2 proteins (2 pos, 0 neg)   mol3: no edges at all
    """
    mol_ids = np.array([0, 0, 0, 1, 1, 2])
    prot_ids = np.array([0, 1, 2, 0, 1, 2])
    y = np.array([1, 0, 1, 1, 1, 0])
    return mol_ids, prot_ids, y, 4, 3


def ladder_edges():
    """Tie-free coverage 4/3/2/1/0 over 5 molecules, 4 proteins."""
    mol_ids, prot_ids = [], []
    for mol, degree in enumerate([4, 3, 2, 1, 0]):
        for prot in range(degree):
            mol_ids.append(mol)
            prot_ids.append(prot)
    y = np.ones(len(mol_ids), dtype=int)
    return np.array(mol_ids), np.array(prot_ids), y, 5, 4


# ----------------------------------------------------------------- resolve_K

def test_resolve_K_ignores_zero_coverage_molecules():
    cov = np.arange(11)  # one molecule with coverage 0, ten eligible
    assert resolve_K(cov, 0.0) == 10
    assert resolve_K(cov, 0.0, "fraction") == 10


def test_resolve_K_coverage_quantile_counts_above_the_quantile():
    cov = np.arange(11)
    # median of 0..10 is 5; cov >= 5 is {5,...,10}
    assert resolve_K(cov, 0.5, "coverage_quantile") == 6


def test_resolve_K_fraction_keeps_the_top_share():
    cov = np.arange(11)
    assert resolve_K(cov, 0.5, "fraction") == 5
    assert resolve_K(cov, 0.9, "fraction") == 1


def test_resolve_K_fraction_never_returns_zero():
    """A q so high it rounds to nothing still has to leave one molecule."""
    cov = np.arange(11)
    assert resolve_K(cov, 0.999, "fraction") == 1


def test_complete_matrix_makes_the_coverage_quantile_a_no_op():
    """The Carey/Hallem case, and the reason `fraction` exists.

    Every molecule is measured against every receptor, so coverage is constant,
    `np.quantile` returns that same value and `cov >= threshold` keeps
    everything -- q=0.99 and q=0 give the identical answer.
    """
    cov = np.full(70, 50)
    assert resolve_K(cov, 0.99, "coverage_quantile") == 70
    assert resolve_K(cov, 0.0, "coverage_quantile") == 70
    # the tie-free reading of the same intent still cuts
    assert resolve_K(cov, 0.99, "fraction") == 1
    assert resolve_K(cov, 0.5, "fraction") == 35


def test_resolve_K_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="k_mode"):
        resolve_K(np.arange(5), 0.5, "quantile")


def test_quality_K_is_the_coverage_quantile_alias():
    cov = np.arange(11)
    for q in (0.0, 0.25, 0.5, 0.9, 0.99):
        assert quality_K(cov, q) == resolve_K(cov, q, "coverage_quantile")


def test_k_modes_are_the_two_documented_ones():
    assert set(K_MODES) == {"coverage_quantile", "fraction"}


# ----------------------------------------------------------- compute_mol_scores

def test_scores_count_coverage_and_the_positive_mix():
    mol_ids, prot_ids, y, n_mol, n_prot = binary_edges()
    sc = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot, need_greedy=False)

    assert list(sc["cov"]) == [3, 2, 1, 0]
    assert sc["g_order"] is None

    S = sc["SCORE"]
    # disc_pairs = n_pos * n_neg: only mol0 has both
    assert list(S["disc_pairs"]) == [2.0, 0.0, 0.0, 0.0]
    assert S["balance_bits"][0] == pytest.approx(h2(2 / 3))
    assert S["entropy_bits"][0] == pytest.approx(3 * h2(2 / 3))
    # a molecule with no train edges scores zero everywhere
    for name in ("coverage", "balance_bits", "entropy_bits", "disc_pairs",
                 "idf_coverage", "composite"):
        assert S[name][3] == 0.0


def test_idf_is_zero_when_every_protein_is_touched_by_every_molecule():
    """Why idf_coverage and composite carry no information on a complete matrix:
    idf = log2(m_active / df) = log2(1) = 0 for every protein."""
    n_mol, n_prot = 6, 4
    mol_ids = np.repeat(np.arange(n_mol), n_prot)
    prot_ids = np.tile(np.arange(n_prot), n_mol)
    y = np.ones(len(mol_ids), dtype=int)

    S = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot, need_greedy=False)["SCORE"]
    assert np.allclose(S["idf_coverage"], 0.0)
    assert np.allclose(S["composite"], 0.0)


def test_continuous_target_without_a_threshold_is_degenerate():
    """The historical `y == 1` / `y == 0` rule matches nothing on a z-scored
    response, so every label-based criterion silently collapses to zero. This is
    the failure `pos_threshold` was added for -- keep it pinned."""
    mol_ids, prot_ids, _, n_mol, n_prot = binary_edges()
    y = np.array([1.5, -0.2, 0.8, 2.0, 0.3, -1.0])

    S = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot, need_greedy=False)["SCORE"]
    assert S["disc_pairs"].sum() == 0.0
    assert np.allclose(S["balance_bits"], 0.0)


def test_pos_threshold_restores_the_label_based_criteria():
    mol_ids, prot_ids, _, n_mol, n_prot = binary_edges()
    y = np.array([1.5, -0.2, 0.8, 2.0, 0.3, -1.0])

    S = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot,
                           need_greedy=False, pos_threshold=0.0)["SCORE"]
    # mol0 now has 2 positives (1.5, 0.8) and 1 negative (-0.2)
    assert list(S["disc_pairs"]) == [2.0, 0.0, 0.0, 0.0]
    assert S["balance_bits"][0] == pytest.approx(h2(2 / 3))


def test_pos_threshold_matches_the_binary_rule_when_it_should():
    """Threshold 0.5 over 0/1 labels is exactly `y == 1`."""
    mol_ids, prot_ids, y, n_mol, n_prot = binary_edges()
    a = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot, need_greedy=False)["SCORE"]
    b = compute_mol_scores(mol_ids, prot_ids, y.astype(float), n_mol, n_prot,
                           need_greedy=False, pos_threshold=0.5)["SCORE"]
    for name in a:
        assert np.allclose(a[name], b[name])


# ------------------------------------------------------------------ keep_mask

def test_keep_mask_takes_the_top_K_and_never_a_zero_coverage_molecule():
    mol_ids, prot_ids, y, n_mol, n_prot = binary_edges()
    sc = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot, need_greedy=False)

    assert list(keep_mask("coverage", sc, 2, n_mol)) == [True, True, False, False]
    # K past the eligible count keeps the eligible ones only -- mol3 stays out
    assert list(keep_mask("coverage", sc, 10, n_mol)) == [True, True, True, False]


def test_keep_mask_rejects_an_unknown_criterion():
    mol_ids, prot_ids, y, n_mol, n_prot = binary_edges()
    sc = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot, need_greedy=False)
    with pytest.raises(ValueError, match="unknown criterion"):
        keep_mask("degree", sc, 2, n_mol)


def test_greedy_needs_its_order_precomputed():
    mol_ids, prot_ids, y, n_mol, n_prot = binary_edges()
    sc = compute_mol_scores(mol_ids, prot_ids, y, n_mol, n_prot, need_greedy=False)
    with pytest.raises(ValueError, match="need_greedy"):
        keep_mask("greedy_pair_cover", sc, 2, n_mol)


def test_criteria_list_is_the_documented_seven():
    assert len(CRITERIA) == 7
    assert CRITERIA[-1] == "greedy_pair_cover"


# ------------------------------------------------------------ select_keep_mask

def test_default_criterion_and_mode_reproduce_the_plain_quantile_rule():
    """The docstring's contract: criterion='coverage' with the coverage-quantile
    reading is `counts >= quantile(counts, q)` bit for bit (coverage tie-free)."""
    mol_ids, prot_ids, y, n_mol, n_prot = ladder_edges()
    cov = np.bincount(mol_ids, minlength=n_mol)

    for q in (0.25, 0.5, 0.75):
        got = select_keep_mask("coverage", mol_ids, prot_ids, y, n_mol, n_prot, q)
        expected = cov >= np.quantile(cov, q)
        assert list(got) == list(expected), q
