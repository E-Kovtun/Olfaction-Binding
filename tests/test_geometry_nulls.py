"""The fast permutation null must equal the naive one, or every z in section 7 is wrong.

`geometry_nulls` precomputes the cosine matrix, its ranks and the principal scores once and
then only permutes rows. That is valid because a row permutation of X permutes those objects
exactly -- these tests pin that identity, and pin the end-to-end equality against the naive
"shuffle the input and call the measure" path it replaced.
"""
import numpy as np
import pytest

from scripts.modeling.analysis.mechanism_holdout import (
    GEOMETRY, _pca, emb_sim, geometry_nulls, permuted_stats)


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 90))
    M = rng.normal(size=(40, 15)) + X[:, :15] * 0.4
    return X, M


def test_permuting_rows_permutes_the_similarity_matrix(data):
    X, _ = data
    p = np.random.default_rng(3).permutation(len(X))
    assert np.allclose(emb_sim(X[p]), emb_sim(X)[np.ix_(p, p)])


def test_permuting_rows_permutes_the_principal_scores(data):
    X, M = data
    p = np.random.default_rng(4).permutation(len(X))
    # sign of a principal direction is arbitrary, so compare up to a per-column sign
    a, b = _pca(X[p], 3), _pca(X, 3)[p]
    assert np.allclose(np.abs(a), np.abs(b))


@pytest.mark.parametrize("g", list(GEOMETRY))
def test_fast_null_equals_the_naive_null(data, g):
    X, M = data
    fn = GEOMETRY[g]
    slow_m, slow_sd = permuted_stats(lambda Xs: fn(Xs, M), X, 25)
    fast_m, fast_sd = geometry_nulls(X, M, 25)[g]
    assert fast_m == pytest.approx(slow_m, abs=1e-12)
    assert fast_sd == pytest.approx(slow_sd, abs=1e-12)


def test_null_is_reproducible_from_the_seed(data):
    X, M = data
    assert geometry_nulls(X, M, 10, seed=5) == geometry_nulls(X, M, 10, seed=5)
    assert geometry_nulls(X, M, 10, seed=5) != geometry_nulls(X, M, 10, seed=6)
