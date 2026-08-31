"""Tests for the from-scratch Adaline implementation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from fakereview.adaline import AdalineDivergedError, AdalineGD


@pytest.fixture
def separable():
    """A linearly separable toy problem: label depends on feature 0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X, y


def test_learns_a_separable_problem(separable):
    X, y = separable
    model = AdalineGD(lr=0.1, n_iter=500).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.9


def test_loss_decreases_monotonically(separable):
    """With a sane learning rate, batch GD on a convex loss never goes up."""
    X, y = separable
    model = AdalineGD(lr=0.01, n_iter=100, tol=None).fit(X, y)
    losses = np.array(model.losses_)
    assert np.all(np.diff(losses) <= 1e-12)


def test_predictions_are_binary(separable):
    X, y = separable
    model = AdalineGD(lr=0.1, n_iter=100).fit(X, y)
    assert set(np.unique(model.predict(X))).issubset({0, 1})


def test_diverges_loudly_on_huge_learning_rate(separable):
    """A too-large lr must raise, not silently return a degenerate model."""
    X, y = separable
    with pytest.raises(AdalineDivergedError):
        AdalineGD(lr=50.0, n_iter=200, tol=None).fit(X, y)


def test_divergence_is_caught_while_the_loss_is_still_finite(separable):
    """Regression test: a loss of 1e30 is diverged but passes np.isfinite.

    An earlier version only checked for inf/NaN, so a run whose loss reached
    ~1e34 was reported as converged and returned a model that predicted a
    single class for every input.
    """
    X, y = separable
    with pytest.raises(AdalineDivergedError, match="diverges"):
        AdalineGD(lr=3.0, n_iter=500, tol=None).fit(X, y)


def test_early_stopping_does_not_fire_on_a_worsening_loss(separable):
    """Stagnation and divergence are different things.

    `best_loss - loss < tol` is satisfied when the loss *grows*, so a naive
    early-stopping check silently terminates a diverging run and reports it as
    converged. Divergence must win that race.
    """
    X, y = separable
    with pytest.raises(AdalineDivergedError):
        AdalineGD(lr=3.0, n_iter=500, tol=1e-9, n_iter_no_change=5).fit(X, y)


def test_accepts_sparse_input(separable):
    """TF-IDF matrices are sparse; densifying them is not an option."""
    X, y = separable
    model_sparse = AdalineGD(lr=0.05, n_iter=100, tol=None).fit(sp.csr_matrix(X), y)
    model_dense = AdalineGD(lr=0.05, n_iter=100, tol=None).fit(X, y)
    # Same maths either way, so the learned weights should match closely.
    np.testing.assert_allclose(model_sparse.w_, model_dense.w_, rtol=1e-9, atol=1e-12)


def test_weights_stay_one_dimensional_with_sparse_input(separable):
    """Guards the np.matrix trap: X.T @ v can return a 2-D matrix."""
    X, y = separable
    model = AdalineGD(lr=0.05, n_iter=20, tol=None).fit(sp.csr_matrix(X), y)
    assert model.w_.ndim == 1


def test_early_stopping_reports_fewer_epochs(separable):
    X, y = separable
    model = AdalineGD(lr=0.3, n_iter=5000, tol=1e-9, n_iter_no_change=5).fit(X, y)
    assert model.n_iter_ < 5000
    assert len(model.losses_) == model.n_iter_


def test_reproducible_across_runs(separable):
    X, y = separable
    a = AdalineGD(lr=0.1, n_iter=50, random_state=7).fit(X, y)
    b = AdalineGD(lr=0.1, n_iter=50, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.w_, b.w_)


def test_rejects_mismatched_shapes():
    X = np.zeros((10, 3))
    y = np.zeros(9)
    with pytest.raises(ValueError, match="10 samples"):
        AdalineGD().fit(X, y)


def test_decision_function_matches_threshold(separable):
    """predict() must be exactly decision_function() >= 0.5."""
    X, y = separable
    model = AdalineGD(lr=0.1, n_iter=100).fit(X, y)
    expected = (model.decision_function(X) >= 0.5).astype(int)
    np.testing.assert_array_equal(model.predict(X), expected)
