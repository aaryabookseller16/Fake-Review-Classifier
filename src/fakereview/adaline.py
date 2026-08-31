"""Adaline (Adaptive Linear Neuron) implemented from scratch in NumPy.

Adaline is the 1960 Widrow-Hoff refinement of the perceptron. The one idea that
matters: instead of updating on the *thresholded* prediction, it computes the
gradient against a **continuous linear activation**. That makes the loss surface
differentiable and convex, so plain gradient descent provably converges -- given
a small enough learning rate.

This module keeps that pedagogy visible (the update rule is four lines in
`fit`) while being robust enough to train on a 139k-dimensional sparse TF-IDF
matrix, which is what the benchmark in the README actually does.

Compared with a naive textbook implementation, three things are added, each
because the naive version fails on real data:

1. **Sparse-matrix support.** Densifying TF-IDF would need ~35 GB. All the
   linear algebra is written so SciPy sparse matrices flow through unchanged.
2. **Divergence detection.** MSE gradient descent explodes for learning rates
   that look reasonable (lr=0.7 on this data sends the loss to 1e34). Note that
   such a loss is still a *finite float*, so checking ``np.isfinite`` alone is
   not enough -- the guard compares against the best loss seen so far.
3. **Convergence check.** Stops early once the loss stops improving, so the
   iteration budget can be set generously. Divergence is checked first: a
   worsening loss also looks "stagnant" to a naive tolerance test, which would
   otherwise terminate a diverging run and report it as converged.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["AdalineGD", "AdalineDivergedError"]


class AdalineDivergedError(RuntimeError):
    """Raised when the learning rate is too high and the loss blows up.

    Adaline's failure mode is not a gentle plateau -- the loss grows by orders
    of magnitude per epoch. It does not necessarily reach inf: a run here
    reached 1e34 and stayed finite. Failing loudly is better than returning a
    model whose predictions are all one class.
    """


class AdalineGD(ClassifierMixin, BaseEstimator):
    """Adaptive Linear Neuron trained by batch gradient descent.

    The learning algorithm below is written from scratch in NumPy. The
    scikit-learn base classes supply only plumbing -- ``get_params`` /
    ``set_params`` and the estimator tags that ``Pipeline`` requires -- so the
    model can be composed with ``TfidfVectorizer`` and persisted as one object.
    No part of the optimisation is inherited.

    The model is linear: ``z = Xw + b``. The *activation* is the identity, so
    the loss is plain mean squared error against the 0/1 labels::

        L(w, b) = mean((y - (Xw + b))**2)

    whose gradients are::

        dL/dw = -2/n * X.T @ (y - z)
        dL/db = -2/n * sum(y - z)

    Prediction thresholds the activation at 0.5, the midpoint of the 0/1 label
    encoding.

    Parameters
    ----------
    lr:
        Learning rate. The single most important knob -- too high and training
        diverges (see :class:`AdalineDivergedError`), too low and it crawls.
    n_iter:
        Maximum number of full passes over the training set.
    tol:
        Early-stopping threshold. Training stops when the loss improves by less
        than ``tol`` for ``n_iter_no_change`` consecutive epochs. Set to
        ``None`` to always run the full ``n_iter`` epochs.
    n_iter_no_change:
        How many stagnant epochs to tolerate before stopping early.
    random_state:
        Seed for the small random weight initialisation.

    Attributes
    ----------
    w_:
        Learned weight vector, shape ``(n_features,)``.
    b_:
        Learned bias (intercept).
    losses_:
        Mean squared error at each epoch -- the training curve plotted in the
        Streamlit app.
    n_iter_:
        Number of epochs actually run (``< n_iter`` if early stopping fired).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> model = AdalineGD(lr=0.1, n_iter=200).fit(X, y)
    >>> float((model.predict(X) == y).mean()) > 0.9
    True
    """

    #: Divergence trips when the loss exceeds this multiple of the best loss
    #: seen so far. Gradient descent on a small batch can spike transiently in
    #: the first few epochs and still converge, so the margin is wide -- but a
    #: genuine blow-up grows by 30+ orders of magnitude, so it is caught within
    #: a handful of epochs regardless.
    _DIVERGENCE_FACTOR = 1e6

    def __init__(
        self,
        lr: float = 0.01,
        n_iter: int = 100,
        tol: float | None = 1e-6,
        n_iter_no_change: int = 10,
        random_state: int = 1,
    ) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y) -> "AdalineGD":
        """Fit the model with batch gradient descent.

        Parameters
        ----------
        X:
            Training matrix, shape ``(n_samples, n_features)``. Dense ndarray
            or any SciPy sparse matrix.
        y:
            Binary targets in ``{0, 1}``, shape ``(n_samples,)``.

        Returns
        -------
        self

        Raises
        ------
        AdalineDivergedError
            If the loss becomes non-finite, i.e. ``lr`` is too large.
        ValueError
            If ``X`` and ``y`` disagree on the number of samples.
        """
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}"
            )

        n_samples, n_features = X.shape
        # Recorded for scikit-learn's classifier API; the labels are always
        # {0, 1} by the convention fixed in fakereview.data.
        self.classes_ = np.unique(y)
        self.n_features_in_ = n_features
        rgen = np.random.RandomState(self.random_state)

        # Small random weights break symmetry without starting far from the
        # origin, where the MSE surface is well conditioned.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = 0.0
        self.losses_ = []

        best_loss = np.inf
        first_loss: float | None = None
        stagnant_epochs = 0

        for epoch in range(self.n_iter):
            # Forward pass. activation() is the identity, but we call it
            # explicitly so the Adaline structure stays legible: the gradient
            # is taken against the *continuous* output, not the thresholded
            # class label. That is the whole difference from the perceptron.
            errors = y - self.activation(self.net_input(X))

            # Gradient descent step. The 2/n factor is the derivative of the
            # mean squared error; folding it in (rather than absorbing it into
            # lr) keeps `lr` comparable to the textbook formulation.
            self.w_ += self.lr * 2.0 * self._rmatvec(X, errors) / n_samples
            self.b_ += self.lr * 2.0 * errors.mean()

            loss = float(np.mean(errors**2))
            self.losses_.append(loss)

            if first_loss is None:
                first_loss = loss
            reference = min(best_loss, first_loss)

            # Divergence guard. Checking only for inf/NaN is not enough: the
            # loss can reach 1e34 -- utterly diverged -- while still being a
            # finite float. The loss on this convex objective must trend down,
            # so growing far past the best value seen means lr is too large.
            if not np.isfinite(loss) or loss > reference * self._DIVERGENCE_FACTOR:
                raise AdalineDivergedError(
                    f"Loss reached {loss:.4g} at epoch {epoch} with "
                    f"lr={self.lr} (best so far {reference:.4g}). Adaline "
                    "diverges when the learning rate is too high -- try a "
                    "smaller lr, or scale your features."
                )

            # Early stopping once the loss stops meaningfully improving.
            # Note the explicit `0 <= improvement`: a *worsening* loss also
            # satisfies `best_loss - loss < tol`, so without this the stagnation
            # counter would fire while the model was actively diverging.
            if self.tol is not None:
                improvement = best_loss - loss
                if 0.0 <= improvement < self.tol:
                    stagnant_epochs += 1
                    if stagnant_epochs >= self.n_iter_no_change:
                        self.n_iter_ = epoch + 1
                        return self
                else:
                    stagnant_epochs = 0
                best_loss = min(best_loss, loss)

        self.n_iter_ = len(self.losses_)
        return self

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def net_input(self, X):
        """Compute the linear combination ``Xw + b``.

        Works for both dense and sparse ``X``; ``X @ self.w_`` dispatches to
        the sparse matmul automatically.
        """
        return X @ self.w_ + self.b_

    @staticmethod
    def activation(z):
        """Adaline's activation: the identity function.

        Kept as an explicit method rather than inlined because it is the
        conceptual heart of the algorithm -- swapping this for a sigmoid (and
        MSE for log-loss) turns Adaline into logistic regression.
        """
        return z

    def decision_function(self, X):
        """Return the raw continuous activation, before thresholding.

        This is what the web app displays as a confidence score, and what ROC
        AUC is computed from. It is *not* a probability -- it is unbounded and
        can fall outside ``[0, 1]``.
        """
        return self.activation(self.net_input(X))

    def predict(self, X):
        """Return predicted class labels in ``{0, 1}``.

        Thresholds at 0.5, the midpoint between the two label values.
        """
        return np.where(self.decision_function(X) >= 0.5, 1, 0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _rmatvec(X, v):
        """Compute ``X.T @ v`` and always return a flat ndarray.

        Sparse ``X.T @ v`` can come back as ``np.matrix``, which would silently
        turn ``self.w_`` into a 2-D matrix and break broadcasting downstream.
        """
        out = X.T @ v
        if sp.issparse(out):
            out = out.toarray()
        return np.asarray(out).ravel()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AdalineGD(lr={self.lr}, n_iter={self.n_iter}, "
            f"tol={self.tol}, random_state={self.random_state})"
        )
