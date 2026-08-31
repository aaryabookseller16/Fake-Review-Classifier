"""The five model configurations this project benchmarks.

The point of the project is a controlled comparison: a 2x2 of representation
against learning algorithm, plus one extra representation tier in the middle.
Each pipeline changes exactly one thing relative to its neighbour, so the
resulting accuracy differences are attributable.

============================  ===================  ==========================
pipeline                      representation       classifier
============================  ===================  ==========================
``handcrafted-adaline``       3 surface features   Adaline (from scratch)
``handcrafted-logreg``        3 surface features   sklearn LogisticRegression
``extended-adaline``          10 surface features  Adaline (from scratch)
``tfidf-adaline``             TF-IDF 1-2 grams     Adaline (from scratch)
``tfidf-logreg``              TF-IDF 1-2 grams     sklearn LogisticRegression
============================  ===================  ==========================

Holding the classifier fixed and varying the representation isolates the
effect of the features. Holding the representation fixed and varying the
classifier (rows 1 vs 2, and rows 4 vs 5) isolates the effect of the optimiser
and loss function -- which is what validates the from-scratch implementation.

Everything is wrapped in a scikit-learn ``Pipeline`` so that fitting,
prediction and persistence are a single object, and so the vectoriser's
vocabulary can never be fit on the test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from .adaline import AdalineGD
from .features import extract, feature_names

__all__ = ["HandcraftedFeatures", "PIPELINES", "build_pipeline", "describe"]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Adapt the pure feature functions in :mod:`.features` to sklearn.

    Stateless: ``fit`` records nothing but the feature names (needed for the
    interpretability table in the web app). Being a proper transformer means
    it composes inside a ``Pipeline`` and gets persisted with the model, so
    serving cannot drift from training.
    """

    def __init__(self, feature_set: str = "basic") -> None:
        self.feature_set = feature_set

    def fit(self, X, y=None) -> HandcraftedFeatures:
        self.feature_names_ = feature_names(self.feature_set)
        return self

    def transform(self, X) -> np.ndarray:
        return extract(list(X), self.feature_set)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(feature_names(self.feature_set), dtype=object)


@dataclass(frozen=True)
class PipelineSpec:
    """Metadata plus a factory for one benchmark configuration."""

    key: str
    label: str
    representation: str
    classifier: str
    summary: str
    factory: Any = field(repr=False)

    def build(self) -> Pipeline:
        return self.factory()


def _tfidf() -> TfidfVectorizer:
    """Shared TF-IDF settings, so the two TF-IDF rows differ only in optimiser.

    ``sublinear_tf`` dampens repeated-token counts, ``min_df=2`` drops
    once-only terms (mostly typos), and 1-2 grams capture short phrases such as
    ``"highly recommend"`` that unigrams alone would miss.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def _handcrafted_adaline(feature_set: str) -> Pipeline:
    """Surface features -> standardise -> Adaline.

    Standardising matters more than it looks: the raw features live on wildly
    different scales (word counts in the hundreds, ratios in [0, 1]). Without
    it the MSE surface is badly conditioned and gradient descent either crawls
    or diverges.
    """
    return Pipeline(
        [
            ("features", HandcraftedFeatures(feature_set=feature_set)),
            ("scale", StandardScaler()),
            (
                "clf",
                AdalineGD(lr=0.1, n_iter=3000, tol=1e-11, n_iter_no_change=100),
            ),
        ]
    )


def _tfidf_adaline() -> Pipeline:
    """TF-IDF -> Adaline (from scratch).

    ``MaxAbsScaler`` is used rather than ``StandardScaler`` because it is the
    only one of the two that preserves sparsity -- centring a 139k-column
    matrix would densify it to roughly 35 GB.

    The learning rate comes from the sweep in ``scripts/lr_sweep.py``: lr=0.1
    is both the most accurate setting *and* the most stable one. Higher rates
    trade accuracy for nothing -- 0.5 converges to a slightly worse optimum,
    and 0.7 and above diverge outright.
    """
    return Pipeline(
        [
            ("tfidf", _tfidf()),
            ("scale", MaxAbsScaler()),
            (
                "clf",
                AdalineGD(lr=0.1, n_iter=6000, tol=1e-9, n_iter_no_change=200),
            ),
        ]
    )


def _handcrafted_logreg(feature_set: str) -> Pipeline:
    """Surface features -> standardise -> LogisticRegression.

    The fourth cell of the 2x2. Holding the representation fixed and swapping
    only the loss function isolates how much of Adaline's weakness on these
    features is the *features* and how much is MSE's sensitivity to the
    heavy-tailed count columns.
    """
    return Pipeline(
        [
            ("features", HandcraftedFeatures(feature_set=feature_set)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def _tfidf_logreg() -> Pipeline:
    """TF-IDF -> scikit-learn LogisticRegression.

    The reference ceiling. Any gap between this and ``tfidf-adaline`` is
    attributable to the optimiser and loss function, not the features.
    """
    return Pipeline(
        [
            ("tfidf", _tfidf()),
            ("clf", LogisticRegression(max_iter=2000, C=4.0)),
        ]
    )


#: Registry, ordered weakest to strongest -- this is the order the README
#: results table and the web app's model picker use.
PIPELINES: dict[str, PipelineSpec] = {
    "handcrafted-adaline": PipelineSpec(
        key="handcrafted-adaline",
        label="Adaline + 3 hand-crafted features",
        representation="3 surface statistics",
        classifier="Adaline (from scratch)",
        summary=(
            "The original project. Length, hype-word count and ALL-CAPS count "
            "carry almost no class signal on this dataset."
        ),
        factory=lambda: _handcrafted_adaline("basic"),
    ),
    "handcrafted-logreg": PipelineSpec(
        key="handcrafted-logreg",
        label="LogisticRegression + 3 hand-crafted features",
        representation="3 surface statistics",
        classifier="scikit-learn LogisticRegression",
        summary=(
            "Same three weak features, log-loss instead of MSE. Isolates how "
            "much of Adaline's deficit here is the loss function."
        ),
        factory=lambda: _handcrafted_logreg("basic"),
    ),
    "extended-adaline": PipelineSpec(
        key="extended-adaline",
        label="Adaline + 10 hand-crafted features",
        representation="10 surface statistics",
        classifier="Adaline (from scratch)",
        summary=(
            "Adds punctuation, lexical diversity and pronoun ratios. Better, "
            "but shows that hand-engineering surface stats plateaus early."
        ),
        factory=lambda: _handcrafted_adaline("extended"),
    ),
    "tfidf-adaline": PipelineSpec(
        key="tfidf-adaline",
        label="Adaline + TF-IDF",
        representation="TF-IDF, word 1-2 grams",
        classifier="Adaline (from scratch)",
        summary=(
            "Same from-scratch optimiser, richer representation. This is the "
            "jump that actually matters."
        ),
        factory=_tfidf_adaline,
    ),
    "tfidf-logreg": PipelineSpec(
        key="tfidf-logreg",
        label="LogisticRegression + TF-IDF",
        representation="TF-IDF, word 1-2 grams",
        classifier="scikit-learn LogisticRegression",
        summary=(
            "Reference ceiling. Confirms the from-scratch optimiser is not "
            "leaving meaningful accuracy on the table."
        ),
        factory=_tfidf_logreg,
    ),
}

#: The pipeline the web app loads by default.
DEFAULT_PIPELINE = "tfidf-adaline"


def build_pipeline(key: str) -> Pipeline:
    """Instantiate a fresh, unfitted pipeline by registry key."""
    if key not in PIPELINES:
        raise KeyError(
            f"Unknown pipeline {key!r}. Available: {sorted(PIPELINES)}"
        )
    return PIPELINES[key].build()


def describe(key: str) -> PipelineSpec:
    """Return the metadata record for a pipeline key."""
    if key not in PIPELINES:
        raise KeyError(
            f"Unknown pipeline {key!r}. Available: {sorted(PIPELINES)}"
        )
    return PIPELINES[key]
