"""Tests for pipeline wiring and the label convention.

The most important test in this file is
:func:`test_label_convention_is_consistent`. The original project mapped
CG (fake) to 1 during training and then displayed 1 as "Genuine" in the web
app, so every prediction the demo showed was inverted. That class of bug is
invisible to accuracy metrics -- it only shows up at the presentation layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from fakereview.data import CLASS_NAMES, NEGATIVE_LABEL, POSITIVE_LABEL
from fakereview.pipelines import PIPELINES, build_pipeline, describe

# A tiny, deliberately easy corpus: hype-speak vs. plain first-person prose.
FAKE = [
    "This product is amazing and the best purchase guaranteed to delight!",
    "Amazing quality, best value, buy now for a limited time guaranteed!",
    "The best product ever, amazing and free shipping guaranteed!",
] * 6
REAL = [
    "I bought this last March and the handle cracked after a few weeks.",
    "Works fine for me though the colour is a bit off from the photo.",
    "My kid uses it daily. Bit noisy but we are happy enough with it.",
] * 6


@pytest.fixture
def toy():
    texts = np.array(FAKE + REAL, dtype=object)
    y = np.array([1] * len(FAKE) + [0] * len(REAL))
    return texts, y


def test_label_convention_is_consistent():
    """1 must mean CG/fake, and CLASS_NAMES must be index-aligned to that."""
    assert POSITIVE_LABEL == "CG"
    assert NEGATIVE_LABEL == "OR"
    assert CLASS_NAMES[0] == "Genuine"  # label 0 == OR == human-written
    assert CLASS_NAMES[1] == "Fake"     # label 1 == CG == computer-generated


@pytest.mark.parametrize("key", sorted(PIPELINES))
def test_every_pipeline_fits_and_predicts(key, toy):
    texts, y = toy
    pipe = build_pipeline(key)
    pipe.fit(texts, y)
    preds = pipe.predict(texts)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.parametrize("key", sorted(PIPELINES))
def test_every_pipeline_exposes_continuous_scores(key, toy):
    """Needed for ROC AUC and for the app's confidence display."""
    texts, y = toy
    pipe = build_pipeline(key).fit(texts, y)
    scores = pipe.decision_function(texts)
    assert np.asarray(scores).shape[0] == len(y)


def test_registry_metadata_is_complete():
    for key, spec in PIPELINES.items():
        assert spec.key == key
        assert spec.label and spec.representation and spec.classifier
        assert spec.summary.endswith(".")


def test_unknown_pipeline_raises():
    with pytest.raises(KeyError, match="Unknown pipeline"):
        build_pipeline("nope")
    with pytest.raises(KeyError, match="Unknown pipeline"):
        describe("nope")


def test_pipeline_accepts_a_single_string_at_serve_time(toy):
    """The web app predicts on a one-element list; that must not break."""
    texts, y = toy
    pipe = build_pipeline("handcrafted-adaline").fit(texts, y)
    out = pipe.predict(["I love this thing, works great."])
    assert out.shape == (1,)
