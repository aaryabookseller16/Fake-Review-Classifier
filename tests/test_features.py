"""Tests for the hand-crafted feature extractors.

The regression these guard against: the original code built the feature matrix
by selecting DataFrame columns by name, and the training script selected a
column (`length`) that the extractor never created -- a guaranteed KeyError
that meant the pipeline had never actually run end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from fakereview import features as F


def test_feature_names_match_matrix_width():
    """Column count must equal the declared name count, for every set."""
    for name in F.FEATURE_SETS:
        matrix = F.extract(["a short review"], name)
        assert matrix.shape[1] == len(F.feature_names(name))


def test_basic_set_is_the_original_three():
    assert F.feature_names("basic") == [
        "word_count",
        "suspicious_word_count",
        "caps_word_count",
    ]


def test_extended_is_a_superset_of_basic():
    assert set(F.feature_names("basic")) <= set(F.feature_names("extended"))


def test_word_count():
    assert F.word_count(["one two three"]) == np.array([3.0])


def test_suspicious_words_are_case_insensitive_and_multiword():
    counts = F.suspicious_word_count(["FREE stuff, buy now!", "nothing here"])
    assert counts[0] == 2.0
    assert counts[1] == 0.0


def test_caps_ignores_single_letters():
    """A lone 'I' is normal English, not shouting."""
    assert F.caps_word_count(["I think this is GREAT"]) == np.array([1.0])


def test_ratios_are_bounded():
    texts = ["Perfectly ordinary review text.", "!!!???", "12345"]
    for fn in (F.punctuation_ratio, F.lexical_diversity, F.pronoun_ratio, F.digit_ratio):
        values = fn(texts)
        assert np.all((values >= 0.0) & (values <= 1.0)), fn.__name__


def test_empty_string_does_not_divide_by_zero():
    """Every extractor must survive an empty review."""
    matrix = F.extract([""], "extended")
    assert matrix.shape == (1, len(F.feature_names("extended")))
    assert np.all(np.isfinite(matrix))


def test_non_string_input_is_coerced():
    """The web app can hand us None if a field was left blank."""
    matrix = F.extract([None], "basic")  # type: ignore[list-item]
    assert np.all(matrix == 0.0)


def test_extraction_is_order_independent():
    """Extracting one review alone must equal its row in a batch.

    The original implementation mutated the caller's DataFrame, so results
    depended on which extractors had already run.
    """
    texts = ["First review here", "SECOND review, buy now!", "third"]
    batch = F.extract(texts, "extended")
    for i, t in enumerate(texts):
        np.testing.assert_allclose(F.extract([t], "extended")[0], batch[i])


def test_unknown_feature_set_raises():
    with pytest.raises(KeyError, match="Unknown feature set"):
        F.extract(["x"], "does-not-exist")
