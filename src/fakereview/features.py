"""Hand-crafted (interpretable) features extracted from raw review text.

Two feature sets live here:

``BASIC_FEATURES``
    The three features the original project used -- review length, count of
    "suspicious" marketing words, and count of ALL-CAPS words. They are kept
    because the README benchmarks them honestly: they reach only ~59% accuracy
    on a balanced task, and showing *why* a feature set fails is part of the
    result.

``EXTENDED_FEATURES``
    A wider set of surface statistics (punctuation, lexical diversity,
    pronouns, digits). Still fully interpretable, and a fairer test of how far
    hand-engineering alone can go before you need distributional features.

Every extractor is a pure function ``Sequence[str] -> np.ndarray`` of shape
``(n_samples,)``, so they compose into a matrix without any DataFrame
mutation. The original implementation mutated the caller's DataFrame in place
and returned it, which made the feature order depend on call order -- a real
source of train/serve skew.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

__all__ = [
    "SUSPICIOUS_WORDS",
    "BASIC_FEATURES",
    "EXTENDED_FEATURES",
    "FEATURE_SETS",
    "extract",
    "feature_names",
]

# Marketing / hype vocabulary often over-represented in generated promotional
# text. Deliberately short and readable -- this is an interpretable baseline,
# not a learned lexicon.
SUSPICIOUS_WORDS: tuple[str, ...] = (
    "free",
    "amazing",
    "best",
    "buy now",
    "limited",
    "guaranteed",
)

# Personal pronouns: genuine reviews tend to narrate first-hand experience.
_PRONOUNS = re.compile(r"\b(i|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z']+")


# ----------------------------------------------------------------------
# Individual extractors
# ----------------------------------------------------------------------
def word_count(texts: Sequence[str]) -> np.ndarray:
    """Number of whitespace-separated tokens in each review."""
    return np.array([len(t.split()) for t in texts], dtype=float)


def char_count(texts: Sequence[str]) -> np.ndarray:
    """Total character count of each review."""
    return np.array([len(t) for t in texts], dtype=float)


def suspicious_word_count(
    texts: Sequence[str], vocabulary: Sequence[str] = SUSPICIOUS_WORDS
) -> np.ndarray:
    """Occurrences of hype/marketing phrases, case-insensitive.

    Counts substring occurrences (not whole tokens) so multi-word phrases like
    ``"buy now"`` are matched.
    """
    return np.array(
        [sum(t.lower().count(w) for w in vocabulary) for t in texts],
        dtype=float,
    )


def caps_word_count(texts: Sequence[str]) -> np.ndarray:
    """Number of fully upper-case words (shouting), e.g. ``GREAT``.

    Single characters are excluded so a standalone ``I`` or ``A`` does not
    inflate the count.
    """
    return np.array(
        [len([w for w in t.split() if w.isupper() and len(w) > 1]) for t in texts],
        dtype=float,
    )


def exclamation_count(texts: Sequence[str]) -> np.ndarray:
    """Number of exclamation marks -- a cheap proxy for forced enthusiasm."""
    return np.array([t.count("!") for t in texts], dtype=float)


def punctuation_ratio(texts: Sequence[str]) -> np.ndarray:
    """Fraction of characters that are punctuation.

    Normalised by length, so it is not just a restatement of ``char_count``.
    """
    out = []
    for t in texts:
        if not t:
            out.append(0.0)
            continue
        n_punct = sum(1 for c in t if not c.isalnum() and not c.isspace())
        out.append(n_punct / len(t))
    return np.array(out, dtype=float)


def mean_word_length(texts: Sequence[str]) -> np.ndarray:
    """Average token length -- a rough register/formality signal."""
    out = []
    for t in texts:
        words = _WORD.findall(t)
        out.append(float(np.mean([len(w) for w in words])) if words else 0.0)
    return np.array(out, dtype=float)


def lexical_diversity(texts: Sequence[str]) -> np.ndarray:
    """Type-token ratio: unique words / total words.

    Generated text often recycles vocabulary within a short span, which shows
    up as lower diversity.
    """
    out = []
    for t in texts:
        words = [w.lower() for w in _WORD.findall(t)]
        out.append(len(set(words)) / len(words) if words else 0.0)
    return np.array(out, dtype=float)


def pronoun_ratio(texts: Sequence[str]) -> np.ndarray:
    """Fraction of tokens that are first-person pronouns.

    Real reviewers describe what happened to *them*; promotional copy tends to
    describe the product instead.
    """
    out = []
    for t in texts:
        words = _WORD.findall(t)
        out.append(len(_PRONOUNS.findall(t)) / len(words) if words else 0.0)
    return np.array(out, dtype=float)


def digit_ratio(texts: Sequence[str]) -> np.ndarray:
    """Fraction of characters that are digits (model numbers, sizes, prices)."""
    out = []
    for t in texts:
        out.append(sum(c.isdigit() for c in t) / len(t) if t else 0.0)
    return np.array(out, dtype=float)


# ----------------------------------------------------------------------
# Feature-set registry
# ----------------------------------------------------------------------
Extractor = Callable[[Sequence[str]], np.ndarray]

#: The original three features, preserved for the honest baseline comparison.
BASIC_FEATURES: dict[str, Extractor] = {
    "word_count": word_count,
    "suspicious_word_count": suspicious_word_count,
    "caps_word_count": caps_word_count,
}

#: A wider interpretable set -- still no learned vocabulary.
EXTENDED_FEATURES: dict[str, Extractor] = {
    **BASIC_FEATURES,
    "char_count": char_count,
    "exclamation_count": exclamation_count,
    "punctuation_ratio": punctuation_ratio,
    "mean_word_length": mean_word_length,
    "lexical_diversity": lexical_diversity,
    "pronoun_ratio": pronoun_ratio,
    "digit_ratio": digit_ratio,
}

FEATURE_SETS: dict[str, dict[str, Extractor]] = {
    "basic": BASIC_FEATURES,
    "extended": EXTENDED_FEATURES,
}


def feature_names(feature_set: str = "basic") -> list[str]:
    """Return the ordered feature names for a registered feature set.

    Order is the dict insertion order and is therefore stable, which is what
    keeps training and serving aligned.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(
            f"Unknown feature set {feature_set!r}. "
            f"Available: {sorted(FEATURE_SETS)}"
        )
    return list(FEATURE_SETS[feature_set])


def extract(texts: Sequence[str], feature_set: str = "basic") -> np.ndarray:
    """Build the feature matrix for ``texts``.

    Parameters
    ----------
    texts:
        Raw review strings.
    feature_set:
        Either ``"basic"`` or ``"extended"``.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), n_features)``, column order matching
        :func:`feature_names`.

    Examples
    --------
    >>> extract(["GREAT product, buy now!"], "basic").shape
    (1, 3)
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(
            f"Unknown feature set {feature_set!r}. "
            f"Available: {sorted(FEATURE_SETS)}"
        )
    texts = [t if isinstance(t, str) else "" for t in texts]
    extractors = FEATURE_SETS[feature_set]
    return np.column_stack([fn(texts) for fn in extractors.values()])
