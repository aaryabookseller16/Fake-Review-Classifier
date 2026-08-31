"""Dataset loading and splitting.

The dataset is the Salminen et al. fake-reviews corpus: 40,432 Amazon-style
product reviews, exactly balanced between two classes.

``CG``
    Computer-generated (fake) -- produced by a GPT-2 model fine-tuned on real
    reviews.
``OR``
    Original review (genuine) -- written by a human.

Label convention used throughout this project::

    1 = CG = fake
    0 = OR = genuine

That mapping is defined once, here, and imported everywhere else. The original
code defined it in the training script and then *inverted* it in the web app,
so every prediction the demo displayed was backwards. Keeping it in one place
is the fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "POSITIVE_LABEL",
    "NEGATIVE_LABEL",
    "CLASS_NAMES",
    "DATA_PATH",
    "Dataset",
    "load_reviews",
    "load_split",
]

#: Label 1 -- computer-generated, i.e. the class we are trying to detect.
POSITIVE_LABEL = "CG"
#: Label 0 -- an original, human-written review.
NEGATIVE_LABEL = "OR"
#: Index-aligned display names: CLASS_NAMES[0] == "Genuine", [1] == "Fake".
CLASS_NAMES: tuple[str, str] = ("Genuine", "Fake")

#: Default dataset location, resolved relative to the repository root so the
#: path works no matter which directory the process was launched from. The
#: original script hard-coded an absolute path to a folder that no longer
#: exists on any machine.
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "fake_reviews_dataset.csv"


@dataclass(frozen=True)
class Dataset:
    """A train/test split of raw review text and binary labels."""

    text_train: np.ndarray
    text_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_test(self) -> int:
        return len(self.y_test)


def load_reviews(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw review CSV.

    Parameters
    ----------
    path:
        Override the default dataset location.

    Returns
    -------
    pandas.DataFrame
        Columns ``category``, ``rating``, ``label``, ``text_``, plus a derived
        integer ``target`` column following the 1=fake convention.

    Raises
    ------
    FileNotFoundError
        With a message pointing at the download instructions in the README,
        since the CSV is the one input a fresh clone might be missing.
    """
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See the 'Data' section of the "
            "README for where to download fake_reviews_dataset.csv."
        )

    df = pd.read_csv(path)

    missing = {"label", "text_"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {sorted(missing)}")

    # Coerce to plain Python strings. Pandas may back this column with a
    # PyArrow string array, which does not support the fancy integer indexing
    # that scikit-learn's splitter performs.
    df["text_"] = df["text_"].astype(str)

    unknown = set(df["label"].unique()) - {POSITIVE_LABEL, NEGATIVE_LABEL}
    if unknown:
        raise ValueError(f"Unexpected label value(s): {sorted(unknown)}")

    df["target"] = (df["label"] == POSITIVE_LABEL).astype(int)
    return df


def load_split(
    path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dataset:
    """Load the dataset and produce a stratified train/test split.

    Stratification keeps the 50/50 class balance in both halves, so accuracy
    stays directly interpretable against a 50% chance baseline.

    Parameters
    ----------
    path:
        Override the default dataset location.
    test_size:
        Fraction held out for evaluation.
    random_state:
        Seed, fixed by default so reported metrics are reproducible.
    """
    df = load_reviews(path)
    texts = np.array(df["text_"].tolist(), dtype=object)
    y = df["target"].to_numpy()

    text_train, text_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # preserve the 50/50 balance in both splits
    )
    return Dataset(text_train, text_test, y_train, y_test)
