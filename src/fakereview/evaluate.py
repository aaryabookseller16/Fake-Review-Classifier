"""Metric extraction and report generation.

Everything reported in the README is produced here, from the held-out test
split only. Nothing is computed on training data, and nothing is hand-copied
into the README -- :func:`results_markdown` renders the table that gets pasted
in, so the documented numbers cannot drift from the measured ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = ["Metrics", "compute_metrics", "results_markdown", "save_report", "load_report"]

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


@dataclass
class Metrics:
    """Held-out test metrics for one pipeline.

    The positive class is ``1`` = fake (CG), so ``recall`` answers the question
    that actually matters for the product: *of all the fake reviews, how many
    did we catch?*
    """

    pipeline: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    #: Row-major [[TN, FP], [FN, TP]].
    confusion: list[list[int]]
    n_train: int
    n_test: int
    train_seconds: float

    @property
    def true_negatives(self) -> int:
        return self.confusion[0][0]

    @property
    def false_positives(self) -> int:
        """Genuine reviews wrongly flagged as fake -- the costly error."""
        return self.confusion[0][1]

    @property
    def false_negatives(self) -> int:
        """Fake reviews that slipped through."""
        return self.confusion[1][0]

    @property
    def true_positives(self) -> int:
        return self.confusion[1][1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    pipeline_key: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_train: int,
    train_seconds: float,
) -> Metrics:
    """Score one set of predictions against the held-out labels.

    Parameters
    ----------
    y_score:
        Continuous decision values (Adaline's raw activation, or logistic
        regression's probability). Used only for ROC AUC, which is
        threshold-independent and therefore the fairest single-number
        comparison across models calibrated differently.
    """
    return Metrics(
        pipeline=pipeline_key,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
        n_train=int(n_train),
        n_test=int(len(y_true)),
        train_seconds=float(train_seconds),
    )


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    """Return ROC curve points, subsampled to keep the JSON report small."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(fpr) > 300:
        idx = np.linspace(0, len(fpr) - 1, 300).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def results_markdown(metrics: list[Metrics], specs: dict[str, Any]) -> str:
    """Render the results table exactly as it appears in the README."""
    header = (
        "| Model | Representation | Accuracy | Precision | Recall | F1 | ROC AUC |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    rows = []
    for m in metrics:
        spec = specs[m.pipeline]
        rows.append(
            f"| {spec.label} | {spec.representation} | "
            f"{m.accuracy:.4f} | {m.precision:.4f} | {m.recall:.4f} | "
            f"{m.f1:.4f} | {m.roc_auc:.4f} |"
        )
    return "\n".join([header, *rows])


def save_report(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write the full metrics report as JSON.

    The web app reads this file to render its Metrics tab, so the numbers a
    visitor sees are the same ones the training run produced.
    """
    path = path or (REPORTS_DIR / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_report(path: Path | None = None) -> dict[str, Any] | None:
    """Load the metrics report, or ``None`` if training has not been run."""
    path = path or (REPORTS_DIR / "metrics.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
