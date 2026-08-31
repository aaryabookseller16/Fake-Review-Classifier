"""Training entry point.

Trains one or more of the registered pipelines, scores them on a held-out
split, persists the fitted models, and writes ``reports/metrics.json``.

Usage::

    python -m fakereview.train                      # train everything
    python -m fakereview.train --only tfidf-adaline # train one pipeline
    python -m fakereview.train --list               # show the registry

Run from the repository root with ``src`` on the path (``pip install -e .``
handles this, as does the ``PYTHONPATH=src`` form shown in the README).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from .data import CLASS_NAMES, load_split
from .evaluate import compute_metrics, results_markdown, roc_points, save_report
from .pipelines import PIPELINES, build_pipeline

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _decision_scores(pipeline, texts) -> np.ndarray:
    """Get continuous scores for ROC AUC, whichever estimator is at the end.

    LogisticRegression exposes ``predict_proba``; our Adaline exposes
    ``decision_function``. ROC AUC only cares about ranking, so either works --
    but they have to be extracted differently.
    """
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(texts)[:, 1]
        except (AttributeError, NotImplementedError):
            pass
    return np.asarray(pipeline.decision_function(texts)).ravel()


def train_one(key: str, data, models_dir: Path = MODELS_DIR) -> dict:
    """Fit one pipeline, score it on the test split, and persist it."""
    spec = PIPELINES[key]
    print(f"\n── {spec.label} ".ljust(70, "─"))
    print(f"   {spec.summary}")

    pipeline = build_pipeline(key)

    start = time.perf_counter()
    pipeline.fit(data.text_train, data.y_train)
    elapsed = time.perf_counter() - start

    y_pred = pipeline.predict(data.text_test)
    y_score = _decision_scores(pipeline, data.text_test)

    metrics = compute_metrics(
        pipeline_key=key,
        y_true=data.y_test,
        y_pred=y_pred,
        y_score=y_score,
        n_train=data.n_train,
        train_seconds=elapsed,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = models_dir / f"{key}.joblib"
    joblib.dump(pipeline, artifact, compress=3)

    print(
        f"   accuracy {metrics.accuracy:.4f} │ f1 {metrics.f1:.4f} │ "
        f"roc_auc {metrics.roc_auc:.4f} │ {elapsed:.1f}s"
    )
    print(
        f"   confusion: TN={metrics.true_negatives} FP={metrics.false_positives} "
        f"FN={metrics.false_negatives} TP={metrics.true_positives}"
    )
    print(f"   saved → {artifact.relative_to(models_dir.parents[0])}")

    # The training curve only exists for the from-scratch model; it is what
    # the web app plots to show gradient descent actually converging.
    clf = pipeline.named_steps.get("clf")
    losses = [float(v) for v in getattr(clf, "losses_", [])]
    if losses and len(losses) > 400:
        idx = np.linspace(0, len(losses) - 1, 400).astype(int)
        losses = [losses[i] for i in idx]

    return {
        "metrics": metrics,
        "losses": losses,
        "roc": roc_points(data.y_test, y_score),
        # sklearn reports n_iter_ as an array (one entry per class/target),
        # our Adaline reports a plain int -- normalise both to a scalar.
        "epochs_run": int(np.ravel(getattr(clf, "n_iter_", 0))[0]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and benchmark the fake-review classifiers."
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(PIPELINES),
        help="Train only this pipeline (repeatable). Default: all of them.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Held-out fraction."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Split seed, for reproducibility."
    )
    parser.add_argument(
        "--list", action="store_true", help="List available pipelines and exit."
    )
    args = parser.parse_args(argv)

    if args.list:
        for key, spec in PIPELINES.items():
            print(f"{key:24s} {spec.label}")
        return 0

    keys = args.only or list(PIPELINES)

    print("Loading dataset…")
    data = load_split(test_size=args.test_size, random_state=args.seed)
    print(
        f"  {data.n_train:,} train / {data.n_test:,} test "
        f"(positive class = {CLASS_NAMES[1]})"
    )

    results = [train_one(key, data) for key in keys]
    metrics = [r["metrics"] for r in results]

    payload = {
        "dataset": {
            "n_train": data.n_train,
            "n_test": data.n_test,
            "test_size": args.test_size,
            "seed": args.seed,
            "class_names": list(CLASS_NAMES),
        },
        "results": {
            r["metrics"].pipeline: {
                **r["metrics"].to_dict(),
                "losses": r["losses"],
                "roc": r["roc"],
                "epochs_run": r["epochs_run"],
            }
            for r in results
        },
    }
    path = save_report(payload)

    print("\n" + "=" * 70)
    print(results_markdown(metrics, PIPELINES))
    print("=" * 70)
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
