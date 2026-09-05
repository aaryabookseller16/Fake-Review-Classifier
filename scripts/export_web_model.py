"""Export the trained TF-IDF Adaline pipeline for exact browser inference.

The browser reproduces scikit-learn's word 1–2 gram TF-IDF transform, L2
normalisation, MaxAbs scaling, and the from-scratch Adaline decision function.
Only inference data is exported; no training records are shipped to visitors.

Run from the repository root:

    PYTHONPATH=src python scripts/export_web_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tfidf-adaline.joblib"
OUTPUT_PATH = ROOT / "web" / "model.json"


def rounded(values: np.ndarray, digits: int = 10) -> list[float]:
    """Return compact JSON-safe floats while retaining prediction parity."""
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def main() -> None:
    pipeline = joblib.load(MODEL_PATH)
    vectorizer = pipeline.named_steps["tfidf"]
    scaler = pipeline.named_steps["scale"]
    classifier = pipeline.named_steps["clf"]

    terms = vectorizer.get_feature_names_out()
    effective_weights = classifier.w_ / scaler.scale_

    payload = {
        "schema": 1,
        "name": "TF-IDF + Adaline",
        "description": "Word 1–2 gram TF-IDF with Adaline trained from scratch in NumPy.",
        "threshold": 0.5,
        "bias": round(float(classifier.b_), 10),
        "accuracy": 0.9488067268,
        "rocAuc": 0.9885234716,
        "trainingReviews": 32345,
        "heldOutReviews": 8087,
        "terms": terms.tolist(),
        "idf": rounded(vectorizer.idf_),
        "weights": rounded(effective_weights),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"Exported {len(terms):,} terms to {OUTPUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
