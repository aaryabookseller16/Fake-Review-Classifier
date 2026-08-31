"""Learning-rate sweep for the TF-IDF Adaline, reproducing the README table.

Adaline's stability is entirely governed by the learning rate, and its failure
mode with MSE loss is divergence rather than a plateau. This script measures
where that boundary sits on the real representation.

Run::

    PYTHONPATH=src python scripts/lr_sweep.py

It uses the same pipeline construction as training (TF-IDF -> MaxAbsScaler ->
Adaline), so the numbers are directly comparable with the main results table
rather than coming from a separate ad-hoc setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import MaxAbsScaler  # noqa: E402

from fakereview.adaline import AdalineDivergedError, AdalineGD  # noqa: E402
from fakereview.data import load_split  # noqa: E402

# (learning rate, epoch budget)
GRID = [(0.1, 6000), (0.3, 6000), (0.5, 6000), (0.7, 6000), (1.0, 6000)]


def main() -> int:
    data = load_split()
    print(f"{data.n_train:,} train / {data.n_test:,} held out\n")
    print(f"{'lr':>5} {'epochs':>7} {'accuracy':>10} {'ROC AUC':>9}  outcome")
    print("-" * 52)

    for lr, n_iter in GRID:
        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                ("scale", MaxAbsScaler()),
                (
                    "clf",
                    AdalineGD(
                        lr=lr, n_iter=n_iter, tol=1e-9, n_iter_no_change=200
                    ),
                ),
            ]
        )
        try:
            pipe.fit(data.text_train, data.y_train)
        except AdalineDivergedError as exc:
            print(f"{lr:>5} {'--':>7} {'--':>10} {'--':>9}  DIVERGED ({exc})")
            continue

        clf = pipe.named_steps["clf"]
        pred = pipe.predict(data.text_test)
        score = pipe.decision_function(data.text_test)
        print(
            f"{lr:>5} {clf.n_iter_:>7} "
            f"{accuracy_score(data.y_test, pred):>10.4f} "
            f"{roc_auc_score(data.y_test, score):>9.4f}  "
            f"converged (final MSE {clf.losses_[-1]:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
