# Fake Review Detector

[![tests](https://github.com/aaryabookseller16/Fake-Review-Classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/aaryabookseller16/Fake-Review-Classifier/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Detecting computer-generated product reviews with an **Adaline classifier written from
scratch in NumPy**, benchmarked against scikit-learn on 40,432 Amazon reviews.

The from-scratch implementation reaches **94.88% accuracy / 0.9885 ROC AUC** — matching
scikit-learn's `LogisticRegression` on identical features (94.47% / 0.9874). That is the
point of the project: it demonstrates the hand-written gradient descent is correct, not
merely plausible.

```bash
git clone https://github.com/aaryabookseller16/Fake-Review-Classifier.git
cd Fake-Review-Classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m fakereview.train    # ~55s, writes models/ + reports/metrics.json
streamlit run app/streamlit_app.py
```

---

## Results

All numbers are from a single stratified 80/20 split (32,345 train / 8,087 held-out,
`random_state=42`), measured on the held-out half only. The positive class is
**CG = computer-generated = fake**.

| Model | Representation | Accuracy | Precision | Recall | F1 | ROC AUC | Train |
|---|---|---:|---:|---:|---:|---:|---:|
| Adaline + 3 hand-crafted features | 3 surface statistics | 0.5313 | 0.5227 | 0.7190 | 0.6054 | 0.5490 | 0.3s |
| LogisticRegression + 3 hand-crafted features | 3 surface statistics | 0.5384 | 0.5276 | 0.7321 | 0.6133 | 0.5535 | 0.3s |
| Adaline + 10 hand-crafted features | 10 surface statistics | 0.7557 | 0.7507 | 0.7655 | 0.7580 | 0.8382 | 3.7s |
| **Adaline + TF-IDF** (from scratch) | TF-IDF, word 1–2 grams | **0.9488** | 0.9632 | 0.9332 | 0.9480 | **0.9885** | 41.5s |
| LogisticRegression + TF-IDF | TF-IDF, word 1–2 grams | 0.9447 | 0.9584 | 0.9298 | 0.9439 | 0.9874 | 2.9s |

The dataset is exactly balanced (20,216 per class), so **0.50 accuracy is the chance
baseline** and every number above is directly interpretable against it.

### What the comparison isolates

This is a 2×2, not a leaderboard. Two representations (3 surface statistics vs TF-IDF)
crossed with two learning algorithms (from-scratch Adaline with MSE loss vs scikit-learn
logistic regression with log-loss), plus one extra representation tier in the middle.

|  | Adaline (MSE, from scratch) | LogisticRegression (log-loss) | **gap** |
|---|---:|---:|---:|
| **3 surface statistics** | 0.5313 | 0.5384 | 0.71 pp |
| **TF-IDF 1–2 grams** | 0.9488 | 0.9447 | 0.41 pp |
| **gap** | **41.75 pp** | **40.63 pp** | |

**Representation is worth ~41 points. The learning algorithm is worth less than one.**
That holds in both rows: swapping the optimiser and the loss function never moves accuracy
by more than 0.71pp, while changing the representation moves it by more than 40. On this
problem feature engineering dominated algorithm choice by roughly **fifty to one**.

The secondary result is that the from-scratch implementation is *correct*: hand-written
batch gradient descent matches a mature L-BFGS solver on identical features, in fact
edging it by 0.41pp. I would not read anything into the sign of that gap — it is one
split, with no cross-validation, and the two models optimise different losses. The
defensible claim is that they are equivalent, not that the hand-written one is better.
It is considerably slower, though: 41.5s against 2.9s.

### The hand-crafted features do not work, and that is the finding

Review length, hype-word count, and ALL-CAPS count reach **53.1% accuracy — 3 points above
a coin flip.** They fail because the corpus's generated reviews were produced by a GPT-2
model fine-tuned on real reviews, so they match the surface statistics of genuine reviews
closely. The generated text does not shout, and it is not unusually long:

> **CG (fake):** "Works great. Exact replace for $60, with the instructions included. If
> you have a larger computer, you'll want to get a solid replacement."
>
> **OR (genuine):** "They may not be picture perfect machine-finished but the metal is good
> they hold up and do their job and were much better than I expected for the amazingly low
> price."

What *does* separate them is vocabulary and phrasing — which is exactly what TF-IDF
captures and what a length counter cannot. The generated reviews lean on templated
constructions ("If you have a larger X, you'll want to get a larger Y") that show up as
n-gram statistics, not as surface anomalies.

Expanding to ten features (punctuation ratio, lexical diversity, first-person pronoun
ratio, digit ratio) recovers a substantial amount — 75.6% — but still leaves ~19 points on
the table relative to TF-IDF.

### Precision/recall trade-off

Both TF-IDF models are markedly more conservative than balanced — they under-predict the
fake class, trading recall for precision:

| | False positives | False negatives | Precision | Recall |
|---|---:|---:|---:|---:|
| Adaline + TF-IDF | 144 | 270 | 0.9632 | 0.9332 |
| LogisticRegression + TF-IDF | 163 | 284 | 0.9584 | 0.9298 |

Adaline is slightly better on both counts here, but the shapes are the same: roughly twice
as many fakes slip through as genuine reviews get wrongly flagged.

That default happens to suit review moderation, where a false positive — publicly flagging
a real customer's review as fake — is the more damaging error. It is a property of where
the 0.5 threshold happens to fall, though, not a designed choice. A deployment with an
explicit cost ratio should tune the threshold against it, which is what the
[calibration work](#what-i-would-do-next) below is for.

---

## The model

Adaline (ADAptive LInear NEuron, Widrow–Hoff 1960) is the perceptron's immediate successor.
The one idea that distinguishes it: compute the gradient against a **continuous linear
activation** rather than the thresholded class label. That makes the loss differentiable
and convex, so gradient descent converges to a global optimum.

The model is linear, `z = Xw + b`, minimising mean squared error:

$$L(w, b) = \frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - (x_i^\top w + b)\bigr)^2$$

$$\frac{\partial L}{\partial w} = -\frac{2}{n}X^\top(y - z), \qquad
  \frac{\partial L}{\partial b} = -\frac{2}{n}\sum_i (y_i - z_i)$$

which is the entire update loop in [`src/fakereview/adaline.py`](src/fakereview/adaline.py):

```python
errors = y - self.activation(self.net_input(X))
self.w_ += self.lr * 2.0 * self._rmatvec(X, errors) / n_samples
self.b_ += self.lr * 2.0 * errors.mean()
```

No autograd, no solver library. scikit-learn's `BaseEstimator`/`ClassifierMixin` are
inherited purely for `get_params` and the estimator tags that `Pipeline` requires — none of
the optimisation is inherited.

### Three things a textbook Adaline gets wrong on real data

**1. It cannot handle sparse input.** TF-IDF here is 32,345 × 139,295. Densified at float64
that is ~35 GB. All the linear algebra is written to pass SciPy sparse matrices through
untouched, with one subtlety: `X.T @ v` on a sparse matrix can return `np.matrix`, which
silently turns the weight vector 2-D and corrupts every subsequent broadcast. `_rmatvec`
normalises it, and a test pins the behaviour.

**2. It diverges, and detecting that is harder than it looks.** With MSE the failure mode
is not a plateau — the loss grows by orders of magnitude per epoch. Measured on the real
TF-IDF representation via [`scripts/lr_sweep.py`](scripts/lr_sweep.py):

| Learning rate | Epochs | Accuracy | ROC AUC | Outcome |
|---|---:|---:|---:|---|
| 0.1 | 6,000 | **0.9488** | **0.9885** | converges, final MSE 0.0316 |
| 0.3 | 6,000 | 0.9486 | 0.9880 | converges, final MSE 0.0137 |
| 0.5 | 6,000 | 0.9429 | 0.9865 | converges, final MSE 0.0081 |
| 0.7 | — | — | — | **diverges** at epoch 36 |
| 1.0 | — | — | — | **diverges** at epoch 10 |

Note that lower learning rates reach a *better* optimum here, and that the loss the model
minimises is not monotonically related to the accuracy it achieves — lr=0.5 reaches the
lowest final MSE (0.0081) and the *worst* accuracy of the three. The 0.5 threshold on a
squared-error fit is a crude decision rule, and driving the regression loss lower does not
automatically move it in the right direction.

**The divergence check went through two revisions, and the first one was wrong.** Checking
`np.isfinite(loss)` is insufficient: at lr=0.7 the loss reached 2 × 10³⁴ — catastrophically
diverged, but still a finite float. Worse, the early-stopping rule `best_loss - loss < tol`
is *satisfied* when the loss grows, so the run terminated at epoch 201 and reported itself
converged, returning a model with 49.99% accuracy — the exact silent-garbage failure the
guard existed to prevent. It took the sweep above to expose it.

`AdalineGD` now compares against the best loss seen so far and raises
`AdalineDivergedError`, with divergence checked *before* stagnation. Two regression tests
pin both halves.

**3. Feature scaling is not optional.** Raw counts span hundreds while ratios sit in [0, 1];
without standardisation the MSE surface is badly conditioned. The dense pipelines use
`StandardScaler`; the TF-IDF pipelines use `MaxAbsScaler`, which is the only one of the two
that **preserves sparsity** — centring would densify the matrix.

### An accident worth recording

An early version of the caps-word counter did not exclude single-character tokens, so a
standalone **"I"** counted as a shouted word. That "bug" scored **58.9%** — about 5 points
*better* than the corrected version's 53.8%.

The reason is that the buggy feature was accidentally measuring first-person pronoun usage,
which genuinely does separate the classes: human reviewers narrate their own experience.
Fixing the counter removed a real signal that had nothing to do with shouting. The signal
is now captured deliberately by `pronoun_ratio` in the extended feature set, which is part
of why that tier jumps to 75.6%.

It is a small thing, but it is the cleanest illustration in this project of why you measure
features rather than reason about them.

---

## Web app

A three-page Streamlit app ([`app/streamlit_app.py`](app/streamlit_app.py)):

- **Classify a review** — paste any text, pick any of the five trained models, get a
  prediction with the raw decision score and the extracted interpretable features. Preloaded
  examples are **real held-out reviews with their true labels in the name**, so the model can
  be checked against ground truth rather than against intuition.
- **Model comparison** — the benchmark table, an accuracy bar chart, ROC curves for all five
  models, and a per-model confusion matrix. Read directly from `reports/metrics.json`, so the
  displayed numbers cannot drift from the ones training produced.
- **How it works** — the update rule, the live training curve, and the learning-rate
  divergence table.

### Deploying

The app is deploy-ready for [Streamlit Community Cloud](https://share.streamlit.io) (free):

1. Push this repository to GitHub.
2. On share.streamlit.io: **New app** → pick the repo → main file `app/streamlit_app.py`.
3. Deploy.

Two things make this work with no extra configuration:

- **The trained models are committed** (`models/*.joblib`, ~7 MB total, plus
  `reports/metrics.json`). Streamlit Community Cloud has no build step in which to train, so
  shipping the artifacts is what makes the deploy one-click. Regenerate them any time with
  `PYTHONPATH=src python -m fakereview.train`.
- **Paths are resolved from `__file__`, never the working directory**, so the app runs
  identically from the repo root, from inside `app/`, or from Streamlit Cloud's runner.

If a model is missing the app says so and prints the training command, rather than throwing
a stack trace at the visitor.

---

## Project layout

```
├── app/
│   └── streamlit_app.py       # 3-page web UI
├── data/
│   └── fake_reviews_dataset.csv
├── models/                    # trained pipelines (generated)
├── reports/
│   └── metrics.json           # all metrics + ROC points + loss curves (generated)
├── src/fakereview/
│   ├── adaline.py             # the from-scratch classifier
│   ├── features.py            # interpretable feature extractors
│   ├── data.py                # loading, splitting, the label convention
│   ├── pipelines.py           # the four benchmark configurations
│   ├── evaluate.py            # metric computation + report I/O
│   └── train.py               # CLI entry point
├── scripts/
│   └── lr_sweep.py            # reproduces the divergence table
├── tests/                     # 37 tests
└── .github/workflows/
    └── tests.yml              # CI on Python 3.11 and 3.12
```

### Commands

```bash
PYTHONPATH=src python -m fakereview.train              # train + evaluate all five
PYTHONPATH=src python scripts/lr_sweep.py              # learning-rate stability sweep
PYTHONPATH=src python -m fakereview.train --list       # show the registry
PYTHONPATH=src python -m fakereview.train --only tfidf-adaline
pytest                                                  # 37 tests
streamlit run app/streamlit_app.py
```

---

## Data

[Salminen et al. fake reviews dataset](https://osf.io/tyue9/) — 40,432 Amazon product
reviews across 10 categories, balanced 20,216 / 20,216. `OR` reviews are genuine; `CG`
reviews were produced by a GPT-2 model fine-tuned on the genuine ones.

Label convention, defined once in `fakereview.data` and imported everywhere:

```
1 = CG = fake       0 = OR = genuine
```

Keeping this in a single module is deliberate. In the original version of this project the
mapping was declared in the training script and then reversed in the web app, so every
prediction the demo displayed was inverted — a bug that no accuracy metric can catch,
because it lives entirely in the presentation layer. `tests/test_pipelines.py` now pins it.

---

## Testing

```
37 passed in 1.39s
```

Coverage is aimed at the things that actually broke, not at a line-count target:

- **Convergence** — loss decreases monotonically on a convex problem with a sane `lr`.
- **Divergence** — a too-large `lr` raises rather than returning a degenerate model,
  including the case where the loss is astronomically large but still finite, and the case
  where a diverging run would otherwise be mistaken for a stagnant one.
- **Sparse/dense equivalence** — identical weights either way, and weights stay 1-D.
- **Feature-name/matrix-width agreement** — the exact class of mismatch that made the
  original training script raise `KeyError` on its first line of real work.
- **Batch/single-row consistency** — extracting one review alone equals its row in a batch,
  which the original in-place DataFrame mutation could not guarantee.
- **Label polarity** — that `1` means fake at every layer.
- **Edge cases** — empty strings and `None` do not divide by zero.

---

## What I would do next

- **Calibration.** Adaline's decision score is an unbounded activation, not a probability.
  Platt scaling on a validation split would make the confidence number meaningful and let the
  threshold be tuned against an explicit false-positive budget.
- **Cross-validation.** Every number here is a single 80/20 split. 5-fold CV would put error
  bars on the 0.18pp Adaline-vs-sklearn gap, which is currently too small to call
  significant.
- **A transformer baseline.** A fine-tuned DistilBERT would likely clear 97% and would
  quantify what the linear models leave behind.
- **Generalisation.** All 40k reviews come from one generator (GPT-2) and one marketplace.
  Whether this transfers to reviews from a modern LLM is untested and, honestly, doubtful —
  the n-gram tics it keys on are generator-specific.

---

## License

MIT
