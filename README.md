# Fake Review Detector

[![Tests](https://github.com/aaryabookseller16/Fake-Review-Classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/aaryabookseller16/Fake-Review-Classifier/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-1d211c)
![License](https://img.shields.io/badge/License-MIT-ed6547)

**Live application:** [fake-review-classifier.vercel.app](https://fake-review-classifier.vercel.app)

An end-to-end NLP study that detects computer-generated Amazon reviews with an
**Adaline classifier written from scratch in NumPy**. The project pairs a reproducible
five-model benchmark with a production-quality browser demo designed for honest,
privacy-preserving model exploration.

The best custom model reaches **94.88% held-out accuracy and 0.9885 ROC AUC** on
8,087 unseen reviews, closely matching scikit-learn's logistic-regression reference.

## Why this project matters

This is a controlled experiment, not a leaderboard. It crosses two text
representations with two learning algorithms to answer a focused question:

> Does model choice or representation matter more for this problem?

The answer is representation. Moving from three hand-crafted surface statistics to
TF-IDF improves Adaline by **41.75 percentage points**. Swapping Adaline for logistic
regression on identical features moves accuracy by less than one point.

That comparison also validates the custom implementation: hand-written batch gradient
descent performs within 0.41 points of a mature solver on the same TF-IDF matrix.

## Results

All metrics use one stratified 80/20 split with `random_state=42`. The dataset is
balanced, so chance accuracy is 50%. The positive class is `CG`—computer-generated.

| Model | Representation | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adaline | 3 surface features | 53.13% | 52.27% | 71.90% | 60.54% | 0.5490 |
| Logistic regression | 3 surface features | 53.84% | 52.76% | 73.21% | 61.33% | 0.5535 |
| Adaline | 10 surface features | 75.57% | 75.07% | 76.55% | 75.80% | 0.8382 |
| **Adaline** | **TF-IDF, 1–2 grams** | **94.88%** | **96.32%** | **93.32%** | **94.80%** | **0.9885** |
| Logistic regression | TF-IDF, 1–2 grams | 94.47% | 95.84% | 92.98% | 94.39% | 0.9874 |

The weak hand-crafted baseline is an intentional and useful negative result. Review
length, hype words, and capitalisation barely separate the classes because the generated
reviews were created by a GPT-2 model fine-tuned on genuine reviews. Vocabulary and
short phrase patterns carry the useful signal.

## Live experience

The Vercel application contains three recruiter-friendly views:

- **Analyse:** Paste a review or load a genuine/generated held-out example. The exact
  trained TF-IDF + Adaline pipeline runs locally in the browser and returns its verdict,
  raw decision score, and strongest matched terms.
- **Benchmark:** Explore the five-model comparison, central finding, and confusion
  matrix without relying on claims copied into the interface.
- **Method:** See the Adaline objective, the complete gradient update, engineering
  decisions, and limitations.

The classifier is not presented as an authorship oracle. Its score is an uncalibrated
linear activation, the corpus contains output from one older generator, and unfamiliar
domains may not generalise. Those limits are visible in the product, not buried here.

### Privacy and architecture

Inference is browser-native. The deployed site has no server endpoint, account, cookie,
or analytics integration, and submitted review text never leaves the visitor's device.

`scripts/export_web_model.py` converts the committed scikit-learn pipeline into three
aligned arrays: 139,295 vocabulary terms, IDF values, and Adaline weights with MaxAbs
scaling folded in. `web/classifier.js` reproduces the trained transformation exactly:

1. Unicode accent stripping and lowercase word tokenisation
2. Word unigram and bigram counts with sublinear term frequency
3. Learned IDF weighting and L2 normalisation
4. MaxAbs scaling and the Adaline linear decision function
5. A decision threshold of `0.500`

Cross-language tests pin browser scores to Python inference within `1e-6`.

## Run locally

### Website

Requires Node.js 24 and Python 3 for the zero-dependency preview server.

```bash
git clone https://github.com/aaryabookseller16/Fake-Review-Classifier.git
cd Fake-Review-Classifier
npm test
npm run build
npm run preview
```

Open [http://localhost:4173](http://localhost:4173).

### Model and research dashboard

Requires Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
streamlit run app/streamlit_app.py
```

To reproduce every trained artifact and metric from the bundled dataset:

```bash
PYTHONPATH=src python -m fakereview.train
```

Training writes `models/*.joblib` and `reports/metrics.json`. Regenerate the exact
browser payload afterward with:

```bash
PYTHONPATH=src python scripts/export_web_model.py
```

## From-scratch Adaline

Adaline learns a linear function `z = Xw + b` by minimising mean squared error:

```text
L(w, b) = mean((y - (Xw + b))²)
```

The complete update is intentionally visible:

```python
errors = y - self.activation(self.net_input(X))
self.w_ += self.lr * 2.0 * self._rmatvec(X, errors) / n_samples
self.b_ += self.lr * 2.0 * errors.mean()
```

The production implementation adds three safeguards a textbook example usually omits:

- **Sparse operations:** Densifying the 32,345 × 139,295 TF-IDF training matrix would
  require roughly 35 GB.
- **Divergence detection:** Very large but finite losses are rejected before early
  stopping can mislabel a worsening run as converged.
- **Feature scaling:** `StandardScaler` serves dense features; `MaxAbsScaler` preserves
  sparsity for TF-IDF.

## Testing

The Python suite contains 37 tests covering:

- convergence and divergence behavior;
- sparse/dense numerical equivalence;
- stable feature ordering and matrix widths;
- batch/single-review consistency;
- label polarity from training through presentation;
- empty, missing, and malformed text inputs;
- all five pipeline configurations.

The browser suite adds exact Python/JavaScript inference parity, model integrity,
tokenisation, safe unknown-vocabulary behavior, and static production-bundle checks.
GitHub Actions runs linting and the Python and browser suites on every push and pull
request to `main`.

## Repository layout

```text
├── app/                        # optional Streamlit research dashboard
├── data/                       # Salminen et al. review corpus
├── models/                     # five committed trained pipelines
├── reports/metrics.json        # source of truth for reported metrics
├── scripts/
│   ├── export_web_model.py     # exact Python → browser model export
│   └── verify_web_build.mjs    # deployment bundle integrity check
├── src/fakereview/
│   ├── adaline.py              # from-scratch sparse Adaline
│   ├── data.py                 # loading, splitting, label convention
│   ├── evaluate.py             # evaluation and report serialization
│   ├── features.py             # interpretable feature extractors
│   ├── pipelines.py            # five-model experiment registry
│   └── train.py                # reproducible training entry point
├── tests/                      # Python model tests
├── web/                        # production Vercel application
│   ├── classifier.js           # dependency-free browser inference
│   ├── model.json              # exported vocabulary and weights
│   └── tests/                  # browser inference tests
└── vercel.json                 # production deployment configuration
```

## Data and responsible use

The project uses the [Salminen et al. fake reviews dataset](https://osf.io/tyue9/):
40,432 Amazon product reviews across ten categories, balanced between organic (`OR`)
and computer-generated (`CG`) text.

The classifier should support human review, not automatically accuse authors or remove
content. The evaluation covers one marketplace and one GPT-2-based generator; transfer
to modern models, other languages, or other domains has not been established.

## License

[MIT](LICENSE)
