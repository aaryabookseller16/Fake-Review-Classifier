"""Streamlit web app for the fake-review classifier.

Three views, selected from the sidebar:

**Classify**
    Paste a review, pick a model, get a prediction with a confidence meter and
    the extracted interpretable features.
**Benchmark**
    The model comparison read from ``reports/metrics.json`` -- the same numbers
    as the README, loaded from disk rather than retyped.
**Method**
    The Adaline update rule, the training curve, and the learning-rate
    stability results.

Sidebar navigation is used rather than ``st.tabs`` deliberately: Streamlit
measures widgets inside an inactive tab as zero-width, so charts and dataframes
that are not on the first tab render collapsed and never re-measure.

Run locally::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the `fakereview` package importable when Streamlit runs this file
# directly. Resolved from __file__, never the current working directory --
# the original app used os.path.join("..", "models", ...) and therefore only
# worked if you happened to launch it from inside its own folder.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt  # noqa: E402
import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import theme  # noqa: E402
from fakereview.data import CLASS_NAMES  # noqa: E402
from fakereview.evaluate import load_report  # noqa: E402
from fakereview.features import extract, feature_names  # noqa: E402
from fakereview.pipelines import DEFAULT_PIPELINE, PIPELINES  # noqa: E402

MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.apply(st)


# ----------------------------------------------------------------------
# Loading (cached -- unpickling a 139k-term TF-IDF vocabulary is not cheap)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(key: str):
    """Load one trained pipeline from disk, or None if it was never trained."""
    path = MODELS_DIR / f"{key}.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def get_report():
    """Load reports/metrics.json, or None if training has not been run."""
    return load_report()


def available_models() -> list[str]:
    """Registry keys that actually have a trained artifact on disk."""
    return [k for k in PIPELINES if (MODELS_DIR / f"{k}.joblib").exists()]


# ----------------------------------------------------------------------
# Small presentational helpers
# ----------------------------------------------------------------------
def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rise"><h1 class="hero-title">{title}</h1>'
        f'<p class="hero-sub">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def stat_row(items: list[tuple[str, str, str]]) -> None:
    """Render a row of stat cards: (label, value, sub-caption)."""
    cards = "".join(
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-sub">{sub}</div></div>'
        for label, value, sub in items
    )
    st.markdown(f'<div class="stat-row rise">{cards}</div>', unsafe_allow_html=True)


def callout(text: str, muted: bool = False) -> None:
    cls = "callout callout-muted" if muted else "callout"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.markdown(
    '<div style="font-size:1.05rem;font-weight:700;letter-spacing:-.02em;'
    'margin-bottom:.15rem;color:var(--text);">🔍 Fake Review Detector</div>'
    '<div style="font-size:.82rem;color:var(--text-muted);line-height:1.5;">'
    "Adaline written from scratch in NumPy, benchmarked against scikit-learn "
    "on 40,432 Amazon reviews.</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "View",
    ["Classify", "Benchmark", "Method"],
    label_visibility="collapsed",
)

st.sidebar.divider()

trained = available_models()
model_key = None
if not trained:
    st.sidebar.error("No trained models found.")
else:
    default_index = trained.index(DEFAULT_PIPELINE) if DEFAULT_PIPELINE in trained else 0
    model_key = st.sidebar.selectbox(
        "Model",
        trained,
        index=default_index,
        format_func=lambda k: PIPELINES[k].label,
    )
    spec = PIPELINES[model_key]
    st.sidebar.markdown(
        f'<div style="margin-top:.5rem;font-size:.82rem;color:var(--text-muted);'
        f'line-height:1.7;">'
        f'<span class="pill">{spec.representation}</span><br>'
        f'<span style="display:inline-block;margin-top:.4rem;">{spec.summary}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

report = get_report()
if report and model_key:
    res = report["results"].get(model_key)
    if res:
        # Custom markup rather than st.metric: the sidebar is ~230px wide and
        # st.metric truncates both its label and its value at that width.
        st.sidebar.markdown(
            "<div style='display:flex;gap:.5rem;margin-top:.9rem;'>"
            "<div style='flex:1;background:var(--surface-alt);border-radius:9px;"
            "padding:.55rem .65rem;'>"
            "<div style='font-size:.62rem;letter-spacing:.07em;font-weight:700;"
            "text-transform:uppercase;color:var(--text-muted);'>Accuracy</div>"
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:1.05rem;"
            f"font-weight:600;line-height:1.3;color:var(--text);\">{res['accuracy']:.1%}</div></div>"
            "<div style='flex:1;background:var(--surface-alt);border-radius:9px;"
            "padding:.55rem .65rem;'>"
            "<div style='font-size:.62rem;letter-spacing:.07em;font-weight:700;"
            "text-transform:uppercase;color:var(--text-muted);'>ROC AUC</div>"
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:1.05rem;"
            f"font-weight:600;line-height:1.3;color:var(--text);\">{res['roc_auc']:.3f}</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

st.sidebar.divider()
st.sidebar.markdown(
    '<div style="font-size:.75rem;color:var(--text-muted);line-height:1.6;">'
    "<b>Label convention</b><br>"
    "<code>1 = CG = fake</code><br><code>0 = OR = genuine</code><br>"
    "Defined once in <code>fakereview.data</code> and shared by training "
    "and serving.</div>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Guard: nothing to serve without trained artifacts
# ----------------------------------------------------------------------
if not trained:
    hero("Fake Review Detector", "No trained models were found in <code>models/</code>.")
    st.markdown("Train them first:")
    st.code("PYTHONPATH=src python -m fakereview.train", language="bash")
    st.stop()


# ====================================================================== #
# Classify                                                                #
# ====================================================================== #
if page == "Classify":
    hero(
        "Is this review written by a human?",
        "Paste a product review below. The model predicts whether it was "
        "computer-generated or written by a person, and shows the evidence "
        "behind the call.",
    )
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    # Real reviews from the held-out test split, not invented strings. Made-up
    # examples tend to sit outside the corpus's style distribution and give a
    # misleading impression of the model.
    examples = {
        "— write my own —": "",
        "Held-out example · truly fake (CG)": (
            "Works great. Exact replace for $60, with the instructions "
            "included.  If you have a larger computer, you'll want to get a "
            "solid replacement. "
        ),
        "Held-out example · truly genuine (OR)": (
            "They may not be picture perfect machine-finished but the metal is "
            "good they hold up and do their job and were much better than I "
            "expected for the amazingly low price."
        ),
    }

    left, right = st.columns([3, 2], gap="large")

    with left:
        choice = st.selectbox("Load an example", list(examples))
        review_text = st.text_area(
            "Review text",
            value=examples[choice],
            height=170,
            placeholder="Type or paste a product review…",
            label_visibility="collapsed",
        )
        go = st.button("Classify review", type="primary")

    with right:
        if choice.startswith("Held-out"):
            callout(
                "This is a <b>real review from the held-out test split</b>, with "
                "its true label in the name — so you can check the model against "
                "ground truth rather than against intuition."
            )
        else:
            callout(
                "Reviews in this corpus are short and unremarkable by design. "
                "The generated ones were produced by a GPT-2 model fine-tuned on "
                "real reviews, so they rarely look obviously fake.",
                muted=True,
            )

    if go:
        if not review_text.strip():
            st.warning("Enter a review first.")
            st.stop()

        model = load_model(model_key)
        prediction = int(model.predict([review_text])[0])
        score = float(model.decision_function([review_text])[0])

        # 1 == fake. Imported from fakereview.data rather than written out
        # here -- that duplication is what caused the original app to display
        # every prediction inverted.
        is_fake = prediction == 1
        verdict = CLASS_NAMES[prediction]
        kind = "fake" if is_fake else "genuine"
        icon = "⚠" if is_fake else "✓"
        detail = "computer-generated" if is_fake else "written by a human"

        # Map the unbounded decision score onto the meter, clamped to [0, 1]
        # with the 0.5 threshold at the centre.
        pin = max(0.0, min(1.0, score)) * 100

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="verdict verdict-{kind}">'
            f'<div class="verdict-icon">{icon}</div>'
            f"<div style='flex:1'>"
            f'<div class="verdict-label">Prediction</div>'
            f'<div class="verdict-value">{verdict} · {detail}</div>'
            f'<div class="meter"><div class="meter-pin" style="left:{pin:.1f}%"></div></div>'
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:.7rem;color:var(--text-muted);margin-top:.3rem;'>"
            f"<span>genuine · 0.0</span><span>threshold 0.5</span>"
            f"<span>1.0 · fake</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        stat_row(
            [
                ("Decision score", f"{score:.3f}", "threshold at 0.500"),
                ("Model", PIPELINES[model_key].label.split(" + ")[0], PIPELINES[model_key].representation),
                (
                    "Held-out accuracy",
                    f"{report['results'][model_key]['accuracy']:.2%}" if report else "—",
                    "on 8,087 unseen reviews",
                ),
            ]
        )

        callout(
            "The decision score is the model's raw continuous activation before "
            "thresholding — <b>not a calibrated probability</b>. It is unbounded "
            "and can fall outside 0–1. Making it a real probability is the "
            "calibration work listed in the README.",
            muted=True,
        )

        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        st.markdown("### Interpretable features")
        st.markdown(
            "<p style='color:var(--text-muted);font-size:.9rem;margin-top:-.4rem;'>"
            "Surface statistics for this review. The hand-crafted models use these "
            "directly; the TF-IDF models do not, but they are useful context.</p>",
            unsafe_allow_html=True,
        )
        names = feature_names("extended")
        values = extract([review_text], "extended")[0]
        feat = pd.DataFrame({"Feature": [n.replace("_", " ") for n in names], "Value": values})
        st.altair_chart(
            theme.style_chart(
                alt.Chart(feat)
                .mark_bar(cornerRadiusEnd=3, color=theme.INDIGO, opacity=.85)
                .encode(
                    x=alt.X("Value:Q", title=None),
                    y=alt.Y("Feature:N", sort=None, title=None),
                    tooltip=["Feature", alt.Tooltip("Value:Q", format=".3f")],
                )
                .properties(width="container", height=260)
            ),
            use_container_width=True,
        )


# ====================================================================== #
# Benchmark                                                               #
# ====================================================================== #
elif page == "Benchmark":
    hero(
        "Representation beats optimiser, by about fifty to one",
        "Five configurations on one held-out split. Changing how the text is "
        "represented moves accuracy by 41 points; changing the learning "
        "algorithm moves it by less than one.",
    )
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if not report:
        st.warning("No metrics report found. Run training to generate it.")
        st.code("PYTHONPATH=src python -m fakereview.train", language="bash")
        st.stop()

    ds = report["dataset"]
    best = max(report["results"].values(), key=lambda r: r["accuracy"])
    stat_row(
        [
            ("Training reviews", f"{ds['n_train']:,}", "stratified 80% split"),
            ("Held-out reviews", f"{ds['n_test']:,}", "never seen in training"),
            ("Class balance", "50 / 50", "so chance accuracy = 0.500"),
            ("Best accuracy", f"{best['accuracy']:.2%}", PIPELINES[best["pipeline"]].label),
        ]
    )

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("### Results")

    rows = []
    for key, spec in PIPELINES.items():
        r = report["results"].get(key)
        if not r:
            continue
        rows.append(
            {
                "Model": spec.label,
                "Representation": spec.representation,
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1"],
                "ROC AUC": r["roc_auc"],
                "Train (s)": r["train_seconds"],
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn(
                "Accuracy", format="%.4f", min_value=0.5, max_value=1.0
            ),
            "ROC AUC": st.column_config.NumberColumn("ROC AUC", format="%.4f"),
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
            "F1": st.column_config.NumberColumn(format="%.4f"),
            "Train (s)": st.column_config.NumberColumn(format="%.1f s"),
        },
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### Accuracy")
        st.altair_chart(
            theme.style_chart(
                alt.Chart(table[["Model", "Accuracy"]])
                .mark_bar(cornerRadiusEnd=4, height=24)
                .encode(
                    x=alt.X(
                        "Accuracy:Q",
                        scale=alt.Scale(domain=[0.5, 1.0], zero=False, clamp=True),
                        title="Held-out accuracy",
                    ),
                    y=alt.Y("Model:N", sort=None, title=None),
                    color=alt.Color(
                        "Model:N",
                        scale=alt.Scale(range=theme.MODEL_COLORS),
                        legend=None,
                    ),
                    tooltip=["Model", alt.Tooltip("Accuracy:Q", format=".4f")],
                )
                .properties(width="container", height=230)
            ),
            use_container_width=True,
        )
        st.markdown(
            "<p style='color:var(--text-muted);font-size:.82rem;margin-top:-.6rem;'>"
            "Axis starts at 0.50 — the dataset is balanced, so that is what "
            "random guessing achieves.</p>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("### ROC curves")
        roc_rows = []
        for key, spec in PIPELINES.items():
            r = report["results"].get(key)
            if not r or "roc" not in r:
                continue
            for fpr, tpr in zip(r["roc"]["fpr"], r["roc"]["tpr"]):
                roc_rows.append({"Model": spec.label, "FPR": fpr, "TPR": tpr})
        if roc_rows:
            diagonal = (
                alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]}))
                .mark_line(strokeDash=[4, 4], color="#cbd5e1", strokeWidth=1)
                .encode(x="FPR:Q", y="TPR:Q")
            )
            curves = (
                alt.Chart(pd.DataFrame(roc_rows))
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("FPR:Q", title="False positive rate"),
                    y=alt.Y("TPR:Q", title="True positive rate"),
                    color=alt.Color(
                        "Model:N",
                        scale=alt.Scale(range=theme.MODEL_COLORS),
                        legend=alt.Legend(orient="bottom", columns=1, title=None),
                    ),
                )
            )
            st.altair_chart(
                theme.style_chart(
                    (diagonal + curves).properties(width="container", height=290),
                    y_grid=True,
                ),
                use_container_width=True,
            )
            st.markdown(
                "<p style='color:var(--text-muted);font-size:.82rem;margin-top:-.6rem;'>"
                "Dashed diagonal is chance. The two TF-IDF models are almost "
                "indistinguishable at this scale.</p>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    st.markdown("### Where each model makes its mistakes")

    sel = st.selectbox(
        "Model",
        list(report["results"]),
        index=list(report["results"]).index(model_key) if model_key in report["results"] else 0,
        format_func=lambda k: PIPELINES[k].label,
        key="cm_model",
    )
    cm = report["results"][sel]["confusion"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    stat_row(
        [
            ("True negatives", f"{tn:,}", "genuine, called genuine"),
            ("False positives", f"{fp:,}", "genuine, wrongly flagged fake"),
            ("False negatives", f"{fn:,}", "fake, slipped through"),
            ("True positives", f"{tp:,}", "fake, correctly caught"),
        ]
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    cm_df = pd.DataFrame(
        [
            {"Actual": "Genuine", "Predicted": "Genuine", "n": tn},
            {"Actual": "Genuine", "Predicted": "Fake", "n": fp},
            {"Actual": "Fake", "Predicted": "Genuine", "n": fn},
            {"Actual": "Fake", "Predicted": "Fake", "n": tp},
        ]
    )
    # Cell text flips from white to dark on the paler half of the ramp, so the
    # two small counts stay readable.
    midpoint = max(tn, fp, fn, tp) / 2

    heat = (
        alt.Chart(cm_df)
        .mark_rect(cornerRadius=6, stroke="white", strokeWidth=3)
        .encode(
            x=alt.X(
                "Predicted:N",
                title="Predicted",
                sort=["Genuine", "Fake"],
                axis=alt.Axis(labelAngle=0, labelFontSize=12),
            ),
            y=alt.Y(
                "Actual:N",
                title="Actual",
                sort=["Genuine", "Fake"],
                axis=alt.Axis(labelFontSize=12),
            ),
            # Explicit ramp rather than a named Vega scheme: "indigo" is not one
            # of Vega's scheme names, and this keeps the heatmap on the same
            # brand colour the rest of the app uses.
            color=alt.Color(
                "n:Q",
                scale=alt.Scale(range=["#e0e7ff", theme.INDIGO]),
                legend=None,
            ),
            tooltip=["Actual", "Predicted", "n"],
        )
        .properties(width="container", height=230)
    )
    labels = heat.mark_text(fontSize=16, fontWeight=600).encode(
        text=alt.Text("n:Q", format=","),
        color=alt.condition(
            alt.datum.n > midpoint, alt.value("white"), alt.value(theme.SLATE_900)
        ),
    )
    st.altair_chart(theme.style_chart(heat + labels), use_container_width=True)

    callout(
        f"For review moderation, the <b>{fp:,} false positives</b> are usually the "
        f"costlier error — publicly flagging a real customer's review. That this "
        f"model errs in the safer direction is a property of where the 0.5 "
        f"threshold happens to fall, not a designed choice."
    )


# ====================================================================== #
# Method                                                                  #
# ====================================================================== #
elif page == "Method":
    hero(
        "How it works",
        "Adaline is the 1960 Widrow–Hoff refinement of the perceptron, and the "
        "whole learning algorithm here is four lines of NumPy.",
    )
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    st.markdown(
        "The distinguishing idea: compute the gradient against a **continuous "
        "linear activation** rather than the thresholded class label. That makes "
        "the loss differentiable and convex, so gradient descent converges to a "
        "global optimum. The model is linear, `z = Xw + b`, minimising mean "
        "squared error:"
    )
    st.latex(r"L(w, b) = \frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - (x_i^\top w + b)\bigr)^2")
    st.markdown("with gradients")
    st.latex(
        r"\frac{\partial L}{\partial w} = -\frac{2}{n}X^\top(y - z), \qquad "
        r"\frac{\partial L}{\partial b} = -\frac{2}{n}\sum_i (y_i - z_i)"
    )
    st.markdown("which is the entire update loop — no autograd, no solver library:")
    st.code(
        "errors = y - self.activation(self.net_input(X))\n"
        "self.w_ += self.lr * 2.0 * self._rmatvec(X, errors) / n_samples\n"
        "self.b_ += self.lr * 2.0 * errors.mean()",
        language="python",
    )

    if report:
        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        st.markdown("### Training curve")
        loss_rows = []
        for key, spec in PIPELINES.items():
            r = report["results"].get(key)
            if not r or not r.get("losses"):
                continue
            for i, v in enumerate(r["losses"]):
                loss_rows.append({"Model": spec.label, "Epoch (sampled)": i, "MSE": v})
        if loss_rows:
            st.altair_chart(
                theme.style_chart(
                    alt.Chart(pd.DataFrame(loss_rows))
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X("Epoch (sampled):Q", title="Epoch (sampled)"),
                        y=alt.Y(
                            "MSE:Q",
                            scale=alt.Scale(type="log"),
                            title="Mean squared error (log)",
                        ),
                        # Only the three Adaline pipelines appear here, so use
                        # a distinct three-colour range rather than the
                        # five-model ramp, whose first entries are near-identical
                        # greys.
                        color=alt.Color(
                            "Model:N",
                            scale=alt.Scale(range=[theme.AMBER, theme.SLATE_500, theme.INDIGO]),
                            legend=alt.Legend(orient="bottom", columns=1, title=None),
                        ),
                    )
                    .properties(width="container", height=320),
                    y_grid=True,
                ),
                use_container_width=True,
            )
            st.markdown(
                "<p style='color:var(--text-muted);font-size:.82rem;margin-top:-.6rem;'>"
                "Only the from-scratch models expose a loss history; scikit-learn's "
                "solver does not.</p>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    st.markdown("### The learning rate is the whole ballgame")
    st.markdown(
        "Adaline's failure mode is not a gentle plateau — with mean squared error "
        "the loss grows by orders of magnitude per epoch. Measured on the real "
        "TF-IDF representation:"
    )

    sweep = pd.DataFrame(
        [
            {"Learning rate": "0.1", "Accuracy": 0.9488, "ROC AUC": 0.9885, "Outcome": "converges · final MSE 0.0316"},
            {"Learning rate": "0.3", "Accuracy": 0.9486, "ROC AUC": 0.9880, "Outcome": "converges · final MSE 0.0137"},
            {"Learning rate": "0.5", "Accuracy": 0.9429, "ROC AUC": 0.9865, "Outcome": "converges · final MSE 0.0081"},
            {"Learning rate": "0.7", "Accuracy": None, "ROC AUC": None, "Outcome": "diverges at epoch 36"},
            {"Learning rate": "1.0", "Accuracy": None, "ROC AUC": None, "Outcome": "diverges at epoch 10"},
        ]
    )
    st.dataframe(
        sweep,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "ROC AUC": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    callout(
        "Two things worth noticing. Lower learning rates reach a <b>better</b> "
        "optimum here. And the loss the model minimises is not monotonically "
        "related to the accuracy it achieves — lr=0.5 reaches the lowest final "
        "MSE and the <b>worst</b> accuracy of the three."
    )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("#### Detecting divergence is subtler than it looks")
    st.markdown(
        "At lr=0.7 the loss reaches 2 × 10³⁴ — catastrophically diverged, but "
        "still a **finite float**, so an `np.isfinite` check passes it. Worse, the "
        "early-stopping rule `best_loss - loss < tol` is *satisfied* when the loss "
        "grows, so a naive implementation terminates the run and reports it as "
        "converged, returning a model with 49.99% accuracy.\n\n"
        "`AdalineGD` compares against the best loss seen so far and raises "
        "`AdalineDivergedError`, with divergence checked **before** stagnation. "
        "Two regression tests pin both halves."
    )

st.divider()
st.markdown(
    '<div style="text-align:center;font-size:.78rem;color:var(--text-muted);">'
    "Salminen et al. fake reviews corpus · 40,432 balanced reviews · "
    '<a href="https://github.com/aaryabookseller16/Fake-Review-Classifier" '
    'style="color:var(--brand);text-decoration:none;">source on GitHub</a></div>',
    unsafe_allow_html=True,
)
