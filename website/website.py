import sys
import os

# Add source directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
import pandas as pd
import joblib

from features import (
    count_sus_words,
    count_capital_words,
    count_review_length
)

# ---------- Page Config ----------
st.set_page_config(page_title="🕵️‍♂️ Fake Review Detector", layout="centered")
st.markdown("<h1 style='text-align: center;'>🕵️‍♂️ Fake Review Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a review below and let the model predict whether it's <b>Genuine</b> or <b>Fake</b>.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------- Load Model and Scaler ----------
MODEL_PATH = os.path.join("..", "models", "adaline_model.pkl")
SCALER_PATH = os.path.join("..", "models", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("❌ Model or scaler not found. Please run `main.py` to train and save them.")
    st.stop()

# ---------- User Input ----------
st.subheader("✍️ Enter Review Text")
review_text = st.text_area("", height=150, placeholder="Type or paste the review here...")

if st.button("🔍 Classify Review"):
    if not review_text.strip():
        st.warning("⚠️ Please enter a review before classifying.")
        st.stop()

    # Create DataFrame
    df = pd.DataFrame({'text_': [review_text]})

    # Extract Features
    try:
        df = count_sus_words(df, suspicious_list=["free", "amazing", "best", "buy now", "limited", "guaranteed"])
        df = count_capital_words(df)
        df = count_review_length(df)
    except Exception as e:
        st.error(f"⚠️ Feature extraction failed: {e}")
        st.stop()

    # Display Features
    st.markdown("### 📊 Extracted Features")
    st.dataframe(df[['len_reviews', 'count_sus', 'count_caps']], use_container_width=True)

    # Prediction
    X_input = df[['len_reviews', 'count_sus', 'count_caps']].values
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    confidence = model.activation(model.net_input(X_scaled))[0]

    # Output
    st.markdown("### 🧠 Prediction Result")
    if prediction == 1:
        st.success("✅ This review is predicted to be **Genuine**.")
    else:
        st.error("⚠️ This review is predicted to be **Fake**.")

    st.markdown(f"**Confidence Score:** `{confidence:.4f}` (Closer to 1 = Genuine, 0 = Fake)")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.9em;'>Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)
