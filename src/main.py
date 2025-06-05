import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from model import AdalineGD
from features import count_sus_words, count_capital_words, count_review_length

import os

# Ensure models/ directory exists
os.makedirs("models", exist_ok=True)


# Load dataset
df = pd.read_csv("/Users/aaryabookseller/Desktop/Projects/Fake Review Classification/data/fake_reviews_dataset.csv")

# Extract features
sus_words = ["free", "amazing", "best", "buy now", "limited", "guaranteed"]
df = count_sus_words(df, sus_words)
df = count_capital_words(df)
df = count_review_length(df)

# Prepare X and y
X = df[["length", "count_sus", "count_caps"]].values
y = df["label"].map({"CG": 1, "OR": 0}).values

# Standardize
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Train model
model = AdalineGD(lr=0.01, n_iter=20)
model.fit(X_std, y)

# Save model and scaler
joblib.dump(model, "models/adaline_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
