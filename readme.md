# 🕵️ Fake Review Detector (Adaline from Scratch)

This project implements a simple, interpretable fake review classifier using **custom-engineered features** and a **from-scratch Adaline model**.

Unlike many black-box models, this one is transparent, fast, and built to demonstrate understanding of:
- Text preprocessing
- Manual gradient descent
- Linear classification theory

---

## 🔍 Problem

Fake (computer-generated) reviews flood online platforms, misleading buyers. Our goal is to build a lightweight classifier to distinguish between:

- **CG** → Computer-Generated (Fake)
- **OR** → Organic Review (Real)

---

## 🧠 Features Used

We extract **hand-crafted features** from each review:

| Feature              | Description                                              |
|----------------------|----------------------------------------------------------|
| `length`             | Character count of the review                            |
| `count_sus`          | Count of suspicious phrases (e.g., “free”, “buy now”)    |
| `count_caps`         | Number of fully capitalized words                        |

All features are standardized before training.

---

## 🧮 Model: Adaline (Adaptive Linear Neuron)

We implemented Adaline with:

- Linear activation
- Mean squared error loss
- Batch gradient descent
- Custom learning loop in NumPy

This helps build a deep understanding of how models actually learn.

---
