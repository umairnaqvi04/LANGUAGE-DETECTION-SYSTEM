"""
Language Detection System - BACKEND (Training Script)
AIC-221 Introduction to Machine Learning - Semester Project #24

Works locally AND on Google Colab.

What it does:
  1. Loads Language_Detection.csv
  2. Cleans text and builds character n-gram TF-IDF features
     (char n-grams generalize well across scripts: English, Arabic,
     Tamil, Russian, etc.)
  3. Trains 4 supervised models (Naive Bayes, Logistic Regression,
     Linear SVM, Random Forest) + 1 unsupervised model (KMeans)
  4. Evaluates with Accuracy, Precision, Recall, F1 + Confusion Matrix
  5. Saves everything app.py needs to run

Run:
    python backend.py
    (or paste into a Colab cell / upload + run -- it auto-detects Colab
    and will prompt you to upload the CSV if it isn't already there)

Outputs written to the SAME folder as this script:
    tfidf_vectorizer.pkl
    label_encoder.pkl
    best_model.pkl
    best_model_name.pkl
    model_metrics.csv
    confusion_matrix.csv
    cluster_summary.csv
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

DATA_FILE = "Language_Detection.csv"


# ---------------------------------------------------------------------
# 1. Load dataset (local path OR Google Colab upload prompt)
# ---------------------------------------------------------------------
def load_dataset(path=DATA_FILE):
    if os.path.exists(path):
        return pd.read_csv(path)
    try:
        import google.colab  # noqa: F401
        from google.colab import files

        print(f"'{path}' not found in this Colab session. Please upload it:")
        uploaded = files.upload()
        fname = list(uploaded.keys())[0]
        return pd.read_csv(fname)
    except ImportError:
        raise FileNotFoundError(
            f"Could not find '{path}'. Put the CSV in the same folder as backend.py."
        )


df = load_dataset()
print("Dataset shape:", df.shape)
print(df["Language"].value_counts())

# ---------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------
df = df.dropna(subset=["Text", "Language"]).drop_duplicates(subset=["Text"])


def clean_text(text):
    text = str(text)
    text = re.sub(r"[\[\]\(\)0-9]", " ", text)   # strip digits/brackets/refs
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


df["clean_text"] = df["Text"].apply(clean_text)

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Language"])

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Character n-grams work across all scripts (Latin, Arabic, Tamil, Cyrillic...)
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=3000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------------------------------------------------------------------
# 3. Supervised models (rubric requires >= 3 ML models compared)
# ---------------------------------------------------------------------
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(max_iter=5000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=40, n_jobs=-1, random_state=42
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )
    results.append(
        {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    )
    trained_models[name] = model
    print(f"{name}: Accuracy={acc:.4f}  F1={f1:.4f}")

metrics_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
metrics_df.to_csv("model_metrics.csv", index=False)
print("\nModel comparison:\n", metrics_df)

best_name = metrics_df.iloc[0]["Model"]
best_model = trained_models[best_name]
print(f"\nBest model selected: {best_name}")

best_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_preds)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
cm_df.to_csv("confusion_matrix.csv")

print("\nClassification report (best model):")
print(
    classification_report(
        y_test, best_preds, target_names=label_encoder.classes_, zero_division=0
    )
)

# ---------------------------------------------------------------------
# 4. Unsupervised approach: KMeans clustering (bonus ML approach)
# ---------------------------------------------------------------------
n_clusters = df["Language"].nunique()
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train)

cluster_summary = (
    pd.DataFrame(
        {"Cluster": clusters, "Language": df["Language"].loc[X_train_text.index].values}
    )
    .groupby(["Cluster", "Language"])
    .size()
    .reset_index(name="Count")
)
cluster_summary.to_csv("cluster_summary.csv", index=False)

# ---------------------------------------------------------------------
# 5. Save all artifacts for app.py
# ---------------------------------------------------------------------
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(best_model, "best_model.pkl")
joblib.dump(best_name, "best_model_name.pkl")

print("\nAll artifacts saved successfully in the current folder:")
print(" - tfidf_vectorizer.pkl")
print(" - label_encoder.pkl")
print(" - best_model.pkl")
print(" - best_model_name.pkl")
print(" - model_metrics.csv")
print(" - confusion_matrix.csv")
print(" - cluster_summary.csv")
print("\nCopy these files + app.py + requirements.txt + Language_Detection.csv")
print("into your GitHub repo root, then deploy app.py on Streamlit Cloud.")
