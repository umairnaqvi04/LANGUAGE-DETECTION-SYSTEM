"""
Language Detection System - FRONTEND (Streamlit App, self-contained)
AIC-221 Introduction to Machine Learning - Semester Project #24

This single file does EVERYTHING needed to deploy:
  - Loads Language_Detection.csv
  - Trains 4 supervised models (Naive Bayes, Logistic Regression,
    Linear SVM, Random Forest) + KMeans (unsupervised) ONCE on app
    startup (cached with @st.cache_resource so it does not retrain
    on every click/page change)
  - Shows EDA, Model Comparison, and Live Prediction pages

No .pkl files needed. Only 3 files required to deploy:
    app.py, requirements.txt, Language_Detection.csv

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push app.py + requirements.txt + Language_Detection.csv to your
    GitHub repo root, then set Main file path = app.py.
"""

import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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
)

st.set_page_config(page_title="Language Detection System", page_icon="🌐", layout="wide")

DATA_FILE = "Language_Detection.csv"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"[\[\]\(\)0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ----------------------------------------------------------------------
# Train everything ONCE (cached) — this replaces the separate backend.py
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models for the first time (~20-30 sec)...")
def train_all():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["Text", "Language"]).drop_duplicates(subset=["Text"])
    df["clean_text"] = df["Text"].apply(clean_text)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Language"])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=3000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Supervised models (rubric requires >= 3 ML models compared)
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
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        results.append(
            {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
        )
        trained_models[name] = m

    metrics_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    # Unsupervised approach: KMeans clustering (bonus ML approach)
    n_clusters = df["Language"].nunique()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_train)
    cluster_df = (
        pd.DataFrame(
            {"Cluster": clusters, "Language": df["Language"].loc[X_train_text.index].values}
        )
        .groupby(["Cluster", "Language"])
        .size()
        .reset_index(name="Count")
    )

    return {
        "df": df,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "metrics_df": metrics_df,
        "best_name": best_name,
        "best_model": best_model,
        "cm_df": cm_df,
        "cluster_df": cluster_df,
    }


# ----------------------------------------------------------------------
# Guard: only the raw CSV is needed now
# ----------------------------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.error(
        f"⚠️ `{DATA_FILE}` not found. Put it in the same folder as `app.py` "
        f"(your GitHub repo root) and reload."
    )
    st.stop()

artifacts = train_all()
df = artifacts["df"]
vectorizer = artifacts["vectorizer"]
label_encoder = artifacts["label_encoder"]
metrics_df = artifacts["metrics_df"]
model_name = artifacts["best_name"]
model = artifacts["best_model"]
cm_df = artifacts["cm_df"]
cluster_df = artifacts["cluster_df"]

# ----------------------------------------------------------------------
# Sidebar navigation (single-file multipage)
# ----------------------------------------------------------------------
st.sidebar.title("🌐 Language Detection")
page = st.sidebar.radio(
    "Navigate", ["🏠 Home", "📊 EDA", "📈 Model Comparison", "🔮 Live Prediction"]
)
st.sidebar.markdown("---")
st.sidebar.caption("AIC-221 Introduction to Machine Learning")
st.sidebar.caption(f"Best Model: **{model_name}**")

# ----------------------------------------------------------------------
# Page 1: Home
# ----------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🌐 Language Detection System")
    st.markdown(
        f"""
        This app identifies the **language** of any given text using Machine Learning.

        **Dataset:** Text Language Dataset (Kaggle) — {len(df)} samples, {df['Language'].nunique()} languages
        **Approaches used:** Multinomial Naive Bayes, Logistic Regression, Linear SVM,
        Random Forest (supervised) + KMeans (unsupervised clustering)

        Use the sidebar to explore the dataset, compare model performance,
        or try a live prediction.
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Languages", df["Language"].nunique())
    col3.metric("Best Model Accuracy", f"{metrics_df.iloc[0]['Accuracy'] * 100:.2f}%")

# ----------------------------------------------------------------------
# Page 2: EDA
# ----------------------------------------------------------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Language Distribution")
    lang_counts = df["Language"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=lang_counts.values, y=lang_counts.index, hue=lang_counts.index, ax=ax, palette="viridis", legend=False)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Language")
    st.pyplot(fig)

    st.subheader("Text Length Distribution")
    text_length = df["Text"].astype(str).apply(len)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(text_length, bins=50, ax=ax2, color="steelblue")
    ax2.set_xlabel("Text Length (characters)")
    st.pyplot(fig2)

    st.subheader("Sample Records")
    st.dataframe(df.sample(10, random_state=1)[["Text", "Language"]], use_container_width=True)

    st.subheader("Unsupervised Clustering Insight (KMeans)")
    st.caption("How the unsupervised KMeans clusters align with the true language labels.")
    st.dataframe(cluster_df, use_container_width=True)

# ----------------------------------------------------------------------
# Page 3: Model Comparison
# ----------------------------------------------------------------------
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison & Evaluation")

    st.subheader("Metrics Table")
    st.dataframe(
        metrics_df.style.format(
            {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}
        ),
        use_container_width=True,
    )

    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=metrics_df, x="Accuracy", y="Model", hue="Model", ax=ax, palette="mako", legend=False)
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    st.subheader(f"Confusion Matrix — {model_name}")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

# ----------------------------------------------------------------------
# Page 4: Live Prediction
# ----------------------------------------------------------------------
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Language Prediction")
    st.write("Enter any text below and the model will predict its language.")

    user_input = st.text_area("Enter text:", height=120, placeholder="Type or paste text here...")

    if st.button("Detect Language", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])

            pred_label = model.predict(vec)[0]
            pred_lang = label_encoder.inverse_transform([pred_label])[0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vec)[0]
            else:
                scores = model.decision_function(vec)[0]
                probs = softmax(scores)

            st.success(f"Predicted Language: **{pred_lang}**")

            top_idx = np.argsort(probs)[::-1][:5]
            top_langs = label_encoder.inverse_transform(top_idx)
            top_probs = probs[top_idx]

            result_df = pd.DataFrame({"Language": top_langs, "Confidence": top_probs})
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=result_df, x="Confidence", y="Language", hue="Language", ax=ax, palette="rocket", legend=False)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
