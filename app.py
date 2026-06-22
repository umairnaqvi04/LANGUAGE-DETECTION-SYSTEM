import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


def predict_with_confidence(m, text_vector):
    """
    Returns (prediction, confidence) for any model.
    Uses predict_proba if available (Naive Bayes, Random Forest).
    Falls back to decision_function + softmax for models like LinearSVC
    that don't support predict_proba.
    """
    pred = m.predict(text_vector)[0]

    if hasattr(m, "predict_proba"):
        confidence = max(m.predict_proba(text_vector)[0])
    elif hasattr(m, "decision_function"):
        scores = m.decision_function(text_vector)[0]
        # decision_function can return a single score for binary,
        # or an array of per-class scores for multi-class (our case)
        scores = np.atleast_1d(scores)
        exp_scores = np.exp(scores - np.max(scores))  # softmax, numerically stable
        probs = exp_scores / exp_scores.sum()
        confidence = max(probs)
    else:
        confidence = 0.95  # last-resort fallback

    return pred, confidence

st.set_page_config(page_title="Language Detection", page_icon="🌍", layout="wide")

@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('models/language_detection_model.pkl', 'rb'))
        vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        all_models = pickle.load(open('models/all_models.pkl', 'rb'))
        return model, vectorizer, all_models
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None

@st.cache_resource
def load_logo():
    try:
        return Image.open('iqra_logo.png')
    except:
        return None

model, vectorizer, all_models = load_models()
logo = load_logo()

# Header
if logo:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=300)

st.markdown("<h1 style='text-align: center; color: #667eea;'>🌍 Language Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #666;'>IQRA UNIVERSITY ISLAMABAD - AIC-221</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
page = st.sidebar.radio("📍 Select", ["🏠 Home", "🔍 Detect Language", "📊 Model Analysis", "ℹ️ About"])

if page == "🏠 Home":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 What is this?")
        st.write("Automatic language detection system supporting 17 languages with 95%+ accuracy.")
        
        st.markdown("### 🎯 Supported Languages")
        st.write("English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Swedish, Danish, Turkish, Arabic, Greek, Hindi, Tamil, Kannada, Malayalam")
    
    with col2:
        st.markdown("### 🤖 ML Models")
        st.write("**SVM**: 95.21% | **Naive Bayes**: 94.87% | **Random Forest**: 85.25%")
        
        st.markdown("### ✨ Features")
        st.write("✅ Single text prediction\n✅ Batch CSV processing\n✅ Confidence scores\n✅ Model comparison")

elif page == "🔍 Detect Language":
    if model is None:
        st.error("❌ Models not loaded!")
    else:
        tab1, tab2 = st.tabs(["📝 Single Text", "📤 Batch Upload"])
        
        with tab1:
            st.markdown("### Enter text to detect language:")
            text_input = st.text_area("", placeholder="Type your text here...", height=150)
            
            if st.button("🚀 Detect Language", use_container_width=True):
                if text_input.strip() == "":
                    st.warning("⚠️ Please enter some text!")
                else:
                    try:
                        text_vector = vectorizer.transform([text_input])
                        pred, confidence = predict_with_confidence(model, text_vector)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🎯 Language", pred)
                        with col2:
                            st.metric("📊 Confidence", f"{confidence:.2%}")
                        with col3:
                            st.metric("📈 Length", len(text_input))
                        
                        st.markdown("---")
                        st.markdown("### 🤖 All Models Prediction:")
                        results = []
                        for name, m in all_models.items():
                            p, conf = predict_with_confidence(m, text_vector)
                            results.append({"Model": name, "Language": p, "Confidence": f"{conf:.2%}"})
                        
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                        st.success("✅ Detection complete!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with tab2:
            st.markdown("### Upload CSV (must have 'Text' column):")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'Text' not in df.columns:
                        st.error("❌ CSV must contain 'Text' column!")
                    else:
                        st.info(f"📊 Processing {len(df)} rows...")
                        
                        if st.button("🚀 Process Batch", use_container_width=True):
                            progress_bar = st.progress(0)
                            predictions = []
                            confidences = []
                            
                            for idx, text in enumerate(df['Text']):
                                try:
                                    text_vector = vectorizer.transform([str(text)])
                                    pred, confidence = predict_with_confidence(model, text_vector)
                                    predictions.append(pred)
                                    confidences.append(confidence)
                                except:
                                    predictions.append("Error")
                                    confidences.append(0)
                                
                                progress_bar.progress((idx + 1) / len(df))
                            
                            df['Language'] = predictions
                            df['Confidence'] = confidences
                            
                            st.markdown("---")
                            st.markdown("### ✅ Results:")
                            st.dataframe(df, use_container_width=True)
                            
                            csv_results = df.to_csv(index=False)
                            st.download_button("📥 Download Results", csv_results, "results.csv", "text/csv")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif page == "📊 Model Analysis":
    st.markdown("### 📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🥇 SVM", "95.21%", "Best")
    with col2:
        st.metric("🥈 Naive Bayes", "94.87%", "Used in API")
    with col3:
        st.metric("🥉 Random Forest", "85.25%", "Ensemble")
    
    st.markdown("---")
    
    metrics_data = {
        'Model': ['SVM', 'Naive Bayes', 'Random Forest'],
        'Accuracy': ['95.21%', '94.87%', '85.25%'],
        'Type': ['Supervised', 'Probabilistic', 'Ensemble']
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Dataset Info")
    dataset_info = {
        'Metric': ['Total Samples', 'Languages', 'Training Set', 'Testing Set', 'Features'],
        'Value': ['10,337', '17', '8,269 (80%)', '2,068 (20%)', '5,000']
    }
    st.dataframe(pd.DataFrame(dataset_info), use_container_width=True)

elif page == "ℹ️ About":
    st.markdown("### 🎓 Project Information")
    
    st.markdown("""
    **University:** IQRA UNIVERSITY ISLAMABAD  
    **Course:** AIC-221 Introduction to Machine Learning  
    **Instructor:** Abdul Baqi Malik  
    **Batch:** AI-SP-24  
    **Semester:** 6th  
    **Topic:** Language Detection System  
    
    ---
    
    ### 🛠️ Technology Stack
    - **Language:** Python 3.8+
    - **ML:** scikit-learn
    - **Frontend:** Streamlit
    - **Data:** pandas, numpy
    - **Vectorization:** TF-IDF
    
    ### 🎯 ML Approaches
    1. Support Vector Machine (SVM)
    2. Naive Bayes Classifier
    3. Random Forest Ensemble
    
    ### ✨ Features
    ✅ Real-time language detection  
    ✅ 17 languages supported  
    ✅ 95%+ accuracy  
    ✅ Batch processing  
    ✅ Confidence scores  
    ✅ Model comparison  
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #999;'><p>Language Detection System | IQRA University © 2026</p></div>", unsafe_allow_html=True)
