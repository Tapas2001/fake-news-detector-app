# fake_news_detector.py
import pandas as pd
import json
import joblib
import streamlit as st
from utils import load_data


# ---------------------- UI Title ----------------------
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")


# ---------- Load and Prepare Data ----------
st.title("📰 Fake News Detector")
@st.cache_data
def load_dataset():
    df = load_data(path="data\sample.csv")
    return df
    
# ---------- Load model and vectorizer ----------
@st.cache_resource
def load_model(model_path="model\model.pkl", vectorizer_path="model\vectorizer.pkl"):
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load model or vectorizer: {e}")
        return None, None
    
# ---------- Load accuracy ----------   
@st.cache_data
def load_accuracy():
    try:
        with open("model\metrics.json") as f:
            metrics = json.load(f)
            return metrics.get("accuracy", None)
    except Exception:
        return None
    

df = load_dataset()
model, vectorizer = load_model()
accuracy = load_accuracy()

if not df.empty:
    st.write("### Dataset Sample")
    st.dataframe(df.head())

    if model and vectorizer:
        
        st.subheader("Model Accuracy")
        if accuracy is not None:
            st.info(f"**Model Testing Accuracy:** {accuracy * 100:.2f}%")
        else:
            st.warning("Model accuracy not available.")


        # ---------- User Input ----------
        st.write("## Test a Single Headline")
        headline = st.text_input("Enter a news headline")

        if st.button("Predict"):
            if headline.strip():
                try:
                    vec = vectorizer.transform([headline])
                    result = model.predict(vec)[0]
                    prob = model.predict_proba(vec).max()
                    st.success(f"**Prediction:** {'Fake' if result == 1 else 'Real'} ({round(prob * 100, 2)}% confidence)")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.warning("Please enter a headline.")


        # ---------- Batch Upload ----------
        st.write("## Batch CSV Prediction")
        file = st.file_uploader("Upload a CSV file with a 'title' column", type=["csv"])
        if file:
            try:
                batch_df = pd.read_csv(file)
                if 'title' in batch_df.columns:
                    batch_vec = vectorizer.transform(batch_df['title'].fillna(""))
                    batch_df['prediction'] = model.predict(batch_vec)
                    batch_df['prediction'] = batch_df['prediction'].map({0: "Real", 1: "Fake"})
                    batch_df['confidence'] = model.predict_proba(batch_vec).max(axis=1)
                    batch_df['confidence'] = batch_df['confidence'].round(2)
                    st.dataframe(batch_df[['title', 'prediction', 'confidence']])
                    st.download_button("Download Results", batch_df.to_csv(index=False), "results.csv")
                else:
                    st.error("Uploaded CSV must contain a 'title' column.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    else:
        st.warning("Model not loaded. Please train the model using `train.py`.")
else:
    st.warning("No data loaded. Cannot proceed with model training.")
