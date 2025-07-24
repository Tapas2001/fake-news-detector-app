# train.py
import pandas as pd
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import load_data

# ---------- Configuration ----------
DATA_PATH = "data/news.csv"  # Update this path as needed
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

def train_and_save_model(df):
    try:
        # # ---------- Splitting Data ----------
        X = df['title']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ---------- Text Vectorization ----------
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        x_train_tfidf = vectorizer.fit_transform(X_train)
        x_test_tfidf = vectorizer.transform(X_test)

        # ---------- Train Model ----------
        model = LogisticRegression()
        model.fit(x_train_tfidf, y_train)
        preds = model.predict(x_test_tfidf)
        accuracy = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds).tolist()
        
        # ---------- Save Model and Vectorizer ----------
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        # Save accuracy
        metrics = {
            "accuracy": accuracy,
            "confusion_matrix": cm
        }
        with open("model/metrics.json", "w") as f:
            json.dump(metrics, f)
    except Exception as e:
        raise Exception(f"Training and model saving failed: {e}")
    
    

# ---------- Run Training ----------
if __name__ == "__main__":
    data = load_data(DATA_PATH)
    if data is not None:
        train_and_save_model(data)