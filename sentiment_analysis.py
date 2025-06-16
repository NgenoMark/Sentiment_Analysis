# sentiment_analysis.py

import pandas as pd
import re
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------- Text Cleaning Function --------
def clean_text(text):
    """Clean the input text: remove HTML, non-letters, and lowercase."""
    text = re.sub(r'<.*?>', '', str(text))        # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)        # Keep letters only
    return text.lower()

# -------- Main Training Function --------
def train_sentiment_model(dataset_path):
    print("📥 Loading dataset from:", dataset_path)
    df = pd.read_csv(dataset_path)

    # ✅ Check columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    # ✅ Drop rows with missing values
    df = df.dropna(subset=['text', 'label'])

    # ✅ Map 1 → positive, 0 → negative
    label_map = {1: 'positive', 0: 'negative'}
    df['label'] = df['label'].map(label_map)

    # ✅ Clean text
    df['text'] = df['text'].apply(clean_text)

    print("✅ Cleaned label values:", df['label'].unique())

    # ✅ Train/Test Split
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Vectorize with TF-IDF
    print("🧠 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ✅ Train Logistic Regression
    print("📊 Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # ✅ Evaluate
    print("\n🔍 Evaluating model...")
    y_pred = model.predict(X_test_vec)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ✅ Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("💾 Model and vectorizer saved in 'models/' folder.")
