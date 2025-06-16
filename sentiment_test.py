# test_sentiment.py
import os

import pandas as pd
import joblib
import re

# -------- Load model and vectorizer --------
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# -------- Text cleaning function (same as used during training) --------
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))        # Remove HTML
    text = re.sub(r'[^a-zA-Z]', ' ', text)        # Keep letters only
    return text.lower()

# -------- Load and clean new test data --------
test_df = pd.read_csv("data/sentiment/sentiment - IMDb Movie Review Sentiment/test.csv")  # Change path to your file

if 'text' not in test_df.columns:
    raise ValueError("Your test file must contain a 'text' column.")

# Optional: if there's a label column, use it later for accuracy
has_labels = 'label' in test_df.columns

test_df['cleaned'] = test_df['text'].apply(clean_text)

# -------- Transform and predict --------
X_test_vec = vectorizer.transform(test_df['cleaned'])
predictions = model.predict(X_test_vec)

test_df['predicted_sentiment'] = predictions

# -------- Optional: Compare with true labels --------
if has_labels:
    label_map = {1: 'positive', 0: 'negative'}
    test_df['label'] = test_df['label'].map(label_map).astype(str)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_df['label'], test_df['predicted_sentiment'])
    print("✅ Test Accuracy on new data:", accuracy)

# -------- Show sample predictions --------
print("\n🧾 Sample Predictions:")
print(test_df[['text', 'predicted_sentiment']].head(10))

# -------- Save to CSV --------
output_path = "data/sentiment/sentiment - IMDb Movie Review Sentiment/sorted_test.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
test_df[['text', 'predicted_sentiment']].to_csv(output_path, index=False)
print(f"💾 Predictions saved to {output_path}")