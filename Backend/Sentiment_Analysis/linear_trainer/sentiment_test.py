# test_sentiment.py

import os
import pandas as pd
import joblib
import re
from sklearn.metrics import accuracy_score

# -------- Load model and vectorizer --------
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# -------- Text cleaning function (same as training) --------
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

# -------- Load CSV Test Data --------
csv_path = "../data/sentiment/Test/sentiment - IMDb Movie Review Sentiment test.csv"
print(f"📥 Loading CSV test data from: {csv_path}")
csv_df = pd.read_csv(csv_path)

if 'text' not in csv_df.columns:
    raise ValueError("CSV test file must contain a 'text' column.")

has_labels = 'label' in csv_df.columns
csv_df['cleaned'] = csv_df['text'].apply(clean_text)
csv_df['source'] = 'csv'

# -------- Load Text Files from Folders --------
def load_text_folder(base_folder):
    all_reviews = []

    for label_folder in ['pos', 'neg']:
        full_path = os.path.join(base_folder, label_folder)
        label = 'positive' if label_folder == 'pos' else 'negative'

        for filename in os.listdir(full_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(full_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        all_reviews.append({
                            'text': content,
                            'cleaned': clean_text(content),
                            'label': label,
                            'source': 'txt'
                        })
    return pd.DataFrame(all_reviews)

txt_df = load_text_folder("../data/Sentiment/Test")
print(f"📥 Loaded {len(txt_df)} reviews from text files.")

# -------- Combine both sources --------
combined_df = pd.concat([csv_df, txt_df], ignore_index=True)

# -------- Predict --------
X_vec = vectorizer.transform(combined_df['cleaned'])
predictions = model.predict(X_vec)
combined_df['predicted_sentiment'] = predictions

# -------- Optional Accuracy --------
if has_labels or 'label' in combined_df.columns:
    # Convert numeric 0/1 labels to strings
    combined_df['label'] = combined_df['label'].map({1: 'positive', 0: 'negative'}).fillna(combined_df['label'])
    combined_df['label'] = combined_df['label'].astype(str).str.lower()

    combined_df['predicted_sentiment'] = combined_df['predicted_sentiment'].astype(str).str.lower()

    acc = accuracy_score(combined_df['label'], combined_df['predicted_sentiment'])
    print(f"✅ Accuracy on combined test set: {acc:.4f}")


# -------- Show Sample Predictions --------
print("\n🧾 Sample Predictions:")
print(combined_df[['text', 'predicted_sentiment']].head(10))

# -------- Save Results --------
output_path = "../data/sentiment/Test/test_predictions_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
combined_df.to_csv(output_path, index=False)
print(f"💾 All predictions saved to: {output_path}")
