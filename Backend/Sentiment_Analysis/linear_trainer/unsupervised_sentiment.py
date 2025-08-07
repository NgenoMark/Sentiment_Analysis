import os
import pandas as pd
import joblib
import re

# -------- Load model and vectorizer --------
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# -------- Text cleaning function --------
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

# -------- Load unlabeled CSV data --------
def load_csv_data(csv_path):
    print(f"📥 Loading unlabeled CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")
    df['cleaned'] = df['text'].apply(clean_text)
    df['source'] = 'csv'
    return df

# -------- Load .txt files from folder --------
def load_text_folder(folder_path):
    print(f"📥 Loading text files from: {folder_path}")
    reviews = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            full_path = os.path.join(folder_path, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    reviews.append({'text': content, 'cleaned': clean_text(content), 'source': 'txt'})
    return pd.DataFrame(reviews)

# -------- Predict Sentiment --------
def predict_sentiment(df):
    X_vec = vectorizer.transform(df['cleaned'])
    df['predicted_sentiment'] = model.predict(X_vec)
    return df

# -------- Paths --------
csv_path = "../data/sentiment/Unsupervised/sentiment - IMDb Movie Review Sentiment unsupervised.csv"
txt_folder = "../data/sentiment/Unsupervised/unsup"

# -------- Load data --------
csv_df = load_csv_data(csv_path)
txt_df = load_text_folder(txt_folder)

# -------- Combine & Predict --------
combined_df = pd.concat([csv_df, txt_df], ignore_index=True)
result_df = predict_sentiment(combined_df)

# -------- Save output --------
output_path = "../data/sentiment/Unsupervised/unsupervised_predictions.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result_df[['text', 'predicted_sentiment', 'source']].to_csv(output_path, index=False)
print(f"✅ Predictions saved to: {output_path}")
