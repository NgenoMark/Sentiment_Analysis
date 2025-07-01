# ollama run mistral
# ollama run llama3

import os
import pandas as pd
import numpy as np
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf
from mistralai.client import MistralClient
from datetime import datetime

# ========================
# 🔧 Load Model & Tokenizer
# ========================
model_path = "../models/bert_sentiment_model"
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# ========================
# 📊 Analyze Movie Reviews
# ========================
def analyze_movie_reviews(movie_title):
    file_path = f"../data/review_recommend/movie_reviews_data/{movie_title}/{movie_title}_review.csv"
    if not os.path.exists(file_path):
        print(f"❌ Error: CSV file not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    if "text" not in df.columns or "label" not in df.columns:
        print("❌ CSV must have 'text' and 'label' columns.")
        return

    total_reviews = len(df)
    pos_reviews = df[df["label"] == 1]
    neg_reviews = df[df["label"] == 0]
    pos_pct = len(pos_reviews) / total_reviews * 100
    neg_pct = len(neg_reviews) / total_reviews * 100

    print(f"\n🎬 Sentiment Summary for '{movie_title}'")
    print(f"Total Reviews: {total_reviews}")
    print(f"✅ Positive: {pos_pct:.2f}%")
    print(f"❌ Negative: {neg_pct:.2f}%")

    # Top reviews (based on prediction confidence)
    def get_confidence(text):
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=256)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
        return np.max(probs)

    df["confidence"] = df["text"].apply(get_confidence)
    print("\n🔝 Top 10 Negative Reviews:")
    for i, row in df[df["label"] == 0].sort_values(by="confidence", ascending=False).head(10).iterrows():
        print(f"- ({row['confidence']*100:.2f}%) {row['text'][:120]}")

    print("\n🔝 Top 10 Positive Reviews:")
    for i, row in df[df["label"] == 1].sort_values(by="confidence", ascending=False).head(10).iterrows():
        print(f"- ({row['confidence']*100:.2f}%) {row['text'][:120]}")

    # ==============================
    # 💡 Generate Recommendations
    # ==============================
    sample_texts = df.sample(n=min(50, len(df)))['text'].tolist()
    joined_sample = "\n".join(sample_texts)

    system_prompt = (
        "You are a movie review analyst assistant. Based on the following audience reviews, summarize major themes, "
        "praises, and issues. Then generate actionable recommendations for the movie director to improve future films."
    )

    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": joined_sample}
    ]

    print("\n🧠 Generating director recommendations...")
    response = client.chat(model="mistral-small", messages=messages)
    feedback = response.choices[0].message.content

    print("\n📋 Recommendations:")
    print(feedback)

    # ==============================
    # 📝 Save Results
    # ==============================
    result_path = f"movie_reviews_data/{movie_title}/{movie_title}_report.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Sentiment Summary:\nPositive: {pos_pct:.2f}%\nNegative: {neg_pct:.2f}%\n\n")
        f.write("Top 10 Positive Reviews:\n")
        for row in df[df['label'] == 1].sort_values(by="confidence", ascending=False).head(10).itertuples():
            f.write(f"({row.confidence*100:.2f}%) {row.text}\n")
        f.write("\nTop 10 Negative Reviews:\n")
        for row in df[df['label'] == 0].sort_values(by="confidence", ascending=False).head(10).itertuples():
            f.write(f"({row.confidence*100:.2f}%) {row.text}\n")
        f.write("\nRecommendations:\n")
        f.write(feedback)

    print(f"\n📁 Analysis saved to: {result_path}")

# ========================
# ▶️ Run
# ========================
if __name__ == "__main__":
    movie_title = input("🎥 Enter the movie title to analyze: ").strip()
    analyze_movie_reviews(movie_title)
