import os
import json
import subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import psutil
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast

# ========== Load Sentiment Model ==========
model_path = "../models/bert_sentiment_model"
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# ========== Dynamic Sample Size Based on System ==========
def get_sample_size():
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_count = psutil.cpu_count(logical=False)
    if ram_gb < 4:
        return 10
    elif ram_gb < 6 or cpu_count < 4:
        return 20
    return 50

# ========== Generate Recommendations ==========
def generate_feedback(prompt):
    try:
        process = subprocess.Popen(
            ["ollama", "run", "mistral"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        full_prompt = (
            "You are a movie review analyst assistant. Based on the following audience reviews, "
            "summarize major themes, praises, and issues. Then generate actionable recommendations "
            "for the movie director to improve future films.\n\n"
            + prompt
        )

        start = time.time()
        stdout, stderr = process.communicate(input=full_prompt.encode("utf-8"), timeout=300)
        elapsed = time.time() - start

        if process.returncode != 0:
            print(f"❌ Ollama error:\n{stderr.decode('utf-8')}")
            return "⚠️ Failed to generate recommendations.", elapsed

        return stdout.decode("utf-8").strip(), elapsed

    except subprocess.TimeoutExpired:
        process.kill()
        return "⚠️ Ollama timed out while generating response.", 0

# ========== Analyze Movie Reviews ==========
def analyze_movie_reviews(movie_title):
    file_path = f"../data/review_recommend/movie_reviews_data/{movie_title}/{movie_title}_review.csv"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    if "text" not in df.columns or "label" not in df.columns:
        print("❌ CSV must contain 'text' and 'label' columns.")
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

    # ========== Confidence Scoring ==========
    def get_confidence(text):
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=256)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
        return np.max(probs)

    print("\n⚙️ Calculating confidence scores...")
    scoring_start = time.time()
    df["confidence"] = df["text"].apply(get_confidence)
    scoring_end = time.time()
    print(f"🕒 Confidence scoring completed in {scoring_end - scoring_start:.2f} seconds.")

    # ========== Top Reviews ==========
    print("\n🔝 Top 10 Negative Reviews:")
    for _, row in df[df["label"] == 0].sort_values(by="confidence", ascending=False).head(10).iterrows():
        print(f"- ({row['confidence']*100:.2f}%) {row['text'][:120]}")

    print("\n🔝 Top 10 Positive Reviews:")
    for _, row in df[df["label"] == 1].sort_values(by="confidence", ascending=False).head(10).iterrows():
        print(f"- ({row['confidence']*100:.2f}%) {row['text'][:120]}")

    # ========== Sample and Generate Feedback ==========
    sample_size = min(get_sample_size(), len(df))
    print(f"\n🧠 Generating director recommendations from {sample_size} sampled reviews (adaptive)...")
    sample_texts = df.sample(n=sample_size)["text"].tolist()
    joined_sample = "\n".join(sample_texts)

    feedback, feedback_time = generate_feedback(joined_sample)
    print(f"\n🕒 Feedback generation time: {feedback_time:.2f} seconds.")
    print("\n📋 Recommendations:")
    print(feedback)

    # ========== Save Report ==========
    output_dir = f"../data/review_recommend/movie_reviews_data/{movie_title}"
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{movie_title}_report.txt")

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
    print(f"✅ Total Runtime: {scoring_end - scoring_start + feedback_time:.2f} seconds.")

# ========== Entry Point ==========
if __name__ == "__main__":
    movie_title = input("🎥 Enter the movie title to analyze: ").strip()
    analyze_movie_reviews(movie_title)
