# ======== app_admin.py (Admin Dashboard Server) ========
import os
import re
import string
import requests
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv
from difflib import get_close_matches

# ======== Load Env Vars ========
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# ======== Flask App Init ========
app = Flask(__name__)

# ======== Utility Functions ========
def sanitize_name(name):
    name = name.lower().replace(" ", "_")
    return re.sub(r'[^a-z0-9_]', '', name)

def guess_original_title(sanitized_folder_name):
    name = sanitized_folder_name.replace("__", " ").replace("_", " ")
    name = name.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(word.capitalize() for word in name.split())

def get_tmdb_movie_details(title):
    search_url = "https://api.themoviedb.org/3/search/movie"
    search_params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }

    try:
        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()
        results = search_data.get("results", [])

        if not results:
            raise Exception("No results from TMDB.")

        movie = results[0]
        movie_id = movie["id"]

        # Get credits
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
        credits_params = {"api_key": TMDB_API_KEY}
        credits_response = requests.get(credits_url, params=credits_params)
        credits = credits_response.json()

        director = "N/A"
        stars = []

        for person in credits.get("crew", []):
            if person["job"] == "Director":
                director = person["name"]
                break

        for actor in credits.get("cast", [])[:3]:  # Top 3 stars
            stars.append(actor["name"])

        return {
            "title": movie.get("title", title),
            "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else "/static/ImageNotFound.png",
            "description": movie.get("overview", "No description available."),
            "genre": ", ".join([str(g) for g in movie.get("genre_ids", [])]) or "N/A",
            "director": director,
            "star": ", ".join(stars) if stars else "N/A"
        }

    except Exception as e:
        print("TMDB lookup failed:", e)
        return {
            "title": title,
            "poster": "/static/ImageNotFound.png",
            "description": "No description available.",
            "genre": "N/A",
            "director": "N/A",
            "star": "N/A"
        }


def get_folder_for_title(input_title):
    base_dir = "../data/review_recommend/movie_reviews_data/"
    input_sanitized = sanitize_name(input_title)

    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    sanitized_map = {sanitize_name(folder): folder for folder in folders}

    if input_sanitized in sanitized_map:
        return sanitized_map[input_sanitized]

    guessed_map = {sanitize_name(guess_original_title(f)): f for f in folders}
    matches = get_close_matches(input_sanitized, guessed_map.keys(), n=1, cutoff=0.6)
    if matches:
        return guessed_map[matches[0]]
    return None

# ======== Routes ========
@app.route("/")
def admin_dashboard():
    movie_base_dir = "../data/review_recommend/movie_reviews_data/"
    movies = []

    if os.path.exists(movie_base_dir):
        print("✅ Review base directory exists.")
        for folder in sorted(os.listdir(movie_base_dir)):
            folder_path = os.path.join(movie_base_dir, folder)
            print(f"Found folder: {folder}")

            if os.path.isdir(folder_path):
                safe_name = sanitize_name(folder)
                review_file = os.path.join(folder_path, f"{safe_name}_review.csv")

                if os.path.exists(review_file):
                    guessed_title = guess_original_title(folder)
                    details = get_tmdb_movie_details(guessed_title)
                    details["folder"] = folder  # track original folder
                    movies.append(details)
                else:
                    print(f"⚠️ No review file found in: {folder_path}")
            else:
                print(f"❌ Skipped (not a directory): {folder_path}")
    else:
        print("❌ Base directory not found!")

    print(f"\nFinal loaded titles: {[m['title'] for m in movies]}\n")
    return render_template("admin_dashboard.html", movie_cards=movies)

@app.route("/admin/search", methods=["POST"])
def search_movie():
    movie_title = request.form.get("movie_title", "").strip()
    if not movie_title:
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("movie_detail", movie_title=movie_title))

@app.route("/admin/movie/<movie_title>")
def movie_detail(movie_title):
    folder = get_folder_for_title(movie_title)
    if not folder:
        return render_template("movie_analysis.html", error="Movie not found.")

    guessed_title = guess_original_title(folder)
    movie = get_tmdb_movie_details(guessed_title)
    movie['movie_name'] = guessed_title
    return render_template("movie_analysis.html", movie=movie)

@app.route("/admin/analyze/<movie_title>")
def analyze_movie_reviews(movie_title):
    from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
    import numpy as np
    import tensorflow as tf
    import time
    import subprocess
    import psutil
    import os

    def get_sample_size():
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count(logical=False)
        if ram_gb < 4:
            return 10
        elif ram_gb < 6 or cpu_count < 4:
            return 20
        return 50

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

            stdout, stderr = process.communicate(input=full_prompt.encode("utf-8"), timeout=300)

            if process.returncode != 0:
                print("Ollama Error:", stderr.decode())
                return "⚠️ Failed to generate feedback."

            return stdout.decode().strip()

        except subprocess.TimeoutExpired:
            process.kill()
            return "⚠️ Ollama timed out."

    folder = get_folder_for_title(movie_title)
    if not folder:
        return jsonify({"error": "No matching folder found for this movie."}), 404

    safe_folder = sanitize_name(folder)
    file_path = f"../data/review_recommend/movie_reviews_data/{folder}/{safe_folder}_review.csv"

    if not os.path.exists(file_path):
        return jsonify({"error": "No reviews found for this movie."}), 404

    model_path = "../models/bert_sentiment_model/"
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    df = pd.read_csv(file_path)
    if "text" not in df.columns or "label" not in df.columns:
        return jsonify({"error": "CSV must have 'text' and 'label' columns."}), 400

    total_reviews = len(df)
    pos_reviews = df[df["label"] == 1]
    neg_reviews = df[df["label"] == 0]

    pos_pct = round(len(pos_reviews) / total_reviews * 100, 2)
    neg_pct = round(len(neg_reviews) / total_reviews * 100, 2)

    def get_confidence(text):
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=256)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
        return np.max(probs)

    # Batch inference for confidence scores
    def batch_confidences(texts, batch_size=32):
        all_confidences = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts.tolist(), return_tensors="tf", padding=True, truncation=True, max_length=256)
            outputs = model(inputs)
            probs = tf.nn.softmax(outputs.logits, axis=1).numpy()
            batch_confidences = probs.max(axis=1).tolist()
            all_confidences.extend(batch_confidences)
        return all_confidences

    # Apply batching
    df["confidence"] = batch_confidences(df["text"], batch_size=32)

    # df["confidence"] = df["text"].apply(get_confidence)

    # top_positive = df[df["label"] == 1].sort_values(by="confidence", ascending=False)["text"].head(5).tolist()
    # top_negative = df[df["label"] == 0].sort_values(by="confidence", ascending=False)["text"].head(5).tolist()

    top_positive = (
        df[df["label"] == 1]
        .drop_duplicates(subset="text")
        .sort_values(by="confidence", ascending=False)["text"]
        .head(5)
        .tolist()
    )

    top_negative = (
        df[df["label"] == 0]
        .drop_duplicates(subset="text")
        .sort_values(by="confidence", ascending=False)["text"]
        .head(5)
        .tolist()
    )

    # Sample for feedback
    sample_size = min(get_sample_size(), len(df))
    # sample_texts = df.sample(n=sample_size)["text"].tolist()
    sample_texts = df["text"].drop_duplicates().sample(n=min(10, len(df)))[:10].tolist()
    joined_sample = "\n".join(sample_texts)
    # feedback = "⚠️ Feedback skipped for testing."

    feedback = generate_feedback(joined_sample)

    return jsonify({
        "total_reviews": total_reviews,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "feedback": feedback,
        "poster": get_tmdb_movie_details(guess_original_title(folder)).get("poster")
    })

# ======== Run ========
if __name__ == "__main__":
    app.run(debug=True)
