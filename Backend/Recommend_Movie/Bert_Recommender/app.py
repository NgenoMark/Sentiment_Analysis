from flask import Flask, request, jsonify, render_template, url_for
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import requests
from tmdb_api import get_poster_url
import os
import csv
import re
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast


app = Flask(__name__)

# ========== Load Model, Data and Embeddings ==========
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('bert_data/movie_data_with_embeddings.csv')
embeddings = np.load('bert_data/movie_embeddings.npy')

# Load sentiment model and tokenizer
model_path = "../../models/bert_sentiment_model/"
sentiment_model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# ========== Semantic Search ==========
def semantic_search(query, top_n=5):
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        title = row.get("movie_name", "N/A")
        poster_url = get_poster_url(title)

        results.append({
            "title": title,
            "score": round(float(similarity_scores[idx]), 3),
            "genre": row.get("genre", "N/A"),
            "overview": row.get("description", "N/A"),
            "director": row.get("director", "N/A"),
            "star": row.get("star", "N/A"),
            "poster": poster_url
        })
    return results


#===================Comment analyser and Storage========================#
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=256)
    outputs = sentiment_model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    label = int(np.argmax(probs))  # 0 for negative, 1 for positive
    return label


def sanitize_name(name):
    name = name.lower().replace(" ", "_")
    return re.sub(r'[^a-z0-9_]', '', name)

def store_comment(movie_name, comment, label, base_path="../../data/review_recommend/movie_reviews_data/"):
    safe_name = sanitize_name(movie_name)
    folder_path = os.path.join(base_path, safe_name)
    os.makedirs(folder_path, exist_ok=True)

    index = 1
    while True:
        file_name = f"{safe_name}_review{'' if index == 1 else index}.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if len(df) < 5000:
                break
            index += 1
        else:
            df = pd.DataFrame(columns=["text", "label"])
            break

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if os.stat(file_path).st_size == 0:
            writer.writerow(["text", "label"])
        writer.writerow([comment, label])


# ========== Routes ==========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    results = semantic_search(query)
    return jsonify(results)

@app.route("/movie/<title>")
def movie_detail(title):
    decoded_title = title.replace("%20", " ")
    row = df[df["movie_name"].str.lower() == decoded_title.lower()].head(1)
    if row.empty:
        return "Movie not found", 404

    movie = row.iloc[0].to_dict()
    movie["poster"] = get_poster_url(movie["movie_name"])

    # Fetch trailer from YouTube API
    trailer_url = url_for("get_trailer", movie_title=decoded_title, _external=True)
    trailer_resp = requests.get(trailer_url)
    trailer_id = trailer_resp.json().get("videoId") if trailer_resp.ok else None

    # Fetch similar movies
    similar_url = url_for("similar_movies", movie_title=decoded_title, _external=True)
    similar_resp = requests.get(similar_url)
    similar_movies = similar_resp.json() if similar_resp.ok else []

    return render_template("movie_detail.html", movie=movie, trailer_id=trailer_id, similar_movies=similar_movies)


@app.route("/comment", methods=["POST"])
def post_comment():
    data = request.get_json()
    comment = data.get("comment", "").strip()
    movie_name = data.get("movie_name", "").strip()

    if not comment or not movie_name:
        return jsonify({"error": "Missing comment or movie name"}), 400

    label = predict_sentiment(comment)
    store_comment(movie_name, comment, label)

    return jsonify({"message": "Comment posted successfully", "label": label})




# ========== YouTube API Integration ==========
YOUTUBE_API_KEY =  "AIzaSyCdXAt28SWhQEtDqoF2a01Zpw19BYW_DMY"  # Replace securely

@app.route("/trailer/<movie_title>")
def get_trailer(movie_title):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"{movie_title} official trailer",
        "key": YOUTUBE_API_KEY,
        "maxResults": 1,
        "type": "video",
        "videoEmbeddable": "true"
    }
    response = requests.get(search_url, params=params)
    data = response.json()

    if data.get("items"):
        video_id = data["items"][0]["id"]["videoId"]
        return jsonify({"videoId": video_id})
    return jsonify({"error": "Trailer not found"}), 404

# ========== Similar Movies ==========
@app.route("/similar/<movie_title>")
def similar_movies(movie_title):
    try:
        movie_index = df[df['movie_name'].str.lower() == movie_title.lower()].index[0]
        movie_vector = embeddings[movie_index].reshape(1, -1)
        similarities = cosine_similarity(movie_vector, embeddings)[0]

        similar_indices = np.argsort(similarities)[::-1][1:6]
        results = []
        for idx in similar_indices:
            row = df.iloc[idx]
            results.append({
                "movie_name": row["movie_name"],
                "poster": get_poster_url(row["movie_name"])
            })
        return jsonify(results)
    except IndexError:
        return jsonify([])

# ========== Run Server ==========
if __name__ == "__main__":
    app.run(debug=True)
