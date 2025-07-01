from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tmdb_api import get_poster_url

app = Flask(__name__)

# Load model, dataset, and embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('Data/movie_data_with_embeddings.csv')
embeddings = np.load('Data/movie_embeddings.npy')

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

if __name__ == "__main__":
    app.run(debug=True)
