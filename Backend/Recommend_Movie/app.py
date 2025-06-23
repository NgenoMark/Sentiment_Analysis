from flask import Flask, request, jsonify, render_template
import requests
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

TMDB_API_KEY = "2ebe4a9d7359a4b32ad7b87397279dff"

def get_movie_id(title):
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(url, params=params)
    data = response.json()
    return data['results'][0]['id'] if data['results'] else None

def get_genre_map():
    url = f"https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)
    genres = response.json().get("genres", [])
    return {genre["id"]: genre["name"] for genre in genres}

def get_similar_movies(title):
    movie_id = get_movie_id(title)
    if not movie_id:
        return []

    genre_map = get_genre_map()

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    return [
        {
            "title": movie["title"],
            "overview": movie["overview"],
            "genres": [genre_map.get(genre_id, "Unknown") for genre_id in movie.get("genre_ids", [])]
        }
        for movie in data.get("results", [])
    ]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    title = request.args.get("title", "")
    recommendations = get_similar_movies(title)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
