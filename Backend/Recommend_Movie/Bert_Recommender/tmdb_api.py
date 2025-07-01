import requests
import os

# Store your API key here or load from env for security
TMDB_API_KEY = "2ebe4a9d7359a4b32ad7b87397279dff"

def get_poster_url(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        results = data.get("results", [])

        if results and results[0].get("poster_path"):
            poster_path = results[0]["poster_path"]
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"[TMDb ERROR]: {e}")
    
    # Fallback image path
    return "/static/ImageNotFound.png"
