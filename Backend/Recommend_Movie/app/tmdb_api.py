import requests

TMDB_API_KEY = "2ebe4a9d7359a4b32ad7b87397279dff"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_movie_id(title):
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if data['results']:
        return data['results'][0]['id']
    return None

def get_recommendations(title):
    movie_id = get_movie_id(title)
    if not movie_id:
        return []

    url = f"{TMDB_BASE_URL}/movie/{movie_id}/recommendations"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    recommendations = []
    for movie in data.get('results', []):
        recommendations.append({
            "title": movie['title'],
            "overview": movie['overview'],
            "poster_path": f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if movie.get('poster_path') else None
        })
    return recommendations