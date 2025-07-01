import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

# Load saved vectorizer and TF-IDF matrix
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = joblib.load(f)

with open("models/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = joblib.load(f)

# Load movie data
df = pd.read_csv("models/movie_data.csv")
df['title_lower'] = df['title'].str.lower()
df['combined'] = df['title'] + ' ' + df['genres']  # same as vectorizer input

# Fuzzy match to correct title input
def get_closest_title(user_input):
    best_match = process.extractOne(user_input.lower(), df['title_lower'].tolist(), score_cutoff=60)
    return best_match[0] if best_match else None

# Generate recommendations
def recommend(title, top_n=5):
    closest = get_closest_title(title)
    if not closest:
        return []

    idx = df[df['title_lower'] == closest].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    scores = list(enumerate(cosine_sim[0]))

    # Optional: filter by shared genres (for higher quality)
    input_genres = set(df.iloc[idx]['genres'].split())
    filtered_scores = [
        (i, score) for i, score in scores
        if i != idx and input_genres & set(df.iloc[i]['genres'].split())
    ]

    top_indices = [
        i for i, _ in sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:top_n]
    ]
    return df.iloc[top_indices][['title', 'genres']].to_dict(orient="records")
