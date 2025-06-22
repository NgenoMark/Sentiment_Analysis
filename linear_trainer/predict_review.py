import re
import joblib

# -------- Load model and vectorizer --------
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# -------- Text cleaning function --------
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

# -------- Get user input --------
print("📝 Enter a movie review below:")
user_input = input("Review: ")

# -------- Clean and transform input --------
cleaned = clean_text(user_input)
vec = vectorizer.transform([cleaned])

# -------- Predict --------
prediction = model.predict(vec)[0]

# -------- Output --------
print(f"\n🎬 Sentiment Prediction: {prediction.upper()} ✅")
