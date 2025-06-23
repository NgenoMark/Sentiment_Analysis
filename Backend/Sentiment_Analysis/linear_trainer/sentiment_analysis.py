import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------- Text Cleaning Function --------
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Keep letters only
    return text.lower()


# -------- Load CSV Dataset --------
def load_csv_data():
    csv_path = os.path.join("..", "data", "sentiment","Train", "sentiment - IMDb Movie Review Sentiment train.csv")
    print("📥 Loading CSV data from:", csv_path)

    df = pd.read_csv(csv_path)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    df = df.dropna(subset=['text', 'label'])

    # Convert 0/1 to string labels
    label_map = {1: 'positive', 0: 'negative'}
    df['label'] = df['label'].map(label_map)

    df['text'] = df['text'].apply(clean_text)

    print(f"✅ CSV samples loaded: {len(df)}")
    return df[['text', 'label']]


# -------- Load Folder-based Text Files --------
def load_text_files_from_folder():
    base_folder = os.path.join("..", "data","Sentiment","Train")
    print("📥 Loading .txt files from:", base_folder)

    reviews = []

    for label_folder in ['pos', 'neg']:
        full_path = os.path.join(base_folder, label_folder)
        sentiment = 'positive' if label_folder == 'pos' else 'negative'
        count = 0

        for filename in os.listdir(full_path):
            file_path = os.path.join(full_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        cleaned = clean_text(content)
                        reviews.append({'text': cleaned, 'label': sentiment})
                        count += 1
        print(f"   📄 {sentiment.title()} reviews loaded: {count}")

    return pd.DataFrame(reviews)


# -------- Main Training Function --------
def train_sentiment_model():
    print("🚀 Starting sentiment training process...")

    # ✅ Load data from CSV and .txt folders
    csv_df = load_csv_data()
    txt_df = load_text_files_from_folder()

    # ✅ Combine all data
    full_df = pd.concat([csv_df, txt_df], ignore_index=True)
    print(f"📊 Total combined samples: {len(full_df)}")

    # ✅ Train/test split
    X = full_df['text']
    y = full_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Vectorize
    print("🧠 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ✅ Train model
    print("📈 Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # ✅ Evaluate
    print("\n🔍 Evaluation Results:")
    y_pred = model.predict(X_test_vec)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("🧾 Classification Report:\n", classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ✅ Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("💾 Model and vectorizer saved to /models/")


if __name__ == "__main__":
    train_sentiment_model()

