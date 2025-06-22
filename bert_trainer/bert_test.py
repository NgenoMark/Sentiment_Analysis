import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# === Step 1: Load model & tokenizer ===
model_dir = "../models/bert_sentiment_model"  # path to your trained model folder
model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# === Step 2: Load CSV test data ===
csv_path = "../data/sentiment/Test/sentiment - IMDb Movie Review Sentiment test.csv"
df_csv = pd.read_csv(csv_path).dropna()
df_csv["label"] = df_csv["label"].astype(int)

# === Step 3: Load .txt test files ===
def load_text_files(folder_path):
    texts, labels = [], []
    for label_folder in ["pos", "neg"]:
        label = 1 if label_folder == "pos" else 0
        folder = os.path.join(folder_path, label_folder)
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if filename.endswith(".txt"):
                with open(file_path, encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels})

txt_folder = "data/sentiment/Test"
df_txt = load_text_files(txt_folder)

# === Step 4: Combine CSV and .txt test data ===
df_test = pd.concat([df_csv, df_txt]).dropna().reset_index(drop=True)

# === Step 5: Tokenize and predict ===
texts = df_test["text"].tolist()
true_labels = df_test["label"].tolist()

encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="tf")
outputs = model(encodings)

pred_probs = outputs.logits.numpy()
preds = np.argmax(pred_probs, axis=1)

# === Step 6: Evaluation ===
accuracy = accuracy_score(true_labels, preds)
report = classification_report(true_labels, preds, target_names=["Negative", "Positive"])
conf_matrix = confusion_matrix(true_labels, preds)

# === Step 7: Display results ===
print("✅ Sentiment Analysis Test Results")
print("-" * 50)
print(f"🔢 Accuracy: {accuracy * 100:.2f}%\n")
print("📊 Classification Report:")
print(report)
print("🧮 Confusion Matrix:")
print(conf_matrix)
