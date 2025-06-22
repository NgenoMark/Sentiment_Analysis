from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf
import numpy as np

# ===============================
# 🔧 Load Model & Tokenizer
# ===============================
model_path = "../models/bert_sentiment_model"  # Folder where tf_model.h5, config.json etc. are
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# ===============================
# ✏️ Predict Sentiment Function
# ===============================
def predict_sentiment(review_text):
    inputs = tokenizer(review_text, return_tensors="tf", truncation=True, padding=True, max_length=256)
    outputs = model(inputs)
    prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]
    return "Positive" if prediction == 1 else "Negative"

# ===============================
# 📥 Input and Predict
# ===============================
while True:
    review = input("\n📝 Enter your review (or type 'exit' to quit):\n> ")
    if review.lower() == "exit":
        break
    sentiment = predict_sentiment(review)
    print(f"✅ Sentiment: {sentiment}")
