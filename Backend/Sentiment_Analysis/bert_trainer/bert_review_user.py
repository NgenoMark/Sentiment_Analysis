from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf
import numpy as np

# ===============================
# 🔧 Load Model & Tokenizer
# ===============================
model_path = "../../models/bert_sentiment_model"
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# ===============================
# ✏️ Predict Sentiment Function
# ===============================
def predict_sentiment(review_text):
    inputs = tokenizer(review_text, return_tensors="tf", truncation=True, padding=True, max_length=256)
    outputs = model(inputs)

    # Convert logits to probabilities
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class]

    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment, confidence * 100  # Convert to percentage

# ===============================
# 📥 Input and Predict
# ===============================
while True:
    review = input("\n📝 Enter your review (or type 'exit' to quit):\n> ")
    if review.lower() == "exit":
        break
    sentiment, confidence = predict_sentiment(review)
    print(f"✅ Sentiment: {sentiment} ({confidence:.2f}% confidence)")
