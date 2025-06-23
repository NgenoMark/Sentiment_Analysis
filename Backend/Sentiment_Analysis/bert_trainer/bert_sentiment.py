from transformers import pipeline

# Load a sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

print("📝 Enter a movie review:")
text = input("Review: ")

# Get prediction
result = classifier(text)[0]

# Format result
label = result['label']
score = round(result['score'] * 100, 2)

print(f"\n🎬 Sentiment: {label} ({score}%)")
