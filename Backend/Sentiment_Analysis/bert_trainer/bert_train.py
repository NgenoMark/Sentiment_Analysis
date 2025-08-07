import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# -------- Load CSV --------
csv_path = "../data/sentiment/Train/sentiment - IMDb Movie Review Sentiment train.csv"
print(f"📥 Loading CSV data from: {csv_path}")
csv_df = pd.read_csv(csv_path)
csv_df = csv_df.dropna(subset=["text", "label"])
csv_df['label'] = csv_df['label'].astype(int)

# -------- Load TXT files --------
def load_text_files(folder_path):
    texts, labels = [], []
    for label_folder in ["pos", "neg"]:
        label = 1 if label_folder == "pos" else 0
        folder = os.path.join(folder_path, label_folder)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if filename.endswith(".txt"):
                with open(file_path, encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels})

txt_folder = "../data/sentiment/Train"
print(f"📥 Loading TXT data from: {txt_folder}")
txt_df = load_text_files(txt_folder)

# -------- Combine --------
df = pd.concat([csv_df, txt_df]).dropna().reset_index(drop=True)
print(f"✅ Total training samples: {len(df)}")

# -------- Tokenization --------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=256)

# -------- Dataset Preparation --------
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -------- Model --------
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# -------- Trainer --------
training_args = TrainingArguments(
    output_dir="models/bert_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# -------- Train --------
print("🚀 Training BERT sentiment model...")
trainer.train()

# -------- Save --------
trainer.save_model("../models/bert_sentiment")
tokenizer.save_pretrained("../models/bert_sentiment")
print("✅ Model and tokenizer saved to 'models/bert_sentiment'")
