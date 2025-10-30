# scripts/train_sentiment.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import json
import os


class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return enc


def map_sentiment(text):
    text = text.lower()
    if any(word in text for word in ["good", "love", "great", "awesome", "excellent", "happy", "satisfied"]):
        return 2  # positive
    elif any(word in text for word in ["bad", "poor", "worst", "hate", "angry", "disappointed"]):
        return 0  # negative
    else:
        return 1  # neutral


def main(args):
    df = pd.read_csv(args.data)
    if 'label' not in df.columns:
        df['label'] = df['text_clean'].apply(map_sentiment)

    X_train, X_val, y_train, y_val = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    train_dataset = FeedbackDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_dataset = FeedbackDataset(X_val.tolist(), y_val.tolist(), tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        learning_rate=5e-5,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    # Evaluate
    preds = trainer.predict(val_dataset)
    y_pred = preds.predictions.argmax(axis=1)
    report = classification_report(y_val, y_pred, target_names=["Negative", "Neutral", "Positive"], output_dict=True)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    json.dump(report, open(os.path.join(args.out_dir, 'eval_report.json'), 'w'), indent=2)

    print("Model and tokenizer saved to:", args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cleaned feedback CSV")
    parser.add_argument("--out_dir", default="models/sentiment_model")
    args = parser.parse_args()
    main(args)
