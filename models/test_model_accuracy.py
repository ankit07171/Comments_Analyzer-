"""
Test Model Accuracy - Evaluate Custom Trained Models

This script tests the custom sentiment and intent models on real-world
comment data to verify their accuracy and provide transparency for
authorization teams.

OUTPUT: Detailed accuracy report with per-class metrics
"""

import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re

os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------------
# LOAD SENTIMENT MODEL
# ------------------------------------------------------------------

class SentimentClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_sentiment_model():
    """Load trained sentiment model"""
    with open("models/sentiment_labels.json", "r") as f:
        label_mapping = json.load(f)
    
    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    
    model = SentimentClassifier(num_labels=3)
    model.load_state_dict(torch.load("models/sentiment_model.pt", map_location="cpu"))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("models/sentiment_tokenizer")
    
    return model, tokenizer, idx_to_label


def predict_sentiment(model, tokenizer, text, idx_to_label):
    """Predict sentiment for a single text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    encoding = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=128, return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    return {
        "sentiment": idx_to_label[pred],
        "confidence": round(confidence, 3),
        "probabilities": {
            "negative": round(probs[0][0].item(), 3),
            "neutral": round(probs[0][1].item(), 3),
            "positive": round(probs[0][2].item(), 3)
        }
    }


# ------------------------------------------------------------------
# LOAD INTENT MODEL
# ------------------------------------------------------------------

class IntentClassifier(nn.Module):
    def __init__(self, num_labels=9):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_intent_model():
    """Load trained intent model"""
    with open("models/intent_labels.json", "r") as f:
        label_mapping = json.load(f)
    
    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    
    model = IntentClassifier(num_labels=9)
    model.load_state_dict(torch.load("models/intent_model.pt", map_location="cpu"))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("models/intent_tokenizer")
    
    return model, tokenizer, idx_to_label


def predict_intent(model, tokenizer, text, idx_to_label):
    """Predict intent for a single text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    encoding = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=128, return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    return {
        "intent": idx_to_label[pred],
        "confidence": round(confidence, 3),
        "probabilities": {v: round(probs[0][k].item(), 3) for k, v in idx_to_label.items()}
    }


# ------------------------------------------------------------------
# TEST DATA - Real-world examples
# ------------------------------------------------------------------

TEST_DATA = [
    # Positive cases
    ("This app is amazing, I love it!", "positive", "praise"),
    ("Best product ever, highly recommend!", "positive", "praise"),
    ("Great job team, well done!", "positive", "praise"),
    
    # Neutral cases
    ("I have a question about pricing", "neutral", "support_request"),
    ("What is the difference between A and B?", "neutral", "general_discussion"),
    ("Looking for reviews before buying", "neutral", "general_discussion"),
    
    # Negative cases
    ("This app keeps crashing!", "negative", "complaint"),
    ("Waste of money, terrible experience", "negative", "complaint"),
    ("Customer support is useless", "negative", "complaint"),
    ("I want my money back immediately", "negative", "refund_risk"),
    
    # Edge cases that caused false positives before
    ("Ashish bhai has class 👏 🙌 🔥", "positive", "praise"),  # Should be positive, not negative!
    ("Great assessment of the situation", "neutral", "general_discussion"),  # Contains "assess" -> "ass"
    ("This is the classic example", "neutral", "general_discussion"),  # Contains "classic"
    ("Assess the situation please", "neutral", "support_request"),
    
    # Toxic comments (should be detected)
    ("You are an idiot!", "negative", "complaint"),  # Toxic
    ("This is garbage", "negative", "complaint"),  # Toxic
    ("Worthless piece of trash", "negative", "complaint"),  # Toxic
    
    # Sarcasm
    ("Oh great, another bug to fix", "negative", "complaint"),  # Sarcasm
    ("Yeah right, this is amazing 😏", "negative", "complaint"),  # Sarcasm
]


# ------------------------------------------------------------------
# RUN TESTS
# ------------------------------------------------------------------

print("=" * 80)
print("CUSTOM MODEL ACCURACY EVALUATION")
print("=" * 80)

# Load models
print("\nLoading models...")
try:
    sentiment_model, sentiment_tokenizer, sentiment_labels = load_sentiment_model()
    print("✅ Sentiment model loaded")
except FileNotFoundError:
    print("❌ Sentiment model not found. Run train_sentiment_model.py first")
    sentiment_model = None

try:
    intent_model, intent_tokenizer, intent_labels = load_intent_model()
    print("✅ Intent model loaded")
except FileNotFoundError:
    print("❌ Intent model not found. Run train_intent_model.py first")
    intent_model = None

# Run predictions
print("\nRunning predictions on test set...")
results = []

for text, expected_sentiment, expected_intent in TEST_DATA:
    result = {
        "text": text,
        "expected_sentiment": expected_sentiment,
        "expected_intent": expected_intent
    }
    
    if sentiment_model:
        sentiment_pred = predict_sentiment(sentiment_model, sentiment_tokenizer, text, sentiment_labels)
        result["predicted_sentiment"] = sentiment_pred["sentiment"]
        result["sentiment_confidence"] = sentiment_pred["confidence"]
    
    if intent_model:
        intent_pred = predict_intent(intent_model, intent_tokenizer, text, intent_labels)
        result["predicted_intent"] = intent_pred["intent"]
        result["intent_confidence"] = intent_pred["confidence"]
    
    results.append(result)

# Calculate accuracy
print("\n" + "-" * 80)
print("SENTIMENT ACCURACY")
print("-" * 80)

correct_sentiment = sum(1 for r in results if r.get("predicted_sentiment") == r["expected_sentiment"])
sentiment_accuracy = correct_sentiment / len(results) * 100

print(f"Total tests: {len(results)}")
print(f"Correct predictions: {correct_sentiment}")
print(f"Accuracy: {sentiment_accuracy:.1f}%")

# Intent accuracy
print("\n" + "-" * 80)
print("INTENT ACCURACY")
print("-" * 80)

correct_intent = sum(1 for r in results if r.get("predicted_intent") == r["expected_intent"])
intent_accuracy = correct_intent / len(results) * 100

print(f"Total tests: {len(results)}")
print(f"Correct predictions: {correct_intent}")
print(f"Accuracy: {intent_accuracy:.1f}%")

# Detailed results
print("\n" + "-" * 80)
print("DETAILED PREDICTIONS")
print("-" * 80)

for r in results:
    sentiment_ok = "✓" if r.get("predicted_sentiment") == r["expected_sentiment"] else "✗"
    intent_ok = "✓" if r.get("predicted_intent") == r["expected_intent"] else "✗"
    
    print(f"\n{sentiment_ok} Sentiment: {r['predicted_sentiment']} (expected: {r['expected_sentiment']})")
    print(f"{intent_ok} Intent: {r['predicted_intent']} (expected: {r['expected_intent']})")
    print(f"   Text: {r['text'][:60]}...")
    if sentiment_model:
        print(f"   Sentiment confidence: {r.get('sentiment_confidence', 'N/A')}")
    if intent_model:
        print(f"   Intent confidence: {r.get('intent_confidence', 'N/A')}")

# False positive analysis
print("\n" + "-" * 80)
print("FALSE POSITIVE ANALYSIS")
print("-" * 80)

# Check for false positives on "class" containing "ass"
false_positives = []
for r in results:
    if "class" in r["text"].lower() and r["predicted_sentiment"] == "negative":
        false_positives.append(r)

if false_positives:
    print("⚠️ False positives detected:")
    for fp in false_positives:
        print(f"  - '{fp['text']}' incorrectly classified as negative")
else:
    print("✅ No false positives on 'class' containing 'ass'!")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Sentiment Accuracy: {sentiment_accuracy:.1f}%")
print(f"Intent Accuracy: {intent_accuracy:.1f}%")

if sentiment_accuracy >= 90 and intent_accuracy >= 85:
    print("\n✅ Models meet quality threshold!")
elif sentiment_accuracy >= 80 and intent_accuracy >= 75:
    print("\n⚠️ Models acceptable but could improve")
else:
    print("\n❌ Models need retraining with more data")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("models/model_test_results.csv", index=False)
print(f"\n✅ Results saved to: models/model_test_results.csv")
