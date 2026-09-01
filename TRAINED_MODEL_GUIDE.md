# 🎓 Custom Trained ML Model — Implementation Guide

## ✅ What's Actually Shipped

A single **multi-task BERT model** — one shared `bert-base-uncased` encoder
with two classification heads trained jointly:

- **Sentiment head** → `negative` / `neutral` / `positive`
- **Intent head** → 11 classes: `advertisement_or_promotion`,
  `complaint_or_problem_report`, `engagement_bait`,
  `feature_or_content_request`, `financial_promotion`,
  `fraudulent_service_offer`, `general_discussion`,
  `giveaway_or_reward_scam`, `praise_or_appreciation`,
  `question_or_information_request`, `user_experience_feedback`

This replaces the old plan of two separate BERT models — one shared backbone
is trained once and both predictions come from a single forward pass, which
is faster and avoids loading two full BERT models into memory.

### Files

| Path | Purpose |
|---|---|
| `models/train_intent_model.py` | Training script (`MultiTaskClassifier`: shared BERT + 2 linear heads) |
| `models/test_model_accuracy.py` | Evaluation script |
| `models/models/multitask_bert/multitask_model.pt` | Trained weights (`bert.*`, `sentiment_head.*`, `intent_head.*`) |
| `models/models/multitask_bert/tokenizer/` | Saved BERT tokenizer |
| `models/models/multitask_bert/sentiment_labels.json` | `{0: "negative", 1: "neutral", 2: "positive"}` |
| `models/models/multitask_bert/intent_labels.json` | `{0: "advertisement_or_promotion", ...}` (11 classes) |
| `models/models/multitask_bert/*validation*metrics.json` / `*confusion_matrix.csv` | Validation reports |
| `src/analyzer.py` | Loads the model and uses it in `analyze_comment()` |

---

## 🔧 How It's Wired Into the App

`src/analyzer.py` lazily loads the model the first time `analyze_comment()`
or `analyze_dataframe()` is called (`load_trained_models()`), then every
comment is passed through it directly:

```
comment text → BertTokenizer → shared BERT encoder → pooled output
                                        │
                        ┌───────────────┴───────────────┐
                sentiment_head (3-way)          intent_head (11-way)
                        │                                │
                softmax → argmax                 softmax → argmax
```

Rule-based lexicons (spam markers, toxicity markers, emotion words) are
**kept alongside** the model rather than replaced — they catch things the
model wasn't trained to classify (spam links, profanity/threats, emotional
tone) and give every result an explainable "why" (`reason_summary`,
`key_phrases`). The priority-score logic combines both: ML sentiment/intent
+ rule-based spam/toxicity/emotion signals.

If `models/models/multitask_bert/multitask_model.pt` is missing, or
`torch`/`transformers` fail to import, `analyzer.py` automatically falls
back to VADER sentiment + the keyword-lexicon intent mapping — the app
never crashes for lack of the trained model, it just runs less accurately.
You can check which mode is active at runtime via
`analyzer.USING_TRAINED_MODEL` and `analyzer.MODEL_VERSION` (also shown in
the dashboard's "Deep Analysis" tab).

---

## 📋 Retraining / Updating the Model

```powershell
pip install transformers torch scikit-learn --break-system-packages

# CSV needs columns: comment_text, sentiment, intent
python models/train_intent_model.py --csv_path models/comments_sentiment.csv --epochs 5

python models/test_model_accuracy.py
```

Output goes to `models/multitask_bert/` by default — this project keeps its
active model under `models/models/multitask_bert/`, so pass
`--output_dir models/models/multitask_bert` to overwrite it in place, or
copy the output folder over manually.

`analyzer.py` requires **no code changes** after retraining as long as the
label files (`sentiment_labels.json`, `intent_labels.json`) and
`multitask_model.pt` are regenerated together — the loader reads the label
counts from those files, so adding/removing intent classes just works.

---

## ⚠️ Data Quality Note

The current `final_validation_metrics.json` reports **100% accuracy on
every class** (sentiment and all 11 intents). That is a red flag, not a
badge of honor — it almost always means the training/validation data is
too small, too templated, or too easy to separate (e.g. one canned sentence
per class repeated with light variation), so the model has memorized
surface patterns rather than learned to generalize.

Spot-checking against real, messy comments confirms this: the model is
confident and correct on phrasing close to its training templates, but
noticeably weaker (low confidence, occasional wrong sentiment/intent) on
more natural or unusual phrasing, sarcasm, or profanity mixed with
otherwise neutral-sounding words. Before relying on this model for a real
demo/deployment, it's worth:

- Expanding the training CSV with more real (not templated) examples per
  class, especially hard negatives (angry comments, sarcasm, profanity).
- Checking `models/models/multitask_bert/validation_*_confusion_matrix.csv`
  after retraining on a validation set the model hasn't effectively seen
  templated duplicates of.
- Keeping the rule-based toxicity/spam lexicons active as a safety net —
  they currently catch several cases (e.g. slurs, threats) that the model
  alone gets wrong.

---

## 📈 What You Get At Runtime

- **Inference:** one forward pass per comment gives both sentiment and
  intent (faster than two separate models).
- **Confidence scores:** softmax probability per prediction, surfaced as
  `sentiment_score` (signed, −1..1) and `intent_confidence` (0..1).
- **Explainability:** `reason_summary` and `key_phrases` still come from
  the rule-based layer, so every flagged comment cites the exact
  spam/toxic/emotion words that contributed — this is preserved even
  though sentiment/intent themselves are now ML-driven.
- **Graceful degradation:** falls back to VADER + lexicons if the model or
  its dependencies aren't available, with no code changes needed.

---

**Document Version:** 2.0
**Last Updated:** 2026-08-08
**Model:** 1 multi-task model (shared BERT + sentiment head + intent head)