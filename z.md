# Comment Intelligence Platform — Full Project Documentation
*A line-by-line, beginner-friendly walkthrough of every part of the project*

## Table of contents

1. [What this project actually does](#1-what-this-project-actually-does-in-plain-english)
2. [The big picture — data flow](#2-the-big-picture--how-data-flows-through-the-system)
3. [Deep dive: `src/analyzer.py` — the brain](#3-deep-dive-srcanalyzerpy--the-brain-of-the-project)
4. [Deep dive: `models/train_intent_model.py` — training](#4-deep-dive-modelstrain_intent_modelpy--how-the-model-was-actually-trained)
5. [What is a `.pt` file? CPU vs GPU](#5-what-is-a-pt-file-and-cpu-vs-gpu--explained-properly)
6. [Deep dive: `src/redis_client.py` — caching](#6-deep-dive-srcredis_clientpy--talking-to-upstash-redis)
7. [Deep dive: `src/alerts.py` — Jira escalation](#7-deep-dive-srcalertspy--deciding-what-becomes-a-jira-ticket)
8. [Deep dive: `dashboard.py` — the Streamlit app](#8-deep-dive-dashboardpy--the-streamlit-web-app)
9. [The smaller supporting files](#9-the-smaller-supporting-files)
10. [One comment's full journey, end to end](#10-putting-it-all-together--one-comments-full-journey)
11. [Known challenges & how to improve them](#11-known-challenges-and-how-to-improve-them)
12. [Quick glossary](#12-quick-glossary-for-terms-used-throughout-this-document)

---

## 1. What this project actually does (in plain English)

Imagine a popular YouTube/Instagram/Bluesky post gets 2,000 comments overnight.
A human can't read all of them, but hidden in there might be: someone
threatening the creator, someone asking a genuine support question, a scam
link, or a wave of angry complaints about something real. This project reads
every comment automatically and answers three questions for each one:

1. **How does this person feel?** (sentiment: positive / neutral / negative)
2. **What do they want?** (intent: complaint, question, praise, scam, spam, ...)
3. **Does a human need to see this right now?** (priority: Low → Critical, and
   for the serious ones, an automatic Jira ticket)

It does this two ways at once, blended together:
- A **custom-trained AI model** (a fine-tuned BERT) reads the sentence and
  predicts sentiment + intent — this is the "understands meaning" part.
- **Rule-based keyword lists** (lexicons) catch spam links, profanity,
  threats, and emotional language — this is the "explainable, always-works"
  part, and it's what lets the app say *exactly* which words caused a
  decision.

Everything is glued together by a **Streamlit dashboard** (a Python library
that turns a script into a web app), with **Upstash Redis** used as a cache
(so the same dataset isn't re-analyzed every time) and a de-dupe store (so
the same bad comment doesn't file 5 duplicate Jira tickets), and **Jira**
used as the "a human needs to act on this" ticketing system.

---

## 2. The big picture — how data flows through the system

```
 ┌──────────────┐   ┌────────────────┐   ┌──────────────────────────┐
 │ YouTube /     │   │  dashboard.py   │   │  src/analyzer.py          │
 │ Instagram /   │──▶│  (Streamlit UI) │──▶│  analyze_comment() runs   │
 │ Bluesky fetch │   │  fetch → cache  │   │  the ML model + lexicons  │
 │ scripts       │   │  check → show   │   │  on every single comment  │
 └──────────────┘   └───────┬────────┘   └─────────────┬─────────────┘
                             │                            │
                             ▼                            ▼
                    ┌──────────────────┐        ┌──────────────────┐
                    │ Upstash Redis     │        │ src/alerts.py     │
                    │ - caches analysis │        │ - decides which   │
                    │ - dedupes alerts  │◀───────│   comments are    │
                    │ - counters        │        │   Jira-worthy     │
                    └──────────────────┘        └────────┬──────────┘
                                                            ▼
                                                   ┌──────────────────┐
                                                   │ Jira Cloud        │
                                                   │ (a ticket per     │
                                                   │  genuine problem) │
                                                   └──────────────────┘
```

Every file has one job. That separation matters for a beginner to notice —
it's a common, good pattern called **separation of concerns**:

| File | Its one job |
|---|---|
| `youtube/`, `instagram/`, `bluesky/fetch_comments.py` | Pull raw comments from each platform's API into a CSV |
| `src/preprocess.py` | Clean text for similarity/spam-cluster math (not for the ML model) |
| `src/analyzer.py` | **The brain.** Takes one comment's text, returns sentiment/intent/priority |
| `src/simi.py` | Detect "spam campaigns" — many near-identical comments |
| `src/burst.py` | Detect a sudden spike in comment volume (bot/brigading signal) |
| `src/score.py` | Turn similarity + burst + spam ratio into one "campaign score" |
| `src/alerts.py` | Decide which analyzed comments deserve a Jira ticket, and file it |
| `src/redis_client.py` | Talk to Upstash Redis (cache, dedupe, counters) |
| `src/jira_client.py` | Talk to Jira Cloud's REST API (create tickets) |
| `src/ui_theme.py` | CSS + small HTML helper functions for the dashboard's look |
| `dashboard.py` | The Streamlit web app that ties everything above together |
| `models/train_intent_model.py` | The script that *trains* the custom AI model |
| `models/models/multitask_bert/` | The *output* of training — the saved, ready-to-use model |

---

## 3. Deep dive: `src/analyzer.py` — the brain of the project

This is the single most important file. Every comment passes through
`analyze_comment()` in this file. We'll go through it top to bottom.

### 3.1 Imports and why each one is there

```python
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional
```

- `from __future__ import annotations` — lets you write type hints like
  `list[str]` without Python complaining on older versions. Purely a
  compatibility switch, has no runtime effect on logic.
- `os` — used to build file paths (`os.path.join`) and read environment
  variables (`os.getenv`) like `GEMINI_API_KEY`.
- `re` — Python's regular-expression module, used everywhere in the lexicon
  matching to avoid false positives (explained in detail below).
- `json` — used to load `sentiment_labels.json` / `intent_labels.json`, the
  files that translate the model's number outputs back into words.
- `dataclass, field` — a decorator that auto-generates the boring parts of a
  class (constructor, `__repr__`, etc.) for a data-holding class —
  `CommentAnalysis` uses this.
- `Optional` — a type hint meaning "this can be `None`."

### 3.2 Locating and loading the trained model

```python
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "models", "multitask_bert"
)
```

Read this right-to-left as a sentence: *"take this file's own folder
(`src/`), go up one level to the project root, then into
`models/models/multitask_bert`."* Building the path this way — relative to
`__file__` — means the code works no matter what folder you happen to be
sitting in when you type `streamlit run dashboard.py`. If it were hardcoded
as `"models/models/multitask_bert"` (a *relative* path with no anchor), it
would silently break the moment someone ran the app from a different
working directory.

```python
_multitask_model = None
_multitask_tokenizer = None
_sentiment_labels = None   # {0: "negative", 1: "neutral", 2: "positive"}
_intent_labels = None      # {0: "advertisement_or_promotion", ...}

USING_TRAINED_MODEL = False     # True once the custom model is loaded
_MODEL_LOAD_ATTEMPTED = False   # only try loading once per process
MODEL_VERSION = "rule-engine-v1.3+vader-fallback"
RULES_VERSION = "rules-v2-toxicity-severity-tiers"
```

These are **module-level variables** — they live for as long as the Python
process runs (the whole time your Streamlit app is up), not just for one
function call. They start out empty/`False`/generic, and get filled in the
first time a comment is analyzed. This pattern is called **lazy loading**:
don't do the expensive work (loading a several-hundred-MB neural network)
until it's actually needed, and then only do it *once*, not on every
comment.

`MODEL_VERSION` and `RULES_VERSION` exist so that any code caching analysis
results (see `dashboard.py`'s Redis cache) can tell "did the model or the
scoring rules change since this was cached?" — if either changes, the cache
key changes too, so stale results never get served silently.

### 3.3 The model's architecture — `MultiTaskClassifier`

Before you can load *trained weights* (the numbers the model learned), you
need to build an *empty* neural network with the exact same shape — like
having the mold before you pour the cast.

```python
class MultiTaskClassifier:
    @staticmethod
    def build(num_sentiment_labels: int, num_intent_labels: int):
        import torch.nn as nn
        from transformers import BertModel

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained("bert-base-uncased")
                hidden_size = self.bert.config.hidden_size
                self.dropout = nn.Dropout(0.3)
                self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)
                self.intent_head = nn.Linear(hidden_size, num_intent_labels)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled = self.dropout(outputs.pooler_output)
                return self.sentiment_head(pooled), self.intent_head(pooled)

        return _Model()
```

Line by line:

- `import torch.nn as nn` — PyTorch's neural-network building blocks
  (layers, activation functions, loss functions all live under `nn`).
  Imported *inside* the function on purpose — if `torch` isn't installed,
  this line never runs unless someone actually calls `build()`, so the rest
  of the app doesn't crash just because PyTorch is missing.
- `class _Model(nn.Module):` — every custom neural network in PyTorch is a
  Python class that inherits from `nn.Module`. That parent class gives you,
  for free, things like `.eval()`, `.parameters()`, `.to(device)`, and the
  ability to save/load weights.
- `self.bert = BertModel.from_pretrained("bert-base-uncased")` — downloads
  (or loads from a local cache) Google's pretrained BERT — 12 transformer
  layers, 768-dimensional hidden size, already trained on a huge amount of
  English text to understand grammar and word meaning. This is **transfer
  learning**: you don't train language understanding from zero, you start
  from something that already understands English and specialize it.
- `hidden_size = self.bert.config.hidden_size` — for `bert-base-uncased`
  this is `768`. Every sentence, no matter how long or short, gets
  summarized by BERT into a vector of 768 numbers.
- `self.dropout = nn.Dropout(0.3)` — during *training only*, this randomly
  zeroes out 30% of the numbers flowing through it on each pass. That
  sounds destructive, but it forces the network to not rely too heavily on
  any single feature, which reduces **overfitting** (memorizing the
  training data instead of learning general patterns). During prediction
  (`model.eval()`), dropout automatically switches itself off.
- `self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)` — a
  single fully-connected layer: takes the 768 numbers in, produces 3
  numbers out (one score per sentiment class). `nn.Linear` is literally
  matrix multiplication plus a bias: `output = input @ weight_matrix +
  bias`. All the "learning" is just adjusting the numbers inside that
  weight matrix until the output scores make sense.
- `self.intent_head = nn.Linear(hidden_size, num_intent_labels)` — the same
  idea, but 768 in → 11 out (one score per intent class).
- `forward(self, input_ids, attention_mask)` — this method is what actually
  *runs* when you feed data through the model (`model(x)` under the hood
  calls `model.forward(x)`).
  - `input_ids` — your comment's text, already converted to a list of
    numbers by the tokenizer (explained below).
  - `attention_mask` — a same-length list of 1s and 0s. 1 means "this is a
    real word, pay attention to it," 0 means "this is padding, ignore it."
  - `outputs = self.bert(...)` — runs the full 12-layer transformer over
    the tokens.
  - `outputs.pooler_output` — BERT gives you output for *every* token, but
    for classification you usually just want one summary vector for the
    *whole sentence* — `pooler_output` is that single 768-number summary
    (technically it's derived from the special `[CLS]` token BERT prepends
    to every input).
  - `self.dropout(pooled)` — apply the regularization described above.
  - `return self.sentiment_head(pooled), self.intent_head(pooled)` — both
    heads read the *same* summary vector and each produce their own
    independent guess. This is why it's called **multi-task**: one shared
    understanding of the sentence, two separate opinions read off it.

Why is `MultiTaskClassifier` written as a class with a `build()` static
method instead of just a plain class? So that `import torch` only happens
if you actually call `.build()` — keeping the rest of the file safely
importable even on a machine that doesn't have PyTorch installed (it'll
just fall back to the rule-based path, explained later).

### 3.4 Actually loading the saved weights — `load_trained_models()`

```python
def load_trained_models():
    global _multitask_model, _multitask_tokenizer
    global _sentiment_labels, _intent_labels
    global USING_TRAINED_MODEL, _MODEL_LOAD_ATTEMPTED, MODEL_VERSION

    if _MODEL_LOAD_ATTEMPTED:
        return
    _MODEL_LOAD_ATTEMPTED = True
```

- The `global` keyword tells Python "when I assign to these names inside
  this function, modify the module-level variable, don't create a new
  local one." Without it, `USING_TRAINED_MODEL = True` later in the
  function would just create a throwaway local variable and the change
  would be invisible outside the function.
- `if _MODEL_LOAD_ATTEMPTED: return` — this is the "only try once" guard.
  The very first time any comment is analyzed, this function does its real
  work (which is slow — loading a whole neural network can take seconds).
  Every call after that just returns instantly, because
  `_MODEL_LOAD_ATTEMPTED` is now `True`.

```python
    weights_path = os.path.join(_MODEL_DIR, "multitask_model.pt")
    tokenizer_path = os.path.join(_MODEL_DIR, "tokenizer")
    sentiment_labels_path = os.path.join(_MODEL_DIR, "sentiment_labels.json")
    intent_labels_path = os.path.join(_MODEL_DIR, "intent_labels.json")

    if not os.path.exists(weights_path):
        print(f"ℹ️ Custom trained model not found at {weights_path}.")
        ...
        return
```

Before doing anything expensive, check the model file actually exists on
disk. If it doesn't (maybe someone cloned the repo without the model
folder, or it hasn't been trained yet), print a helpful message and return
— `USING_TRAINED_MODEL` stays `False`, and the rest of the app will use the
VADER + lexicon fallback instead of crashing.

```python
    try:
        import torch
        from transformers import BertTokenizer

        with open(sentiment_labels_path, "r", encoding="utf-8") as f:
            _sentiment_labels = {int(k): v for k, v in json.load(f).items()}
        with open(intent_labels_path, "r", encoding="utf-8") as f:
            _intent_labels = {int(k): v for k, v in json.load(f).items()}
```

- `import torch` — only imported here, inside the `try`, so a machine
  without PyTorch installed doesn't crash the whole app just by starting
  it; it only fails *this* function, which is caught below.
- `json.load(f)` reads the label file, e.g. `{"0": "negative", "1":
  "neutral", "2": "positive"}` — note the keys are **strings** in JSON
  (JSON has no concept of integer dictionary keys), so
  `{int(k): v for k, v in ...}` converts `"0"` → `0` so we can later do
  `_sentiment_labels[predicted_index]` directly.

```python
        _multitask_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        model = MultiTaskClassifier.build(
            num_sentiment_labels=len(_sentiment_labels),
            num_intent_labels=len(_intent_labels),
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        _multitask_model = model
```

- `BertTokenizer.from_pretrained(tokenizer_path)` — loads the *exact* rules
  used to chop text into tokens during training (which sub-words exist,
  what number each maps to). This must match training exactly, which is
  why the training script *saves* its own tokenizer into the model folder
  instead of assuming you'll load a fresh one later.
- `MultiTaskClassifier.build(...)` — builds the empty "mold" described in
  section 3.3, sized to match however many sentiment/intent classes were
  in *this* label file (so if you retrain with a 12th intent class added,
  this code needs zero changes — it reads the count from the JSON).
- `torch.load(weights_path, map_location="cpu")` — reads the `.pt` file
  (explained in depth in section 8) into a Python dictionary of tensors —
  the actual learned numbers. `map_location="cpu"` means "even if this was
  trained on a GPU, put the numbers on the CPU" — makes the file portable
  to a machine with no GPU.
- `model.load_state_dict(state_dict)` — pours those numbers into the empty
  mold. This is the step that fails loudly if the architecture doesn't
  match (e.g. wrong number of output classes, or different layer names) —
  which is exactly why `MultiTaskClassifier` here had to be built to *match
  precisely* what `train_intent_model.py` produced.
- `model.eval()` — switches the model into "prediction mode": turns off
  dropout, and tells layers like batch-normalization to use their learned
  statistics instead of the current batch's statistics. **Always** call
  this before predicting; forgetting it is one of the most common PyTorch
  beginner bugs (predictions become inconsistent/random).

```python
        USING_TRAINED_MODEL = True
        MODEL_VERSION = "multitask-bert-base-uncased-v1.0 (sentiment+intent, fine-tuned)"
        ...
    except Exception as e:
        print(f"⚠️ Failed to load trained model, using rule-based fallback: {e}")
        USING_TRAINED_MODEL = False
        MODEL_VERSION = "rule-engine-v1.3+vader-fallback"
        _multitask_model = None
        _multitask_tokenizer = None
```

The whole loading process is wrapped in `try/except Exception`. If
*anything* goes wrong — missing library, corrupted file, no internet to
download `bert-base-uncased`'s architecture config — the `except` block
catches it, logs why, and flips `USING_TRAINED_MODEL` back to `False`. This
is defensive programming: a failure here degrades the app's accuracy, it
never crashes it.

### 3.5 Turning one comment into a prediction — `_predict_with_trained_model()`

```python
def _predict_with_trained_model(text: str):
    import torch

    encoding = _multitask_tokenizer(
        text[:1000],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
```

- `text[:1000]` — a cheap safety cap so an absurdly long comment (someone
  pasting an essay) doesn't slow down tokenization; BERT will truncate to
  128 tokens anyway, so anything past ~1000 characters is guaranteed to be
  cut regardless.
- Calling the tokenizer like a function (`_multitask_tokenizer(text, ...)`)
  runs the actual tokenization:
  - `truncation=True` — if the text produces more than `max_length` tokens,
    cut it off rather than erroring.
  - `padding="max_length"` — if the text produces *fewer* than 128 tokens,
    pad the rest with a special `[PAD]` token. Why pad at all? Because
    PyTorch tensors need a fixed, rectangular shape — you can't have one
    row of a matrix be 12 numbers long and another be 47.
  - `max_length=128` — the cap. Chosen in training as a trade-off: most
    comments are short, and larger values mean slower, more memory-hungry
    computation for little benefit on short text.
  - `return_tensors="pt"` — return PyTorch tensors (`"pt"`) instead of
    plain Python lists, so they can go straight into the model.

```python
    with torch.no_grad():
        sentiment_logits, intent_logits = _multitask_model(
            encoding["input_ids"], encoding["attention_mask"]
        )
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0]
        intent_probs = torch.softmax(intent_logits, dim=1)[0]
```

- `torch.no_grad()` — PyTorch normally tracks every calculation so it can
  later compute gradients (needed for *training*, via backpropagation).
  During prediction we're never going to call `.backward()`, so tracking
  that is pure waste — `no_grad()` turns it off, making prediction faster
  and using less memory. This is one of the most important lines to
  understand: forgetting it doesn't break correctness, but it silently
  wastes resources on every single prediction.
- `sentiment_logits, intent_logits = _multitask_model(...)` — calls
  `forward()` from section 3.3. `logits` is the standard term for a
  network's *raw, un-normalized* output scores — they can be any real
  number (including negative), and don't yet mean "probability."
- `torch.softmax(sentiment_logits, dim=1)` — converts raw scores into
  probabilities that sum to 1. E.g. logits `[1.2, -0.3, 3.9]` might become
  probabilities `[0.09, 0.02, 0.89]` — now you can read that last number as
  "89% confident it's the third class." `dim=1` means "do this normalization
  across the class dimension," since the tensor's shape is
  `[batch_size, num_classes]` even though our batch size here is 1.
- `[0]` — pulls out the single row (we only sent one comment in this
  batch), leaving a flat probability vector.

```python
    sentiment_idx = int(torch.argmax(sentiment_probs).item())
    intent_idx = int(torch.argmax(intent_probs).item())

    sentiment_label = _sentiment_labels[sentiment_idx]
    sentiment_conf = float(sentiment_probs[sentiment_idx].item())
    intent_label = _intent_labels[intent_idx]
    intent_conf = float(intent_probs[intent_idx].item())

    return sentiment_label.capitalize(), sentiment_conf, intent_label, intent_conf
```

- `torch.argmax(...)` — finds the *index* of the largest probability
  ("which class won"), not the probability's value.
- `.item()` — PyTorch tensors are their own data type, even a single
  number; `.item()` converts a one-element tensor into a plain Python
  `int`/`float` so the rest of the (non-PyTorch) code can use it normally.
- `_sentiment_labels[sentiment_idx]` — translates the winning index (e.g.
  `2`) back into its word (`"positive"`) using the JSON file loaded
  earlier.
- `.capitalize()` — turns `"positive"` into `"Positive"`, matching the
  Title-case convention the rest of the app (and the old rule-based code)
  already used, so downstream code and the UI didn't need to change.
### 3.6 Lexicons — the "explainable" half of the system

```python
EMOTION_LEXICON = {
    "anger": ["angry", "furious", "rage", "pissed", ...],
    "sarcasm_risk": ["yeah right", "sure jan", "totally", ...],
    "urgency": ["asap", "immediately", "right now", ...],
}

INTENT_LEXICON = {
    "refund_risk": ["refund", "money back", "chargeback", ...],
    "purchase_intent": ["where can i buy", "how much", ...],
    "complaint": ["worst", "terrible", "broken", ...],
    "support_request": ["how do i", "need help", ...],
    "feature_request": ["please add", "feature request", ...],
    "misinformation_risk": ["fake news", "hoax", ...],
    "praise": ["love this", "amazing", "best video", ...],
}

TOXIC_MARKERS_SEVERE = ["kill yourself", "kys", "hope you die", "kill you", ...]
TOXIC_MARKERS_MILD = ["idiot", "stupid", "fuck", "trash", "disgusting", ...]
SPAM_MARKERS = ["subscribe to my channel", "check my bio", "dm me", "click link", ...]
```

These are just **Python dictionaries and lists of strings** — no AI
involved at all. `EMOTION_LEXICON` and `INTENT_LEXICON` are dictionaries
where each key (e.g. `"anger"`) maps to a list of trigger phrases.
`TOXIC_MARKERS_SEVERE`/`_MILD` and `SPAM_MARKERS` are flat lists (no
sub-categories needed — a spam phrase is just a spam phrase). Why keep
these *alongside* the ML model instead of replacing them entirely?
Because:
1. They're instantly explainable — "flagged because it contains the word
   X," which a neural network can't easily produce on its own.
2. They catch things the model was never trained to catch (spam links,
   specific profanity), acting as a safety net.
3. They cost essentially zero compute — no GPU/CPU-heavy math, just string
   searching.

### 3.7 How phrase-matching actually works — `_find_hits()` / `_find_flat_hits()`

```python
def _find_flat_hits(text: str, phrases: list) -> list:
    matched = []
    for phrase in phrases:
        if ' ' in phrase:
            if phrase in text:
                matched.append(phrase)
        else:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(phrase)
    return matched
```

This function answers "which of these phrases appear in this text?" Two
different strategies depending on the phrase:

- **Multi-word phrases** (`' ' in phrase`, e.g. `"kill yourself"`) — a
  simple substring check (`phrase in text`) is safe, because it's very
  unlikely `"kill yourself"` accidentally appears as part of an unrelated
  longer phrase.
- **Single words** (e.g. `"ass"`) — a plain substring check would be a
  disaster: `"ass" in "class"` is `True` in Python! That would falsely flag
  the word "class" as profanity. The fix is `re.escape(phrase)` wrapped in
  `\b...\b` (word boundaries): `\bass\b` only matches "ass" as a **whole
  word**, with a non-letter (space, punctuation, start/end of string) on
  each side. `re.escape()` protects against phrases that happen to contain
  regex-special characters (like `.` or `?`) being misinterpreted as regex
  syntax instead of literal characters.
- `_find_hits()` does the same thing but for a *dictionary* of categories
  (like `EMOTION_LEXICON`), returning `{category: [matched phrases]}`
  instead of a flat list — used where we need to know *which* emotion or
  intent category was hit, not just whether *something* was hit.

### 3.8 `CommentAnalysis` — the data container for one comment's results

```python
@dataclass
class CommentAnalysis:
    comment: str
    author: str = "Unknown"
    platform: str = "unknown"

    sentiment: str = "Neutral"
    sentiment_score: float = 0.0

    emotions: list = field(default_factory=list)
    primary_emotion: str = "none"

    is_spam: bool = False
    spam_score: float = 0.0
    is_toxic: bool = False
    toxicity_score: float = 0.0
    toxicity_severity: str = "none"   # "none" | "mild" | "severe"

    intent: str = "general_discussion"
    intent_confidence: float = 0.0

    priority: str = "Low"
    priority_score: int = 0

    key_phrases: list = field(default_factory=list)
    confidence: float = 0.0
    model_version: str = MODEL_VERSION
    reason_summary: str = ""
```

`@dataclass` is a decorator (Python's way of wrapping extra behavior around
a class) that auto-generates `__init__`, so you *don't* have to write:

```python
def __init__(self, comment, author="Unknown", platform="unknown", ...):
    self.comment = comment
    self.author = author
    ...
```

...by hand for 15+ fields. You just declare the fields with their types
and defaults, and `@dataclass` writes that constructor for you.

- `comment: str` — no default value, so it's a **required** argument;
  you must always provide the comment's text.
- `author: str = "Unknown"` — has a default, so it's optional.
- `emotions: list = field(default_factory=list)` — you might expect
  `emotions: list = []` to work, but **that's a classic Python trap**: a
  plain `[]` as a default value would be created *once* when the class is
  defined and then *shared* across every single instance — appending to
  one comment's `emotions` list would silently corrupt every other
  comment's list too. `field(default_factory=list)` tells the dataclass
  "call `list()` fresh for every new instance" instead, avoiding that bug.
  This is true for any mutable default (lists, dicts, sets) in Python, not
  just dataclasses.
- `model_version: str = MODEL_VERSION` — this default is evaluated **once**,
  at import time, capturing whatever `MODEL_VERSION` was *before*
  `load_trained_models()` ever ran (i.e., always the fallback string). This
  is why `analyze_comment()` explicitly overwrites `result.model_version =
  MODEL_VERSION` near the end, after the real value is known — the
  dataclass default here is just a safe placeholder.

```python
    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["emotions"] = ", ".join(self.emotions) if self.emotions else "none"
        d["key_phrases"] = "; ".join(self.key_phrases) if self.key_phrases else "—"
        return d
```

Converts the object into a plain dictionary so it can become a row in a
pandas DataFrame (DataFrames don't like Python list objects sitting inside
cells — they display and sort better as plain strings), joining the lists
into comma/semicolon-separated text.

### 3.9 `analyze_comment()` — the full walkthrough, one comment at a time

This is the function everything above exists to support. Let's trace a
single comment through it start to finish.

```python
def analyze_comment(text: str, author: str = "Unknown", platform: str = "unknown",
                     like_count: int = 0) -> CommentAnalysis:
    load_trained_models()

    raw = text or ""
    norm = raw.lower().strip()

    result = CommentAnalysis(comment=raw, author=author or "Unknown", platform=platform)
```

- `load_trained_models()` — the lazy-load guard from section 3.4. Cheap to
  call every time; does real work only once.
- `raw = text or ""` — if `text` is `None` (can happen with messy CSV
  data), fall back to an empty string instead of crashing on `.lower()`
  later.
- `norm = raw.lower().strip()` — the **normalized** version used for all
  keyword matching: lowercased (so "IDIOT" and "idiot" both match) and
  stripped of leading/trailing whitespace. `raw` (original casing/spacing)
  is kept separately because it's what gets shown to the user and fed to
  the ML model — the model doesn't need lowercasing, BERT's tokenizer
  handles that itself.
- `result = CommentAnalysis(...)` — create the container that will be
  filled in and returned at the end. Every field not passed here starts at
  its dataclass default (Neutral, 0.0, etc.), and gets overwritten below.

**Spam detection:**
```python
    spam_hits = _find_flat_hits(norm, SPAM_MARKERS)
    result.is_spam = len(spam_hits) > 0
    result.spam_score = min(1.0, 0.4 * len(spam_hits) + (0.3 if result.is_spam else 0))
```
Find which spam phrases appear. `is_spam` is just "did we find at least
one." `spam_score` is a 0–1 confidence number: `0.4` per matched phrase
plus a flat `0.3` bonus the moment *any* match exists, capped at `1.0` by
`min()` so it never exceeds "100% confident."

**Toxicity detection (severity-tiered):**
```python
    severe_hits = _find_flat_hits(norm, TOXIC_MARKERS_SEVERE)
    mild_hits = _find_flat_hits(norm, TOXIC_MARKERS_MILD)
    toxic_hits = severe_hits + mild_hits

    result.is_toxic = bool(severe_hits or mild_hits)
    if severe_hits:
        result.toxicity_severity = "severe"
    elif mild_hits:
        result.toxicity_severity = "mild"
    else:
        result.toxicity_severity = "none"

    directed = _mentions_second_person(norm)
    result.toxicity_score = min(
        1.0,
        0.6 * len(severe_hits) + 0.2 * len(mild_hits) + (0.15 if directed and (severe_hits or mild_hits) else 0)
    )
```
This is a deliberately two-tier system (covered in depth in section 9,
"Challenges & fixes"). `severe_hits` are real threats/hate speech;
`mild_hits` are generic name-calling ("idiot," "stupid"). `is_toxic` stays
`True` if *either* list has anything (so nothing downstream that only
checks the boolean silently misses mild cases), but `toxicity_severity`
lets later code (priority scoring, Jira escalation) treat them very
differently. `directed = _mentions_second_person(norm)` checks whether the
comment seems aimed at "you" (someone in the conversation) rather than a
third party — a small nudge, explained in section 3.10 below.

**Sentiment + intent — the ML/fallback fork:**
```python
    intent_hits = _find_hits(norm, INTENT_LEXICON)

    if USING_TRAINED_MODEL and _multitask_model is not None:
        sentiment_label, sentiment_conf, intent_label, intent_conf = _predict_with_trained_model(raw)

        result.sentiment = sentiment_label
        if sentiment_label == "Positive":
            result.sentiment_score = round(sentiment_conf, 3)
        elif sentiment_label == "Negative":
            result.sentiment_score = round(-sentiment_conf, 3)
        else:
            result.sentiment_score = 0.0

        result.intent = intent_label
        result.intent_confidence = round(intent_conf, 3)

        if intent_label in ("advertisement_or_promotion", "financial_promotion",
                             "fraudulent_service_offer", "giveaway_or_reward_scam",
                             "engagement_bait"):
            result.is_spam = True
            result.spam_score = max(result.spam_score, round(min(1.0, 0.5 + 0.15 * intent_conf * 3), 2))
```
This is the `if` branch that runs when the trained model is available.
- `_predict_with_trained_model(raw)` — runs the actual neural network
  (section 3.5), using `raw` (original casing) rather than `norm`, since
  BERT's own tokenizer handles casing internally and was trained on
  natural, unlowercased text.
- `sentiment_score` is put on a **signed −1..1 scale** to match the
  convention the rest of the dashboard expects (positive numbers = happy,
  negative = unhappy, magnitude = confidence). The model itself just gives
  a label + a confidence between 0 and 1; this code translates that into
  the sign convention.
- The last block is a deliberate **cross-check**: if the ML model's intent
  prediction is one of the "spam-flavoured" categories, treat the comment
  as spam even if the keyword-based `SPAM_MARKERS` list didn't catch
  anything — the two detection methods reinforce each other instead of
  working in isolation. `max(result.spam_score, ...)` means whichever
  method is more confident wins, rather than one overwriting the other.

```python
    else:
        global _vader
        if _vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        vs = _vader.polarity_scores(raw)
        compound = vs["compound"]
        result.sentiment_score = round(compound, 3)
        if norm in GENERIC_SHORT or (len(norm.split()) <= 2 and not spam_hits):
            result.sentiment = "Neutral"
        elif compound >= 0.25:
            result.sentiment = "Positive"
        elif compound <= -0.25:
            result.sentiment = "Negative"
        else:
            result.sentiment = "Neutral"
```
This is the **fallback path** — runs only if the trained model isn't
available. VADER (`SentimentIntensityAnalyzer`) is a rule-based sentiment
tool, not machine learning — it has its own built-in dictionary of words
with pre-assigned positive/negative weights and a set of grammar rules
(e.g. "not good" flips the sign of "good," "!!!" amplifies intensity).
`vs["compound"]` is VADER's single overall score from −1 (most negative)
to +1 (most positive). Lazy-loaded the same way as the trained model
(`if _vader is None: ...`) so importing `vaderSentiment` only happens if
it's actually needed. `GENERIC_SHORT` is a small set of extremely common
short comments (like `"nice"`, `"first"`, `"👍"`) that VADER tends to
mis-score — they're hardcoded to `Neutral` instead of trusting VADER's
number.

```python
        if result.is_spam:
            result.intent = "advertisement_or_promotion"
            result.intent_confidence = round(min(0.95, 0.5 + 0.15 * len(spam_hits)), 2)
        elif intent_hits:
            for label, mapped in [
                ("refund_risk", "complaint_or_problem_report"),
                ("complaint", "complaint_or_problem_report"),
                ("support_request", "question_or_information_request"),
                ("misinformation_risk", "fraudulent_service_offer"),
                ("purchase_intent", "question_or_information_request"),
                ("feature_request", "feature_or_content_request"),
                ("praise", "praise_or_appreciation"),
            ]:
                if label in intent_hits:
                    result.intent = mapped
                    result.intent_confidence = round(min(0.95, 0.45 + 0.15 * len(intent_hits[label])), 2)
                    break
        else:
            result.intent = "general_discussion"
            result.intent_confidence = 0.35
```
Still inside the fallback branch: since there's no ML intent classifier
running, this maps the old `INTENT_LEXICON` hits onto the *same* 11-class
naming scheme the trained model uses (e.g. `"refund_risk"` →
`"complaint_or_problem_report"`). This matters because it means the rest of
the codebase (priority scoring, Jira gating) only ever has to think about
one label vocabulary, whether the ML model ran or not. The `for label,
mapped in [...]` loop checks each mapping *in order* and takes the
**first** match — the order is deliberately risk-first (`refund_risk`,
`complaint` before `praise`), so if a comment somehow trips both a
complaint phrase and a praise phrase, the more actionable one wins.

**Emotion detection:**
```python
    emotion_hits = _find_hits(norm, EMOTION_LEXICON)
    result.emotions = list(emotion_hits.keys())
    if emotion_hits:
        result.primary_emotion = max(emotion_hits, key=lambda k: len(emotion_hits[k]))
    else:
        result.primary_emotion = "none"
```
Same keyword-lookup pattern as before. `max(emotion_hits, key=lambda k:
len(emotion_hits[k]))` picks whichever emotion category had the *most*
matched phrases as the "primary" one — e.g. if a comment matches 3 anger
words and 1 urgency word, `primary_emotion` becomes `"anger"`.

### 3.10 Priority scoring — turning many signals into one number

```python
    score = 0
    reasons = []

    if result.toxicity_severity == "severe":
        score += 35 + (10 if directed else 0)
        reasons.append(f"severe toxic language / threat ({', '.join(severe_hits)})")
    elif result.toxicity_severity == "mild":
        score += 12 + (10 if directed else 0)
        target_note = "aimed at the channel" if directed else "third-party commentary, e.g. about a public figure"
        reasons.append(f"mild toxic language ({', '.join(mild_hits)}) — {target_note}")
    if result.is_spam:
        score += 15
        ...
    if result.sentiment == "Negative":
        score += 25
        ...
    if result.intent == "fraudulent_service_offer":
        score += 30
        ...
    if result.intent == "complaint_or_problem_report":
        score += 25
        ...
    if result.intent in ("financial_promotion", "giveaway_or_reward_scam"):
        score += 15
        ...
    if result.intent == "question_or_information_request":
        score += 10
        ...
    if result.intent == "user_experience_feedback" and result.sentiment == "Negative":
        score += 10
        ...
    if "urgency" in result.emotions:
        score += 10
        ...
    if "anger" in result.emotions or "frustration" in result.emotions:
        score += 10
        ...
    if like_count and like_count >= 25:
        score += 10
        reasons.append(f"high engagement ({like_count} likes — amplifies visibility)")

    score = min(100, score)
    result.priority_score = score

    if score >= 70:
        result.priority = "Critical"
    elif score >= 45:
        result.priority = "High"
    elif score >= 20:
        result.priority = "Medium"
    else:
        result.priority = "Low"
```
This is plain, transparent **additive scoring** — no ML here either, by
design: a human should be able to look at this code and understand exactly
why any comment got the score it did. Each `if` is an independent signal
that adds points if present; `reasons` collects a human-readable string for
each contributing signal (used later to build `reason_summary`, the "Why:"
text you see in the dashboard). `score = min(100, score)` caps the total at
100 even if many signals stack up. The final `if/elif` chain buckets the
raw number into four human-friendly labels.

Why these specific point values (35, 25, 15, 12, 10...)? They were tuned
so that severe toxicity or fraud alone is enough to reach "High" by itself,
while any single *mild* signal (a name-calling word, a high like-count)
only nudges the score partway — reaching "High" or "Critical" should
generally require *multiple* signals stacking up, not one keyword match.

**Explainability — collecting the evidence:**
```python
    all_hits = list({*spam_hits, *toxic_hits})
    for label_hits in emotion_hits.values():
        all_hits.extend(label_hits)
    for label_hits in intent_hits.values():
        all_hits.extend(label_hits)
    result.key_phrases = list(dict.fromkeys(all_hits))[:6]
```
- `{*spam_hits, *toxic_hits}` — the `*` unpacks both lists into a single
  **set** (deduplicating any phrase that happens to appear in both lists).
- `list(dict.fromkeys(all_hits))` — a common Python idiom for
  "deduplicate a list while preserving order" (sets alone don't preserve
  order; this trick uses the fact that dictionary keys are both unique
  *and*, since Python 3.7, insertion-ordered).
- `[:6]` — cap at 6 phrases so the "Why" text in the dashboard doesn't
  become an unreadable wall of words for a comment that trips many
  lexicons at once.

**Wrapping up:**
```python
    result.model_version = MODEL_VERSION
    if USING_TRAINED_MODEL:
        result.confidence = round((result.intent_confidence + abs(result.sentiment_score if result.sentiment != "Neutral" else 0.7)) / 2, 2)
    else:
        result.confidence = round(min(0.97, 0.5 + 0.08 * len(reasons)), 2)

    if reasons:
        result.reason_summary = f"Flagged {result.priority} priority — " + "; ".join(reasons) + "."
    else:
        result.reason_summary = f"No risk signals found; routine {result.sentiment.lower()} comment."

    return result
```
`result.model_version = MODEL_VERSION` — set *now*, after
`load_trained_models()` has definitely run, overwriting the dataclass's
stale default (see section 3.8). `confidence` is a single overall "how sure
are we" number — for the ML path, it averages intent confidence and
sentiment confidence; for the fallback path, it scales up with how many
independent rule-based signals fired (more corroborating evidence = more
confidence). `reason_summary` joins all the collected `reasons` strings
into the one sentence shown throughout the dashboard.

### 3.11 The second-person heuristic — `_mentions_second_person()`

```python
_SECOND_PERSON_MARKERS = [
    "you", "you're", "youre", "ur", "your", "u r", "u are", " u ", "@you",
]

def _mentions_second_person(norm: str) -> bool:
    return any(re.search(r'\b' + re.escape(m).replace(r'\ ', r'\s+') + r'\b', norm) for m in _SECOND_PERSON_MARKERS)
```
A cheap proxy (not real grammar analysis) for "is this comment addressing
someone directly?" It checks whether any second-person marker appears as a
whole word/phrase. `.replace(r'\ ', r'\s+')` handles multi-word markers
like `"u r"` — after `re.escape`, the literal space becomes `\ `, and this
replaces it with `\s+` (one-or-more whitespace characters) so `"u   r"`
(extra spaces) still matches. `any(... for m in ...)` returns `True` the
moment *any* marker matches, without checking the rest (short-circuit
evaluation — efficient, though at this scale performance barely matters).

### 3.12 `analyze_dataframe()` — running this over a whole CSV

```python
def analyze_dataframe(df, text_col="comment", author_col="author", platform="unknown"):
    import pandas as pd

    records = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        author = str(row.get(author_col, "Unknown")) if author_col in df.columns else "Unknown"
        likes = int(row.get("like_count", 0)) if "like_count" in df.columns and str(row.get("like_count", "")).strip() != "" else 0
        analysis = analyze_comment(text, author=author, platform=platform, like_count=likes)
        records.append(analysis.to_dict())

    analysis_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), analysis_df.drop(columns=["comment", "author"], errors="ignore")], axis=1)
    return out
```
This is what the dashboard actually calls — not `analyze_comment()`
directly. `df.iterrows()` loops over every row of the CSV **one at a
time**, calling `analyze_comment()` on each (this is the performance
bottleneck discussed in section 10 — no batching happens here today).
`row.get(text_col, "")` safely handles a missing column by defaulting to
an empty string rather than raising a `KeyError`. At the end:
`pd.concat([...], axis=1)` glues the *original* CSV columns (video ID,
timestamp, like count, etc.) side-by-side with the *new* analysis columns
(sentiment, intent, priority...), so you get one wide table with
everything. `.drop(columns=["comment", "author"], errors="ignore")` avoids
duplicate `comment`/`author` columns, since those already exist in the
original `df`.

### 3.13 `llm_explain()` — the optional AI-written explanation

```python
def llm_explain(comment_text: str, analysis: CommentAnalysis) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    import requests
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
        prompt = (
            "You are a content-moderation assistant. In 1-2 short sentences, explain in plain "
            "English why the following comment was flagged..."
        )
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    return None
```
A completely separate, **optional** third AI call — not the trained BERT
model, and not VADER. It only runs when explicitly asked (called from
`alerts.py`, only for comments that are actually about to get a Jira
ticket, to keep API usage low) and only if a `GEMINI_API_KEY` environment
variable is set. It sends the comment plus everything already computed
(sentiment, intent, key phrases) to Google's Gemini API and asks for a
short, human-readable sentence. If anything fails — no key, network error,
bad response shape — it returns `None` and the caller falls back to the
already-computed `reason_summary` instead. This is a good pattern to
notice: an *optional enhancement* should never be able to break the core
feature it's enhancing.
## 4. Deep dive: `models/train_intent_model.py` — how the model was actually trained

This is the script you run *once* (or whenever you want to retrain) to
produce the `.pt` file that `analyzer.py` loads. Training is a completely
different process from prediction — prediction is "use the learned
numbers"; training is "*find* good numbers by trial and error." Let's walk
through it.

### 4.1 Imports

```python
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
```

- `argparse` — lets you run `python train_intent_model.py --epochs 10
  --batch_size 16` from the terminal, instead of editing the script every
  time you want different settings.
- `Path` (from `pathlib`) — modern, cross-platform way to build file paths
  (nicer than string-concatenating `"models/" + "multitask_bert"`).
- `sklearn.model_selection.train_test_split` — randomly splits your data
  into a training set and a validation set.
- `sklearn.utils.class_weight.compute_class_weight` — computes how much
  extra "attention" to give to under-represented classes (explained in
  4.6).
- `torch.optim.AdamW` — the specific optimization algorithm used to update
  the model's weights during training (explained in 4.7).
- `get_linear_schedule_with_warmup` — controls how the learning rate
  changes over the course of training (explained in 4.7).

### 4.2 Reproducibility — `seed_everything()`

```python
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```
Training involves a lot of randomness: which rows go into the training
split vs validation split, how the model's weights are initialized, the
order data is shuffled in. If you don't control that randomness, running
the exact same script twice gives you a *slightly different* model each
time, which makes debugging and comparing results confusing. Setting a
fixed **seed** (here, whatever number is passed in, default `42`) for every
random-number source Python/NumPy/PyTorch use makes the "randomness"
actually reproducible — same seed, same result, every run.

### 4.3 Cleaning text before training — `clean_text()`

```python
def clean_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " USER ", text)
    text = re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
```
- `str(value).lower()` — handles the case where a CSV cell might not
  technically be a string (e.g. pandas reading a number), and lowercases
  everything.
- `re.sub(r"https?://\S+|www\.\S+", " URL ", text)` — replaces any full URL
  with the literal placeholder word `"URL"`. Why replace instead of just
  deleting? Because the *presence* of a link might be meaningful signal
  (e.g. correlated with spam), but the specific link itself is just noise
  that would blow up the model's vocabulary with one-off garbage tokens.
- `re.sub(r"@[A-Za-z0-9_]+", " USER ", text)` — same idea for @mentions →
  `"USER"`.
- `re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)` — strips the `#` off hashtags
  but *keeps* the word itself (`#amazing` → `amazing`), since hashtag text
  is often meaningful (unlike a random username).
- `re.sub(r"\s+", " ", text).strip()` — collapses multiple spaces/newlines
  into a single space and trims the ends.

**Important beginner note:** this `clean_text()` is used during *training*
to normalize the CSV data. It is **not** used by `analyzer.py` at
prediction time on the ML path — the trained BERT tokenizer handles casing
and tokenization itself. Training-time cleaning and prediction-time input
should ideally match as closely as possible; keep this in mind if you ever
notice a mismatch (it's a common, subtle source of accuracy loss in real
projects — "training/serving skew").

### 4.4 Loading and validating the CSV — `load_data()`

```python
def load_data(csv_path: str, text_col: str, sentiment_col: str, intent_col: str):
    df = pd.read_csv(csv_path)

    required = {text_col, sentiment_col, intent_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Available columns: {list(df.columns)}")
```
Fails loudly and immediately with a clear error message if the CSV doesn't
have the expected columns, rather than crashing confusingly deep inside
training 20 minutes later. `required - set(df.columns)` is Python **set
subtraction** — "everything required that isn't in the actual columns."

```python
    df = df[[text_col, sentiment_col, intent_col]].copy()
    df[text_col] = df[text_col].map(clean_text)
    df[sentiment_col] = df[sentiment_col].astype(str).str.strip().str.lower()
    df[intent_col] = df[intent_col].astype(str).str.strip().str.lower()

    df = df.dropna().drop_duplicates(subset=[text_col])
    df = df[df[text_col].str.len() > 0]
    df = df[df[sentiment_col].str.len() > 0]
    df = df[df[intent_col].str.len() > 0].reset_index(drop=True)
```
- Keep only the 3 columns actually needed, `.copy()` to avoid a pandas
  "SettingWithCopyWarning" later when modifying it.
- Clean the text column, normalize the label columns (strip whitespace,
  lowercase) so `"Positive"`, `"positive "`, and `"POSITIVE"` all become
  the same class instead of three different ones by accident.
- `dropna()` removes rows with missing values; `drop_duplicates(subset=[text_col])`
  removes exact duplicate comments (a comment appearing twice shouldn't be
  double-counted, and could otherwise leak into *both* the train and
  validation split, silently inflating validation accuracy).
- The three `df[df[col].str.len() > 0]` lines throw out any row where a
  column ended up as an empty string after cleaning.
- `.reset_index(drop=True)` — after all that filtering, pandas' row index
  has gaps (e.g. 0, 1, 4, 7...); this renumbers it cleanly 0, 1, 2, 3...

```python
    default_sentiment_order = ["negative", "neutral", "positive"]
    present_sentiments = sorted(df[sentiment_col].unique())
    sentiment_order = [s for s in default_sentiment_order if s in present_sentiments] or present_sentiments
    sentiment_to_id = {label: i for i, label in enumerate(sentiment_order)}

    intent_order = sorted(df[intent_col].unique())
    intent_to_id = {label: i for i, label in enumerate(intent_order)}
```
Neural networks output *numbers*, not words — so every unique label string
needs to be assigned a fixed integer ID. `sentiment_to_id` forces a
consistent order (`negative=0, neutral=1, positive=2`) whenever the CSV
actually contains those three words, rather than whatever arbitrary order
`.unique()` happens to return — this makes label `1` always mean "neutral"
across different training runs. Intent classes get sorted alphabetically
and numbered 0, 1, 2... in that alphabetical order (there's no natural
fixed order like sentiment has).

```python
    sent_counts = df[sentiment_col].value_counts()
    intent_counts = df[intent_col].value_counts()

    if sent_counts.min() < 2:
        raise ValueError("Every sentiment class needs at least 2 examples for a stratified split. ...")
    if intent_counts.min() < 2:
        raise ValueError("Every intent class needs at least 2 examples for a stratified split. ...")
```
`train_test_split(..., stratify=...)`, used later, needs *at least 2*
examples of every class to be able to put at least 1 in training and 1 in
validation. This check fails fast with a clear message instead of letting
scikit-learn throw a confusing error later.

```python
    df["sentiment_label"] = df[sentiment_col].map(sentiment_to_id).astype(int)
    df["intent_label"] = df[intent_col].map(intent_to_id).astype(int)
```
Adds two new integer columns — these are what the model will actually be
trained to predict, using the ID mapping built above.

### 4.5 Feeding data to PyTorch — `MultiTaskDataset`

```python
class MultiTaskDataset(Dataset):
    def __init__(self, texts, sentiment_labels, intent_labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.sentiment_labels = list(sentiment_labels)
        self.intent_labels = list(intent_labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment_label": torch.tensor(self.sentiment_labels[index], dtype=torch.long),
            "intent_label": torch.tensor(self.intent_labels[index], dtype=torch.long),
        }
```
`Dataset` is a PyTorch base class with a strict contract: implement
`__len__` (how many items total) and `__getitem__` (given an index, return
one item). PyTorch's `DataLoader` (used next) relies on this contract to
automatically handle shuffling and batching for you — you never have to
write that looping/batching logic by hand.

- `__getitem__` tokenizes **one comment at a time**, on demand, rather than
  tokenizing the entire dataset upfront — this keeps memory usage low even
  for a huge CSV, since only the current batch's worth of text is ever
  tokenized at once.
- `.squeeze(0)` — the tokenizer always returns tensors shaped for a batch
  (even a batch of 1), so `input_ids` comes out shaped `[1, 128]`.
  `squeeze(0)` removes that extra leading dimension of size 1, leaving a
  clean `[128]` — `DataLoader` will re-add the batch dimension itself when
  it groups several of these together.
- `dtype=torch.long` — PyTorch's loss functions expect integer class labels
  as the 64-bit integer type called `long`, not floats.

### 4.6 The model — same architecture as `analyzer.py`, but trainable

```python
class MultiTaskClassifier(nn.Module):
    def __init__(self, num_sentiment_labels: int, num_intent_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)
        self.intent_head = nn.Linear(hidden_size, num_intent_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.sentiment_head(pooled), self.intent_head(pooled)
```
This is **identical in shape** to the class rebuilt inside `analyzer.py`
(section 3.3) — that's not a coincidence, it's a requirement. When you
later do `model.load_state_dict(saved_weights)`, PyTorch checks that every
layer name and every tensor shape in `saved_weights` matches exactly what
the class defines. If `analyzer.py`'s version drifted out of sync with this
one (e.g. someone changed `dropout=0.3` to `dropout=0.5`, or renamed a
layer), loading would fail — this is exactly the bug that existed in the
project before it was fixed (see the "Custom Model Wiring" documentation
section).

**Why class weights (`compute_class_weight`)?** Real-world comment data is
almost always **imbalanced** — you'll naturally have far more
`general_discussion` comments than `fraudulent_service_offer` ones. If you
train without correcting for this, the model can reach high *overall*
accuracy just by always predicting the majority class and effectively
ignoring the rare-but-important ones. `compute_class_weight("balanced",
...)` calculates a multiplier per class (rare classes get a bigger
multiplier) so mistakes on rare classes are penalized more heavily during
training, forcing the model to actually pay attention to them:

```python
    sentiment_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(sentiment_to_id)), y=train_df["sentiment_label"])
    intent_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(intent_to_id)), y=train_df["intent_label"])
    sentiment_criterion = nn.CrossEntropyLoss(weight=torch.tensor(sentiment_weights, dtype=torch.float32, device=device))
    intent_criterion = nn.CrossEntropyLoss(weight=torch.tensor(intent_weights, dtype=torch.float32, device=device))
```
`nn.CrossEntropyLoss` is the standard **loss function** for multi-class
classification — it measures "how wrong were the model's predicted
probabilities, compared to the true answer," and that single number (the
**loss**) is exactly what training tries to minimize. `weight=...` plugs
the class-imbalance correction directly into that measurement.

### 4.7 The training loop — where the actual "learning" happens

```python
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
```
- **`optimizer`** — the algorithm that actually *changes* the model's
  weights, using the gradients computed from the loss. `AdamW` is the
  standard, well-tested choice for training transformer models like BERT.
  `lr=2e-5` (the default) is the **learning rate** — how big a step to take
  each update; too high and training becomes unstable/never converges, too
  low and training takes forever. `weight_decay=0.01` is a mild penalty
  that discourages weights from growing unnecessarily large, another
  overfitting-prevention technique (like dropout, but applied differently).
- **`scheduler`** — the learning rate isn't kept constant throughout
  training. `get_linear_schedule_with_warmup` starts the learning rate low,
  **ramps it up** for the first 10% of training steps (`"warmup"` — jumping
  straight to full speed on a randomly-initialized model tends to
  destabilize training), then **linearly decreases** it back toward zero
  for the rest of training (fine-tuning with smaller and smaller
  adjustments as training progresses, similar to how you'd make finer
  corrections as you get close to a target).

```python
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)
            intent_labels = batch["intent_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            sentiment_logits, intent_logits = model(input_ids, attention_mask)

            loss = (
                args.sentiment_loss_weight * sentiment_criterion(sentiment_logits, sentiment_labels)
                + args.intent_loss_weight * intent_criterion(intent_logits, intent_labels)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
```
This is the heart of machine learning — walk through it slowly:

- **`for epoch in range(args.epochs)`** — an **epoch** means "one complete
  pass through the entire training dataset." Doing multiple epochs (5, by
  default) lets the model see the same data several times, refining its
  weights a little more each pass.
- **`model.train()`** — the opposite of `model.eval()` from section 3.4:
  turns dropout *on* and tells other layers to behave in "training mode."
- **`for batch in train_loader`** — `DataLoader` automatically groups the
  dataset into chunks of `batch_size` (8, by default) and shuffles their
  order every epoch. Processing several examples at once (a "batch")
  rather than one-by-one is both faster (more parallelism) and produces
  more stable training (the loss/gradient is averaged over several
  examples instead of swinging wildly based on just one).
- **`.to(device)`** — moves the tensors onto whichever device (`"cuda"` for
  GPU, `"cpu"` otherwise) the model itself lives on — a tensor on the CPU
  and a model on the GPU can't interact; they must match.
- **`optimizer.zero_grad(set_to_none=True)`** — PyTorch *accumulates*
  gradients by default (adds new ones on top of old ones) rather than
  overwriting them — a design choice useful for some advanced techniques,
  but here we want a clean slate before each new batch, so we explicitly
  clear them first. Forgetting this line is one of the most common PyTorch
  training bugs — gradients from previous batches would silently leak into
  the current update.
- **`sentiment_logits, intent_logits = model(input_ids, attention_mask)`**
  — the **forward pass**: run the batch through the network and get raw
  predictions for both tasks (same `forward()` method explained in section
  3.3).
- **`loss = weight1 * sentiment_loss + weight2 * intent_loss`** — since one
  network is predicting *two* different things, you need one combined
  number to optimize. This just adds the two task losses together
  (optionally weighted differently via `--sentiment_loss_weight` /
  `--intent_loss_weight`, both `1.0` by default, i.e. equal importance).
- **`loss.backward()`** — this is **backpropagation**: PyTorch
  automatically works out, for every single one of the model's millions of
  weights, "if I nudge this weight up slightly, does the loss go up or
  down, and by how much?" (the gradient). This is the calculation that
  `torch.no_grad()` at prediction time deliberately skips, because it's
  only needed here, during training.
- **`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`** — **gradient
  clipping**: occasionally a batch produces an unusually huge gradient
  (e.g. from one weird outlier example), which would cause a violently
  large weight update and destabilize training. This caps the total
  gradient size at `1.0`, protecting against that.
- **`optimizer.step()`** — *now* actually update every weight in the
  network, using the gradients just computed and the AdamW algorithm's
  rules.
- **`scheduler.step()`** — advance the learning-rate schedule by one step
  (see the warmup/decay explanation above).
- **`losses.append(loss.item())`** — just bookkeeping, to report the
  average loss for this epoch afterward.

### 4.8 Evaluating and saving the best checkpoint

```python
        metrics = evaluate(model, val_loader, device, id_to_sentiment, id_to_intent, output_dir, "validation")
        combined_f1 = (metrics["sentiment"]["macro_f1"] + metrics["intent"]["macro_f1"]) / 2
        ...
        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            torch.save(model.state_dict(), output_dir / "multitask_model.pt")
            print("  New best model saved!")
```
After *every* epoch, the model is evaluated on the validation set (data it
never trained on — see `evaluate()` below) and its **F1 score** is
computed. F1 is a single number balancing precision (when the model says
"complaint," how often is it actually a complaint?) and recall (of all the
*actual* complaints, how many did the model catch?) — generally a better
metric than raw accuracy for imbalanced classes, since a model that just
always guesses the majority class can still get high *accuracy* while
having terrible F1 on the classes that matter.

Crucially: the model is only saved to disk (`torch.save`) if this epoch's
combined F1 is *better* than every previous epoch's. This is called
**checkpointing on the best epoch** — it protects against a subtle problem:
training for too many epochs can cause the model to start *overfitting*
(memorizing training examples instead of generalizing), at which point
validation performance actually gets *worse* even as training loss keeps
going down. By only ever keeping the best-scoring version, the final saved
`.pt` file is the best-generalizing checkpoint, not necessarily the
*last-trained* one.

```python
def evaluate(model, loader, device, id_to_sentiment, id_to_intent, output_dir, name):
    model.eval()
    sent_true, sent_pred, intent_true, intent_pred = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            ...
            sentiment_logits, intent_logits = model(input_ids, attention_mask)
            sent_true.extend(batch["sentiment_label"].numpy())
            sent_pred.extend(sentiment_logits.argmax(dim=1).cpu().numpy())
            intent_true.extend(batch["intent_label"].numpy())
            intent_pred.extend(intent_logits.argmax(dim=1).cpu().numpy())
```
Notice this uses `model.eval()` and `torch.no_grad()` — exactly the same
"prediction mode" settings used in `analyzer.py`, because evaluation *is*
prediction, just on data where we already know the right answer, so we can
score how well the model did. It collects every true label and every
predicted label across the whole validation set into plain lists.

```python
    def summarize(true, pred, id_to_label, task_name):
        ...
        report = classification_report(true, pred, labels=label_ids, target_names=target_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(true, pred, labels=label_ids)
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(output_dir / f"{name}_{task_name}_confusion_matrix.csv")
        return {"accuracy": ..., "macro_f1": ..., "classification_report": report, "confusion_matrix": cm.tolist()}
```
`classification_report` (from scikit-learn) computes precision/recall/F1
**per class**, not just one overall number — this is what lets you spot
"the model is great at detecting praise but terrible at detecting scams."
`confusion_matrix` builds a grid showing, for every true class, what the
model actually predicted instead — e.g. it might reveal the model
regularly confuses `financial_promotion` with `advertisement_or_promotion`
(which makes intuitive sense, since they're conceptually similar). Both
get written to disk (`*_confusion_matrix.csv`, `*_metrics.json`) so you can
inspect them later without re-running training.

### 4.9 Saving everything needed for later use

```python
    model.load_state_dict(torch.load(output_dir / "multitask_model.pt", map_location=device))
    evaluate(model, val_loader, device, id_to_sentiment, id_to_intent, output_dir, "final_validation")

    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "sentiment_labels.json").write_text(json.dumps({str(k): v for k, v in id_to_sentiment.items()}, indent=2))
    (output_dir / "intent_labels.json").write_text(json.dumps({str(k): v for k, v in id_to_intent.items()}, indent=2))
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
```
- Reload the *best* checkpoint (not necessarily the one from the final
  epoch) and run one last, clearly-labeled evaluation on it
  (`final_validation_*`) — this is the number you should actually trust as
  "how good is this model," and it's exactly the file
  (`final_validation_metrics.json`) flagged as suspicious in section 9.
- `tokenizer.save_pretrained(...)` — saves the tokenizer's own config/vocab
  files, so prediction-time code uses the *exact* same tokenization rules
  as training (see section 3.4's note on why this matters).
- The label mapping dictionaries get saved as JSON — this is what lets
  `analyzer.py` translate the model's numeric output back into human words
  without needing to duplicate this mapping logic anywhere else.
- `training_history.csv` — one row per epoch, so you can plot loss/accuracy
  over time afterward and visually check things like "did the model
  actually converge, or was it still improving when training stopped?"

### 4.10 Command-line arguments — `argparse`

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="models/comments_sentiment.csv")
    parser.add_argument("--text_col", default="comment_text")
    parser.add_argument("--sentiment_col", default="sentiment")
    parser.add_argument("--intent_col", default="intent")
    parser.add_argument("--output_dir", default="models/multitask_bert")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--sentiment_loss_weight", type=float, default=1.0)
    parser.add_argument("--intent_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
```
`if __name__ == "__main__":` is standard Python for "only run this block if
the file is executed directly (`python train_intent_model.py`), not if it's
imported by another file." Everything below defines what flags you can pass
on the command line, e.g.:
```
python models/train_intent_model.py --csv_path my_data.csv --epochs 8 --batch_size 16
```
`validation_size=0.2` means 20% of your data is held out for validation,
80% used for training — the standard default split ratio.
## 5. What is a `.pt` file, and CPU vs GPU — explained properly

### 5.1 What `torch.save(model.state_dict(), "multitask_model.pt")` actually writes

A neural network, stripped down, is just **a huge pile of numbers**
(weights and biases) arranged into named groups (layers). `model.state_dict()`
returns a Python dictionary where each key is a layer's name (e.g.
`"bert.encoder.layer.5.attention.self.query.weight"`) and each value is a
PyTorch tensor (a multi-dimensional array of numbers) — the actual learned
values for that layer. `torch.save(...)` serializes that dictionary to disk
using Python's `pickle` format, wrapped inside a zip archive (you can
literally open a `.pt` file with any zip tool and see internal folders like
`data/0`, `data/1`... — each one is one tensor's raw bytes).

**A `.pt` file is data, not code.** It has no idea what a "BERT model" is —
it's just numbers with names. That's precisely why `analyzer.py` needs its
own `MultiTaskClassifier` class (the empty "mold," section 3.3): the
`.pt` file tells PyTorch "put this number here, this number here..." but
only *if* there's already an object with matching names and shapes to pour
those numbers into. `model.load_state_dict(state_dict)` is that pouring
step, and it's why the architecture in `analyzer.py` must match the
architecture in `train_intent_model.py` exactly.

### 5.2 CPU vs GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskClassifier(...).to(device)
```
Training a transformer means doing an enormous number of matrix
multiplications — repeated for every layer, every batch, every epoch. A
**CPU** has a small number of very fast, general-purpose cores — great for
varied, sequential tasks. A **GPU** has thousands of smaller, simpler cores
built specifically to do the *same* operation (like multiplying numbers)
on massive amounts of data simultaneously — exactly what matrix
multiplication needs. That's why GPU training is commonly 10–50x faster
than CPU training for the same model.

`torch.cuda.is_available()` checks whether the machine has an NVIDIA GPU
with the CUDA drivers PyTorch needs. If not, `device` falls back to
`"cpu"` — training still works, just slower. This project's training was
run and produced valid results either way; the important thing to
understand is that **the resulting `.pt` file works identically regardless
of which device trained it** — it's just numbers. That's why
`analyzer.py` can safely always load with `map_location="cpu"`
(section 3.4) even if the model happened to be trained on a GPU — you're
just choosing which device to put those same numbers on for *prediction*.

### 5.3 Why training creates so many files inside `multitask_bert/`

Recap, now with the training code behind each one:

| File | Produced by (section) | What it is |
|---|---|---|
| `multitask_model.pt` | 4.8, `torch.save(model.state_dict(), ...)` | The actual learned weights — the "brain" |
| `tokenizer/*` | 4.9, `tokenizer.save_pretrained(...)` | The exact word-splitting rules used during training — must match at prediction time |
| `sentiment_labels.json` / `intent_labels.json` | 4.9, `id_to_sentiment` / `id_to_intent` dicts written as JSON | Translates the model's number output back into words |
| `training_history.csv` | 4.9, `pd.DataFrame(history).to_csv(...)` | One row per epoch: loss, accuracy, F1 — shows whether training was still improving or had plateaued |
| `validation_*_metrics.json` | 4.8, `evaluate()`'s `result` dict written to JSON | Per-class precision/recall/F1 on data the model never trained on — the honest report card |
| `*_confusion_matrix.csv` | 4.8, `confusion_matrix()` written to CSV | For each true class, what the model actually guessed — reveals *which* classes get confused with which |

---

## 6. Deep dive: `src/redis_client.py` — talking to Upstash Redis

### 6.1 Why Upstash specifically

Upstash offers Redis (a fast key-value store, normally accessed via its own
binary network protocol) over a plain HTTPS REST API instead. That matters
here because it means this whole file only ever needs Python's `requests`
library — no separate Redis server to install/run, no extra network
protocol/port to configure. Good fit for a project like this one, deployed
simply.

```python
class UpstashRedis:
    def __init__(self):
        self.base_url = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
        self.token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

    def is_configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}
```
Credentials come from environment variables, not hardcoded — standard
practice so secrets never end up committed to source control.
`is_configured()` is checked before every real operation throughout the
codebase — this is the mechanism behind "Redis not configured? The app
still works, just without caching/dedupe" (graceful degradation, mentioned
in the file's own docstring).

```python
    def _call(self, *segments) -> dict | None:
        if not self.is_configured():
            return None
        try:
            path = "/".join(str(s) for s in segments)
            resp = requests.get(f"{self.base_url}/{path}", headers=self._headers(), timeout=8)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            return None
        return None
```
Upstash's REST API lets you run Redis commands by hitting a URL shaped like
`base_url/COMMAND/ARG1/ARG2`. `*segments` (variadic arguments) lets callers
write `self._call("get", key)` or `self._call("incr", key)` naturally.
Wrapped in `try/except` so any network hiccup returns `None` instead of
crashing the caller — every method built on top of `_call()` treats `None`
as "couldn't reach Redis, behave as if this key doesn't exist."

```python
    def _call_post(self, path: str, body: str, timeout: int = 15) -> dict | None:
        ...
        resp = requests.post(f"{self.base_url}/{path}", headers=self._headers(), data=body.encode("utf-8"), timeout=timeout)
```
A second, separate method for sending the value in the **request body**
instead of the URL. This exists specifically for large payloads — putting
a whole analyzed dataset's JSON inside a URL would either get rejected
(URLs have practical length limits, commonly a few KB to ~8KB depending on
server config) or silently truncated. Sending it as a POST body has no
such length restriction.

```python
    def get(self, key: str):
        result = self._call("get", key)
        return result.get("result") if result else None

    def set(self, key: str, value, ex_seconds: int | None = None):
        ...
        result = self._call("set", key, value)
        if ex_seconds:
            self._call("expire", key, ex_seconds)
        return bool(result)
```
Upstash always wraps its response in `{"result": ...}` — these methods
unwrap that. `set()` optionally follows up with a separate `expire` call
to attach a **TTL (time-to-live)** — after `ex_seconds` seconds, Redis
automatically deletes the key. This is what makes the cache "self-cleaning"
— you never have to manually clear out old entries.

```python
    def set_raw(self, key: str, text: str, ex_seconds: int | None = None) -> bool:
        path = f"set/{key}"
        if ex_seconds:
            path += f"?EX={ex_seconds}"
        result = self._call_post(path, text)
        return bool(result and result.get("result") == "OK")

    def get_raw(self, key: str) -> str | None:
        return self.get(key)
```
`set_raw`/`get_raw` are the large-value pair used specifically for caching
whole analyzed datasets (used from `dashboard.py`, section 8). Note the TTL
here is passed as a `?EX=` **query parameter** instead of a follow-up
`expire` call — Upstash's `SET` command supports an inline expiry option
this way, saving a second round-trip.

```python
    def setnx_with_ttl(self, key: str, value, ex_seconds: int) -> bool:
        if not self.is_configured():
            return True
        existing = self.get(key)
        if existing is not None:
            return False
        self.set(key, value, ex_seconds=ex_seconds)
        return True
```
This is the **alert de-duplication** mechanism (used in `alerts.py`).
"NX" is Redis shorthand for "only set if the key does **N**ot e**X**ist."
Check if the key is already there; if so, this comment has already been
alerted on recently — return `False` ("not new"), don't file a second
ticket. If not, claim the key (with a TTL, so after `ex_seconds` the same
comment *could* re-alert if it reappears, e.g. resurfaces in a later scrape)
and return `True` ("genuinely new, go ahead and alert"). Note the fallback:
if Redis isn't configured at all, this always returns `True` — no dedupe
store available means every run is treated as fresh, safer than silently
suppressing real alerts.

```python
    def incr(self, key: str) -> int | None:
        result = self._call("incr", key)
        return result.get("result") if result else None
```
Redis's `INCR` command atomically increments a counter by 1 (creating it at
`0` first if it doesn't exist) — this is what powers the "negative alerts
today" counter (`stats:negative_alerts:2026-08-08`), incremented once per
new alert filed.

### 6.2 Singleton pattern

```python
_client_singleton: UpstashRedis | None = None

def get_client() -> UpstashRedis:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = UpstashRedis()
    return _client_singleton
```
Same lazy-singleton pattern as the ML model in `analyzer.py` — create the
`UpstashRedis` object once, reuse it everywhere. This is the standard way
`dashboard.py` and `alerts.py` both get access to *the same* configured
client without either file needing to construct or pass one around
manually.

---

## 7. Deep dive: `src/alerts.py` — deciding what becomes a Jira ticket

```python
PRIORITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

ACTIONABLE_INTENTS = {
    "complaint_or_problem_report",
    "fraudulent_service_offer",
    "financial_promotion",
    "giveaway_or_reward_scam",
}

def _is_jira_worthy(row) -> bool:
    if row.get("toxicity_severity") == "severe":
        return True
    intent = row.get("intent", "")
    if intent in ACTIONABLE_INTENTS:
        return True
    if intent == "user_experience_feedback" and row.get("sentiment") == "Negative":
        return True
    return False
```
`PRIORITY_ORDER` turns the text labels ("Low"/"Medium"/"High"/"Critical")
into comparable numbers, since you can't do `"High" >= "Medium"` directly
in Python the way you'd want (string comparison is alphabetical, not
severity-based). `_is_jira_worthy()` is the **stricter gate** discussed in
depth earlier in this conversation — priority score alone answers "should
a human skim this in the dashboard," this function answers the separate
question "is there something *actionable* here." `row.get("key", default)`
is used everywhere instead of `row["key"]` so that a row from an
older-schema cached dataset (missing a newer column) doesn't crash the
whole batch — it just falls back to a safe default and evaluates as "not
Jira-worthy" for that missing signal.

```python
def _comment_hash(author: str, comment: str) -> str:
    raw = f"{author.strip().lower()}::{comment.strip().lower()}"
    return "alert:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
```
Builds a **stable, deterministic ID** for a given author+comment pair, used
as the Redis dedupe key. `hashlib.sha256(...)` turns arbitrary text into a
fixed-length string of hex characters (a "hash") — the same input always
produces the same hash, and different inputs almost never collide.
`[:24]` just trims it to 24 characters — plenty unique for this purpose,
and shorter keys are marginally cheaper to store/transmit. Lowercasing
both fields before hashing means "John" and "john" (or a comment with
different capitalization somehow re-scraped) still hash identically and
correctly dedupe.

```python
def _load_local_log() -> list:
    if os.path.exists(ALERT_LOG_PATH):
        try:
            with open(ALERT_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_local_log(entries: list):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ALERT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries[-500:], f, indent=2, ensure_ascii=False)
```
A local JSON file (`data/alerts_log.json`) acting as a backup audit trail —
works even if Redis/Jira aren't configured at all, so you never lose
visibility into what got flagged. `entries[-500:]` — Python slicing that
keeps only the **last 500** entries, a simple, unbounded-growth guard so
this file doesn't grow forever.

### 7.1 `process_alerts()` — the main function, step by step

```python
def process_alerts(analyzed_df, platform: str, use_llm: bool = True, max_llm_calls: int = 5) -> list:
    threshold = os.getenv("ALERT_PRIORITY_THRESHOLD", "High")
    ttl_hours = int(os.getenv("ALERT_DEDUPE_TTL_HOURS", "24"))
    threshold_rank = PRIORITY_ORDER.get(threshold, 2)

    rc = redis_client.get_client()
    jc = jira_client.get_client()

    candidates = analyzed_df[analyzed_df["priority"].map(lambda p: PRIORITY_ORDER.get(p, 0) >= threshold_rank)]
    candidates = candidates[candidates.apply(_is_jira_worthy, axis=1)]
```
Both the priority threshold and the dedupe window are configurable via
environment variables (with sensible defaults) rather than hardcoded —
lets you tune sensitivity without touching code. `analyzed_df[...]` is
pandas **boolean-mask filtering**: `.map(lambda p: ...)` produces a
True/False value for every row, and putting that inside `df[...]` keeps
only the rows where it's `True`. Two filters are chained: first "is the
priority score high enough to matter," then (from `.apply(_is_jira_worthy,
axis=1)`) "is it actually actionable" — `axis=1` tells `.apply()` to pass
each *row* (not each column) to the function.

```python
    for _, row in candidates.iterrows():
        author = str(row.get("author", "Unknown"))
        comment = str(row.get("comment", ""))
        key = _comment_hash(author, comment)

        is_new = rc.setnx_with_ttl(key, "1", ex_seconds=ttl_hours * 3600)
```
For every surviving candidate, build its dedupe key and atomically check
"have we already alerted on this exact comment recently?" `ex_seconds=
ttl_hours * 3600` converts the configured hours into seconds (Redis TTLs
are always in seconds).

```python
        explanation = row.get("reason_summary", "")
        if use_llm and is_new and llm_calls_used < max_llm_calls:
            from .analyzer import CommentAnalysis
            pseudo = CommentAnalysis(comment=comment, author=author, platform=platform, ...)
            llm_text = llm_explain(comment, pseudo)
            if llm_text:
                explanation = llm_text
                llm_calls_used += 1
```
Defaults to the already-computed, free `reason_summary`. *Only* for genuinely
new alerts (no point spending an API call explaining a duplicate we're not
even going to file), and only up to `max_llm_calls` (default 5) per batch —
a deliberate cost/rate-limit control — it optionally asks Gemini
(`llm_explain()`, section 3.13) for a nicer, natural-language explanation
instead. `pseudo = CommentAnalysis(...)` rebuilds a lightweight version of
the original analysis object from the DataFrame row, since `llm_explain()`
expects that object type, not a raw pandas row.

```python
        jira_result = {"ok": False, "key": None, "url": None, "error": "skipped (duplicate or dry-run)"}
        if is_new:
            summary = f"[{platform.title()}] {row.get('priority', 'High')} priority comment from @{author}"
            description = (
                f"Platform: {platform}\nAuthor: {author}\n"
                f"Sentiment: {row.get('sentiment')} ({row.get('sentiment_score')})\n"
                f"Intent: {row.get('intent')}\nPriority score: {row.get('priority_score')}/100\n\n"
                f'Comment: "{comment}"\n\nWhy flagged: {explanation}'
            )
            jira_result = jc.create_issue(summary=summary, description=description,
                                           priority=row.get("priority", "High"),
                                           labels=["comment-intelligence", platform, str(row.get("intent", "complaint"))])
            rc.incr(f"stats:negative_alerts:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
```
Only *actually files a Jira ticket* if this is a genuinely new alert
(`is_new`) — this is the safety valve preventing duplicate tickets when
you re-run analysis on data you've already processed before. The
description string is built with an f-string spanning multiple lines
(`\n` for newlines), giving whoever picks up the ticket all the context
needed without opening the dashboard. The daily counter is only
incremented for genuinely-new, genuinely-filed alerts — not duplicates,
not skipped ones.

```python
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform, "author": author, "comment": comment,
            "sentiment": row.get("sentiment"), "intent": row.get("intent"),
            "priority": row.get("priority"), "priority_score": int(row.get("priority_score", 0)),
            "explanation": explanation, "is_new": is_new,
            "jira_ok": jira_result["ok"], "jira_key": jira_result["key"],
            "jira_url": jira_result["url"], "jira_error": jira_result["error"],
        }
        alerts.append(alert)
        if is_new:
            log.append(alert)

    _save_local_log(log)
    return alerts
```
Every candidate (new or duplicate) gets recorded in the `alerts` list that
this function returns — the dashboard uses this to show *all* flagged
comments, including duplicates marked as such. Only genuinely new ones get
appended to the persistent local log file though, avoiding it filling up
with repeat entries.

```python
def get_today_alert_count() -> int | None:
    rc = redis_client.get_client()
    val = rc.get(f"stats:negative_alerts:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0
```
Reads back the daily counter incremented above — this is what powers a
"live risk pulse" number in the dashboard that reflects alerts filed
*today*, even across different sessions/restarts, because it's read from
Redis, not from `st.session_state`.
## 8. Deep dive: `dashboard.py` — the Streamlit web app

### 8.1 A quick mental model of Streamlit, before anything else

Streamlit is unlike a normal web framework. **There is no separate
frontend/backend split, and no "event handlers" you write yourself.**
Instead: every time the user interacts with *any* widget (clicks a button,
changes a dropdown), Streamlit **re-runs your entire Python script from
top to bottom**, and whatever gets `st.write()`/`st.markdown()`/etc'd along
the way becomes the new page. This explains several patterns you'll see
throughout this file:
- `st.session_state` is used to remember things *across* those re-runs
  (otherwise every click would wipe all previous state).
- Expensive work (like running the ML model) is carefully guarded behind
  `if` checks so it only happens when actually necessary, not on every
  single re-run.

### 8.2 Setup section

```python
import streamlit as st
import pandas as pd
import subprocess
import os
import sys
import glob
import json
import io
import hashlib
import warnings

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*proactor.*')

from dotenv import load_dotenv
load_dotenv()
```
- `subprocess` — used later to actually *run* the ingestion scripts
  (`youtube/fetch_comments.py` etc.) as separate processes.
- `glob` — used to find existing CSV files on disk (`glob.glob("data/comments_*.csv")`
  matches a wildcard pattern, like typing `comments_*.csv` in a file
  explorer's search box).
- The Windows-specific block works around a known cosmetic issue where
  Python's `asyncio` library prints harmless but noisy warnings on
  Windows — purely a quality-of-life fix, has no effect on Mac/Linux.
- `load_dotenv()` — reads a local `.env` file (API keys, Jira credentials,
  Redis credentials) and loads them into `os.environ`, so the rest of the
  code can just use `os.getenv("JIRA_API_TOKEN")` normally, without ever
  hardcoding secrets in the source.

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

from src import ui_theme
from src.preprocess import clean_text
from src.analyzer import analyze_dataframe
from src.simi import spam_similarity_score
from src.burst import burst_detect
from src.score import campaign_score, explain_campaign
from src import alerts as alerts_engine
from src.redis_client import get_client as get_redis_client
from src.jira_client import get_client as get_jira_client
```
`sys.path.insert(0, BASE_DIR)` adds the project's root folder to Python's
list of places it looks for importable modules — this is what makes
`from src import ...` work regardless of which directory you happen to
launch `streamlit run` from. `os.makedirs(DATA_DIR, exist_ok=True)`
creates the `data/` folder if it doesn't exist yet, `exist_ok=True` meaning
"don't error if it's already there."

### 8.3 Page config and session state

```python
st.set_page_config(page_title="Comment Intelligence Platform", page_icon="🛰️", layout="wide")
ui_theme.inject_css()

for key, default in [
    ("csv_path", None), ("df", None), ("post_metadata", None), ("ai_summary", None),
    ("analyzed_df", None), ("analyzed_for", None), ("last_alerts", None), ("platform_used", "unknown"),
]:
    if key not in st.session_state:
        st.session_state[key] = default
```
`st.set_page_config` must be the *first* Streamlit command in the script —
it configures browser tab title, icon, and layout width. `ui_theme.inject_css()`
injects the custom dark-theme CSS (section 9 below) once. The `for key,
default in [...]` loop is the standard Streamlit idiom for initializing
`st.session_state`: `if key not in st.session_state` means "only set this
the very first time — on every subsequent re-run, leave whatever value is
already there alone" (otherwise every re-run would reset all your stored
data back to `None`!).

### 8.4 The analysis cache functions

```python
redis_client = get_redis_client()
jira_client = get_jira_client()

ANALYSIS_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

def _dataset_cache_key(csv_path: str, model_version: str) -> str:
    try:
        stat = os.stat(csv_path)
        fingerprint = f"{os.path.basename(csv_path)}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        fingerprint = csv_path
    digest = hashlib.md5(f"{fingerprint}:{model_version}".encode("utf-8")).hexdigest()
    return f"cip:analysis:{digest}"
```
`os.stat(csv_path)` reads filesystem metadata without opening the file —
`st_size` (byte count) and `st_mtime` (last-modified timestamp). Combining
filename + size + mtime gives a cheap **fingerprint**: if the CSV changes
in any way (different content, re-fetched later), at least one of those
three values changes too, producing a different fingerprint and correctly
triggering fresh analysis instead of serving a stale cached result.
`hashlib.md5(...)` compresses that fingerprint (plus the model/rules
version) into one fixed-length, URL/key-safe string — this becomes the
actual Redis key.

```python
def _load_analyzed_df_cached(df_raw: pd.DataFrame, csv_path: str, platform: str):
    from src import analyzer as _analyzer
    _analyzer.load_trained_models()

    cache_key = _dataset_cache_key(csv_path, f"{_analyzer.MODEL_VERSION}::{_analyzer.RULES_VERSION}")

    if redis_client.is_configured():
        cached_json = redis_client.get_raw(cache_key)
        if cached_json:
            try:
                return pd.read_json(io.StringIO(cached_json), orient="records"), True
            except Exception:
                pass

    analyzed = analyze_dataframe(df_raw, platform=platform)

    if redis_client.is_configured():
        try:
            redis_client.set_raw(cache_key, analyzed.to_json(orient="records"), ex_seconds=ANALYSIS_CACHE_TTL_SECONDS)
        except Exception:
            pass

    return analyzed, False
```
This is the function that actually decides "do we need to run the (slow)
ML model, or can we skip straight to a cached answer?" Walk through the
logic: load the model first (cheap after the first call — section 3.4's
guard) purely so `MODEL_VERSION`/`RULES_VERSION` are accurate for building
the key. Check Redis; if there's a hit, `pd.read_json(io.StringIO(...))`
turns the cached JSON text back into a DataFrame and we return immediately
— `True` meaning "yes, this came from cache." `io.StringIO(cached_json)`
wraps the string in a file-like object because `pd.read_json()` expects
something file-like, not a raw string, in current pandas versions (passing
a raw string directly triggers a deprecation warning). On a cache **miss**
(nothing found, or the cached entry couldn't be parsed — the `except`
guards against a corrupted/incompatible old entry crashing the app), fall
through to `analyze_dataframe(df_raw, ...)` — the real, slow path — then
write the fresh result back to Redis for next time, and return `False`
("no, this was freshly computed"). Every Redis interaction is wrapped in
its own `try/except` — a caching failure should never be able to break the
actual analysis feature.

### 8.5 Fetching new comments — running the ingestion scripts

```python
PLATFORM_CONFIG = {
    "YouTube": {"script": os.path.join(BASE_DIR, "youtube", "fetch_comments.py"), "placeholder": "YouTube video URL or ID"},
    "Instagram": {"script": os.path.join(BASE_DIR, "instagram", "fetch_comments.py"), "placeholder": "Instagram reel/post URL"},
    "Bluesky": {"script": os.path.join(BASE_DIR, "bluesky", "fetch_comments.py"), "placeholder": "bsky.app post URL"},
}
```
A simple lookup table mapping each platform name to the script that
handles it and the placeholder text shown in the URL input box —
avoids a chain of `if platform == "YouTube": ... elif platform ==
"Instagram": ...` scattered through the UI code.

```python
if fetch_clicked:
    if platform == "Select":
        st.warning("Please select a platform")
        st.stop()
    if not url:
        st.warning("Please enter a valid URL")
        st.stop()

    script = PLATFORM_CONFIG[platform]["script"]
    with st.spinner(f"Fetching comments from {platform}..."):
        env = os.environ.copy()
        if platform == "YouTube":
            env["API_KEY"] = os.getenv("YOUTUBE_API_KEY") or os.getenv("API_KEY", "")
            cmd = [sys.executable, script, url]
        elif platform == "Instagram":
            cmd = [sys.executable, script, "--url", url]
        else:
            cmd = [sys.executable, script, "--url", url]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR, env=env)
```
`st.stop()` — a Streamlit-specific function that halts script execution at
that exact point for this run (nothing below it executes), used instead of
`return` since this isn't inside a function. `sys.executable` — the full
path to the *currently running* Python interpreter, guaranteeing the
ingestion script runs with the same Python installation/environment
(and thus the same installed libraries) as the dashboard itself, rather
than whatever `python` happens to resolve to on the system `PATH`.
`subprocess.run(cmd, capture_output=True, text=True, ...)` actually
launches the ingestion script as a **separate process**, waits for it to
finish, and captures whatever it printed to stdout/stderr as plain text
(rather than raw bytes, because of `text=True`). Running ingestion as a
subprocess (rather than importing and calling it directly) is a clean
isolation boundary — a crash in the ingestion script can't take down the
dashboard process itself.

```python
        csv_path = None
        for line in result.stdout.splitlines():
            if line.strip().endswith(".csv") and os.path.exists(line.strip()):
                csv_path = line.strip()
                break

        if not csv_path:
            st.error("❌ No data returned")
            with st.expander("🔍 Debug information"):
                st.code(result.stdout or "No stdout")
                st.code(result.stderr or "No stderr")
                st.write("Return code:", result.returncode)
            st.stop()
```
The "protocol" between the dashboard and each ingestion script is simple:
the script prints the path to the CSV it just wrote, somewhere in its
output, and the dashboard scans every line of stdout looking for something
that ends in `.csv` and actually exists on disk. If nothing matches, show
the *raw* stdout/stderr in a collapsible debug panel — genuinely useful
when something goes wrong (wrong API key, invalid URL, rate limit) since
you see the ingestion script's real error message instead of a generic
failure.

```python
        st.session_state.csv_path = csv_path
        st.session_state.df = pd.read_csv(csv_path)
        st.session_state.platform_used = platform.lower()
        st.session_state.analyzed_df = None  # force re-analysis
```
Storing the new CSV's path and contents in session state, and — this line
matters — explicitly resetting `analyzed_df` to `None`. Since the earlier
"analyze only once" logic (section 8.6) checks `if
st.session_state.analyzed_df is None`, this forces the *next* render to
actually run/cache-check analysis for the newly fetched data, rather than
continuing to show stale results from whatever was analyzed before.

```python
        metadata_path = csv_path.replace("comments_", "metadata_").replace(".csv", ".json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                st.session_state.post_metadata = meta.get("metadata")
                st.session_state.ai_summary = meta.get("ai_summary")
```
Each ingestion script writes a companion metadata JSON file alongside its
CSV (view count, like count, channel name, etc.) — this derives that
file's expected name via simple string substitution (`comments_X.csv` →
`metadata_X.json`) and loads it if present, for the "Source" info panel
shown at the top of the dashboard.

### 8.6 Loading a previous dataset

```python
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "comments_*.csv")), reverse=True)
if csv_files:
    selected = st.selectbox("📂 Or load a previous dataset", ["—"] + csv_files)
    if selected != "—" and selected != st.session_state.csv_path:
        st.session_state.csv_path = selected
        st.session_state.df = pd.read_csv(selected)
        st.session_state.analyzed_df = None
        ...
```
`glob.glob(...)` finds every previously-saved CSV in `data/`;
`sorted(..., reverse=True)` puts the most recent one first (filenames
embed a timestamp, so alphabetical sort ≈ chronological sort). This is
what lets you revisit an old dataset without re-fetching from the
platform's API — and since analysis results are now cached in Redis
(section 8.4), reopening an old dataset is instant rather than re-running
the model, exactly the feature this conversation's earlier fix was about.

### 8.7 The main analysis trigger

```python
if st.session_state.df is not None:
    df_raw = st.session_state.df.copy()
    df_raw["comment"] = df_raw["comment"].astype(str)
    if "author" not in df_raw.columns:
        df_raw["author"] = "Unknown"

    if st.session_state.analyzed_df is None or st.session_state.analyzed_for != st.session_state.csv_path:
        with st.spinner("Checking cache / running multi-dimensional analysis..."):
            analyzed, was_cached = _load_analyzed_df_cached(
                df_raw, st.session_state.csv_path, st.session_state.platform_used
            )
            st.session_state.analyzed_df = analyzed
            st.session_state.analyzed_for = st.session_state.csv_path
        if was_cached:
            st.toast("⚡ Loaded analysis from Redis cache — skipped re-running the model", icon="⚡")

    df = st.session_state.analyzed_df.copy()
    df["cleaned"] = df["comment"].apply(clean_text)
```
The condition `analyzed_df is None or analyzed_for != csv_path` is the
guard preventing wasted re-analysis: it's only `True` (meaning "go analyze")
the very first time, or when the loaded CSV has actually *changed* since
the last analysis. On every other Streamlit re-run (e.g. the user just
toggled a filter checkbox elsewhere on the page), this whole block is
skipped entirely and the previously-computed `analyzed_df` is reused
straight from session state — instant, no cache check even needed.
`df["cleaned"] = df["comment"].apply(clean_text)` adds one more column
using `preprocess.py`'s lightweight cleaner (section 9.1) — needed for the
spam-similarity/campaign-detection logic just below, which is separate
from (and much simpler than) the main ML analysis.

### 8.8 Campaign/spam-cluster detection (tying `simi.py`, `burst.py`, `score.py` together)

```python
    spam_texts = df[df["is_spam"] == True]["cleaned"].tolist()  # noqa: E712
    similarity_score, spam_clusters = spam_similarity_score(spam_texts)
    burst_flag, burst_series = burst_detect(df.get("published_at", pd.Series([None] * len(df))))
    spam_ratio = (df["is_spam"] == True).mean()  # noqa: E712
    risk_score = campaign_score(similarity_score, burst_flag, spam_ratio)
    campaign_reasons = explain_campaign(similarity_score, burst_flag, spam_ratio)
```
- `df["is_spam"] == True` — the `# noqa: E712` comment silences a linter
  warning that would otherwise suggest `df["is_spam"]` alone (Python style
  guides discourage `== True`), but with pandas boolean *Series* comparisons
  explicitly with `== True` is a deliberate, common, and clearer style
  choice — the `noqa` comment tells the linter "this is intentional, don't
  flag it."
- `spam_similarity_score(spam_texts)` — from `src/simi.py` (explained in
  section 9.3): checks whether many spam comments are near-duplicates of
  each other (a spam *campaign*, not isolated individual spam).
- `burst_detect(...)` — from `src/burst.py` (section 9.4): checks whether
  comments arrived in an unusual, sudden spike.
- `.mean()` on a boolean Series in pandas computes "fraction that are
  True" — a clean one-liner for "what percentage of comments are spam."
- `campaign_score()` / `explain_campaign()` — from `src/score.py`
  (section 9.5): combine all three signals into one overall 0-100 "is a
  coordinated campaign likely happening" score, with human-readable
  reasons.

This is a genuinely different kind of detection from the main
`analyzer.py` pipeline — it's not about any *one* comment being risky,
it's about the *pattern across many comments* (bot networks, brigading,
coordinated spam) that only becomes visible in aggregate.

### 8.9 The five tabs

```python
    tab_overview, tab_deep, tab_alerts, tab_moderation, tab_data = st.tabs(
        ["📊 Overview", "🔬 Deep Analysis", "🚨 Alerts & Actions", "🛡️ Moderation", "🗂️ Raw Data"]
    )
```
`st.tabs([...])` creates clickable tabs and returns one context-manager
object per tab; everything indented under `with tab_x:` renders only when
that tab is active.

**Overview tab** — five KPI cards (`ui_theme.kpi_card`, section 9.2) built
from simple pandas aggregations (`.sum()`, `.mean()`), four bar charts via
Streamlit's built-in `st.bar_chart()` (which just needs a pandas Series or
DataFrame — no manual chart-building code required), and a "top engaged
comments" section using `df.nlargest(6, "like_count")` (pandas' built-in
"give me the N largest by this column" — cleaner than manually sorting
and slicing).

**Deep Analysis tab** — shows which model is active
(`analyzer._MODEL_LOAD_ATTEMPTED` / `USING_TRAINED_MODEL`, straight from
section 3), then three `st.multiselect()` filter widgets (sentiment,
intent, priority), combined via pandas boolean masking:
```python
filtered = df[df["sentiment"].isin(f_sentiment) & df["intent"].isin(f_intent) & df["priority"].isin(f_priority)]
filtered = filtered.sort_values("priority_score", ascending=False)
```
`.isin([...])` checks membership against the selected filter values;
`&` combines the three boolean masks (must use `&`, not Python's `and`,
for element-wise combination across a whole pandas Series). Sorted by
`priority_score` descending, so the riskiest comments always show first.
Capped at `.head(40)` — rendering hundreds of expandable comment cards at
once would make the page sluggish, so only the top 40 (by priority) render
directly, with a note pointing to the Raw Data tab's CSV export for
everything else.

**Alerts & Actions tab** — the UI for `alerts.py` (section 7). Shows a
live count of Jira-worthy candidates (using the *exact same*
`alerts_engine._is_jira_worthy` function as the actual escalation logic,
so this preview number is never out of sync with what clicking the button
will really do), then a button that calls `alerts_engine.process_alerts(df,
...)` and displays each returned alert via `ui_theme.alert_card()`.

**Moderation tab** — displays the campaign/spam-cluster signals from
section 8.8: risk score, warning messages for each triggered reason, an
expandable list per spam cluster, and (if timestamp data exists) a line
chart of comment volume over time via `st.line_chart(burst_series)`.

**Raw Data tab** — a filterable, scrollable `st.dataframe(...)` of every
analysis column, plus `st.download_button(...)` which lets the browser
download the full DataFrame as a CSV — `df.to_csv(index=False)` converts
it to CSV text in-memory (no temp file needed), `index=False` means don't
include pandas' internal row-numbering column in the export.
## 9. The smaller supporting files

### 9.1 `src/preprocess.py`

```python
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()
```
A much *more aggressive* cleaner than `train_intent_model.py`'s
`clean_text()` — this one **deletes** URLs and @mentions entirely (rather
than replacing them with placeholder words) and strips out **every
non-letter character**, including punctuation and numbers. That's
deliberate and specific to what this cleaned text is used for: TF-IDF
similarity comparison (`simi.py`) — a bag-of-words technique that cares
about which *words* appear, not punctuation, emphasis, or exact numbers.
This is exactly why the file's own docstring calls out that the *main* ML
analysis in `analyzer.py` works on the **original**, uncleaned text — punctuation
and capitalization ("AMAZING!!!" vs "amazing") can be meaningful signal for
sentiment, so it would be a mistake to reuse this aggressive cleaner there.

### 9.2 `src/ui_theme.py`

This file has no logic to speak of — it's CSS plus small HTML-generating
helper functions, kept separate purely to stop `dashboard.py` from
becoming a wall of inline styling mixed with business logic.

```python
CSS = """
<style>
:root {
    --bg-0: #0b0f1a;
    ...
}
...
</style>
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
```
`:root { --bg-0: ... }` defines **CSS custom properties** (variables) —
change a color once here, and every rule referencing `var(--bg-0)`
updates. `st.markdown(CSS, unsafe_allow_html=True)` — Streamlit sanitizes
HTML in `st.markdown()` by default (a security precaution against
accidentally rendering user-supplied HTML/JS); `unsafe_allow_html=True`
explicitly opts back in, which is fine here since this string is a fixed
constant in the source code, not derived from any user input.

```python
def kpi_card(label: str, value, sub: str = "", col=None):
    target = col if col is not None else st
    target.markdown(f"""<div class="kpi-card">...</div>""", unsafe_allow_html=True)
```
`target = col if col is not None else st` — a small but neat trick: if
you pass a Streamlit column object (`col`), the card renders *inside* that
column; if you don't, it falls back to rendering directly on the page
(`st`). Both `col.markdown(...)` and `st.markdown(...)` work identically,
so this one function serves both cases without duplicating code.

```python
def priority_pill(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority, "#94a3b8")
    return f'<span class="pill" style="background:{color};">{priority}</span>'
```
Returns a small colored HTML "pill" badge as a string, rather than
rendering it directly — this lets callers embed it *inline* within a
larger markdown string (like the "Deep Analysis" tab's per-comment header,
which combines a priority pill, sentiment pill, author name, and intent
into one line).

### 9.3 `src/simi.py` — spam-similarity clustering

```python
def spam_similarity_score(texts, threshold=0.75):
    if len(texts) < 5:
        return 0.0, []

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, 0)
    ...
```
- **TF-IDF** ("Term Frequency-Inverse Document Frequency") turns each
  comment into a numeric vector where common, uninformative words (like
  "the", filtered out by `stop_words="english"`) count for little, and
  words that are frequent in *this* comment but rare *overall* count for
  a lot. It's a classic, simple technique for comparing text similarity —
  no deep learning needed.
- `cosine_similarity(X)` computes, for every pair of comments, a similarity
  score from 0 (completely different) to 1 (identical topic/wording) based
  on the angle between their TF-IDF vectors.
- `np.fill_diagonal(sim_matrix, 0)` zeroes out each comment's similarity
  to *itself* (which would trivially always be 1.0) so it doesn't
  interfere with the clustering logic below.

```python
    visited = set()
    clusters = []
    for i in range(len(texts)):
        if i in visited:
            continue
        similar = np.where(sim_matrix[i] > threshold)[0].tolist()
        if similar:
            cluster = [i] + similar
            cluster = list(set(cluster))
            visited.update(cluster)
            if len(cluster) > 1:
                clusters.append(cluster)

    similarity_score = sum(len(c) for c in clusters) / len(texts)
    return round(similarity_score, 2), clusters
```
A simple greedy clustering pass: for each not-yet-visited comment, find
every *other* comment more than `threshold` (0.75) similar to it, group
them into one cluster, and mark them all visited so they aren't
re-processed individually. The final `similarity_score` is "what fraction
of all spam comments belong to *some* near-duplicate cluster" — a high
number suggests a coordinated spam campaign (many bots/accounts posting
near-identical text), not just scattered, unrelated spam.

### 9.4 `src/burst.py` — sudden activity spike detection

```python
def burst_detect(timestamps, window="10min", z_thresh=2.5):
    if timestamps.isnull().all():
        return False, None

    ts = pd.to_datetime(timestamps, errors="coerce")
    counts = ts.dt.floor(window).value_counts().sort_index()

    if counts.std() == 0:
        return False, counts

    z = (counts - counts.mean()) / counts.std()
    return (z > z_thresh).any(), counts
```
- `ts.dt.floor(window)` rounds every timestamp *down* to the nearest
  10-minute mark (e.g. `14:37` → `14:30`), then `.value_counts()` counts
  how many comments landed in each 10-minute bucket — effectively building
  a comment-volume-over-time histogram.
- `z = (counts - counts.mean()) / counts.std()` computes a **Z-score** for
  each time bucket — a standard statistics technique measuring "how many
  standard deviations away from the average is this value?" A Z-score of
  0 means exactly average volume; a Z-score of 3 means unusually,
  statistically-significantly high volume for that period.
- `(z > z_thresh).any()` — `True` the moment *any* single time bucket's
  volume is more than 2.5 standard deviations above the mean — a classic,
  simple way to flag "something unusual happened here" without needing any
  ML model, just basic statistics.

### 9.5 `src/score.py` — combining signals into one campaign score

```python
def campaign_score(similarity, burst, spam_ratio):
    score = similarity * 40 + (30 if burst else 0) + spam_ratio * 30
    return round(min(score, 100), 2)

def explain_campaign(similarity, burst, spam_ratio):
    reasons = []
    if similarity > 0.3:
        reasons.append("High repeated / similar comments detected")
    if burst:
        reasons.append("Sudden comment burst detected")
    if spam_ratio > 0.3:
        reasons.append("High spam percentage in comments")
    return reasons
```
The same additive, fully-transparent scoring style used in
`analyzer.py`'s priority score (section 3.10): similarity contributes up
to 40 points, a detected burst is a flat 30, and spam ratio contributes up
to 30 more — a simple weighted formula, easy to read and tune, with a
plain-English explanation function alongside it so the dashboard's
"Moderation" tab can show *why* a risk score is what it is, not just the
number.

### 9.6 `src/jira_client.py`

```python
class JiraClient:
    def __init__(self):
        self.domain = os.getenv("JIRA_DOMAIN", "")
        self.email = os.getenv("JIRA_EMAIL", "")
        self.token = os.getenv("JIRA_API_TOKEN", "")
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "CS")
        self.issue_type = os.getenv("JIRA_ISSUE_TYPE", "Task")
```
Same credentials-from-environment pattern as `redis_client.py`.

```python
    def create_issue(self, summary: str, description: str, priority: str = "High", labels: list | None = None) -> dict:
        ...
        payload = {
            "fields": {
                "project": {"key": self.project_key},
                "summary": summary[:250],
                "description": {
                    "type": "doc", "version": 1,
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": description[:1900]}]}],
                },
                "issuetype": {"name": self.issue_type},
                "labels": labels,
            }
        }
        resp = requests.post(f"{self._base_url()}/rest/api/3/issue", json=payload, auth=(self.email, self.token), ...)
```
`summary[:250]` / `description[:1900]` are defensive truncations — Jira's
API has field length limits, and silently truncating (rather than letting
a very long comment cause an API error and lose the whole alert) is the
safer failure mode here. The nested `{"type": "doc", "version": 1,
"content": [...]}` structure isn't arbitrary — it's Jira Cloud's
**Atlassian Document Format (ADF)**, a required structured format for rich
text fields in API v3 (replacing the plain-string descriptions that older
Jira API versions accepted). `auth=(self.email, self.token)` — Jira Cloud
uses HTTP Basic Auth with your email + an API token (not your actual
password) as the credential pair.

### 9.7 Ingestion scripts — `youtube/`, `instagram/`, `bluesky/fetch_comments.py`

These three scripts share the same overall shape, differing only in which
platform API they call:

1. Parse a URL/ID from the command line (each platform has its own
   `extract_*_id()`-style regex logic — e.g. YouTube's `extract_video_id()`
   handles `youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/`, and
   several other URL formats a person might paste in).
2. Call that platform's official API (using an API key/token read from
   environment variables) to pull comments plus post metadata (view count,
   like count, caption/title).
3. Write two files into `data/`: `comments_<timestamp>.csv` (the actual
   comment text/author/likes/etc.) and a companion
   `metadata_<timestamp>.json` (post-level stats) — this pairing is exactly
   what `dashboard.py`'s metadata-loading logic (section 8.5) expects.
4. **Print the CSV's path to stdout** as the final step — this is the
   simple "protocol" `dashboard.py` relies on (section 8.5) to know where
   the new data landed, since the dashboard runs each script as a
   subprocess and can't directly inspect its return value.

Being separate, standalone scripts (each runnable on its own via `python
youtube/fetch_comments.py <url>`) rather than being imported as a shared
library is a deliberate simplicity choice — each platform's API has
different auth, pagination, and rate-limit quirks, and keeping them fully
independent means a bug or API change in one never risks breaking the
others.

---

## 10. Putting it all together — one comment's full journey

To tie every file above into a single mental picture, here's exactly what
happens to one comment, start to finish:

1. **Ingestion** — `youtube/fetch_comments.py` calls the YouTube Data API,
   gets back a comment's text/author/like-count, and writes it as one row
   in `data/comments_20260808_140000.csv`.
2. **Load** — `dashboard.py` reads that CSV into a pandas DataFrame
   (`st.session_state.df`) when you click "Fetch & Analyze" or pick it
   from the "load a previous dataset" dropdown.
3. **Cache check** — `_load_analyzed_df_cached()` computes a cache key from
   the CSV's fingerprint + `MODEL_VERSION` + `RULES_VERSION`, and asks
   Upstash Redis if this exact combination has been analyzed before. If
   yes → skip straight to step 6 with the cached result.
4. **Analysis (cache miss)** — `analyze_dataframe()` loops over every row
   and calls `analyze_comment()` on this specific comment:
   - `load_trained_models()` ensures the fine-tuned BERT model is loaded
     (only actually does work the very first time).
   - Rule-based lexicons check for spam phrases, severe/mild toxic
     language, and second-person addressing.
   - The BERT model tokenizes the comment, runs it through 12 transformer
     layers, and both heads independently predict sentiment and intent
     with confidence scores.
   - Priority scoring adds up points from every signal (toxicity severity,
     spam, sentiment, intent, emotion, engagement) into one 0-100 score
     and a Low/Medium/High/Critical label.
   - A `reason_summary` string is built listing exactly which signals
     fired.
5. **Cache write** — the whole analyzed DataFrame (including this comment's
   row) is serialized to JSON and written to Redis with a 7-day expiry, so
   the *next* time this same dataset is opened, step 4 is skipped entirely.
6. **Display** — the comment appears in the dashboard's Overview KPIs,
   in the Deep Analysis tab's filterable/sortable card list (with its
   priority pill, sentiment pill, and "Why:" explanation visible), and in
   the Raw Data tab's exportable table.
7. **Escalation (if applicable)** — when you click "Run alert & ticketing
   pass," `alerts.py` filters for comments at/above the priority threshold
   *and* passing the stricter `_is_jira_worthy()` actionability gate. If
   this comment qualifies: check Redis for a dedupe key (has this exact
   comment already been alerted on in the last 24 hours?); if genuinely
   new, optionally get a nicer explanation from Gemini, then call
   `jira_client.create_issue()` to file an actual Jira ticket, and
   increment the "alerts today" counter in Redis.
8. **Audit trail** — regardless of Jira/Redis being configured, the alert
   (or the fact that it was a duplicate) is appended to
   `data/alerts_log.json` as a permanent local record.

---

## 11. Known challenges, and how to improve them

### 11.1 The model's training data quality (covered earlier, repeated here for completeness)
`final_validation_metrics.json` reports 100% accuracy across every class —
a strong sign of overfitting to a small/templated dataset rather than
genuine generalization. Confirmed by spot-checking real, messy comments
against the model directly: confident and correct on training-like
phrasing, noticeably weaker on natural language, sarcasm, or profanity
mixed with otherwise neutral wording. **Fix:** expand `comments_sentiment.csv`
with more real (not templated) examples per class, especially "hard
negatives" (angry-but-not-obviously-so comments, sarcasm, mixed-sentiment
text), and treat the confusion matrices as a checklist of which specific
classes need more/better examples, not just an overall score to chase.

### 11.2 Toxicity used to be flat-weighted (already fixed in this project, explained for context)
Originally, one flat `TOXIC_MARKERS` list meant a single mild insult
("idiot," used about a third party) scored identically to an actual threat.
Fixed by splitting into `TOXIC_MARKERS_SEVERE`/`_MILD` with very different
point contributions, plus a second-person-address heuristic and a
stricter, separate Jira-escalation gate (`_is_jira_worthy()`) that
requires genuine actionability, not just a high priority-score. **Further
improvement:** the "aimed at you vs. third party" detection is still a
keyword proxy, not real grammar/coreference analysis — a small classifier
trained specifically for that distinction (or a well-scoped prompt to the
existing Gemini integration) would be materially more reliable.

### 11.3 Analysis speed on large comment batches
`analyze_dataframe()` calls `analyze_comment()` **once per row, in a plain
Python `for` loop**, with no batching. On CPU, each BERT forward pass takes
roughly 50–150ms, so 1,000 comments takes on the order of 1–2.5 minutes —
all single-threaded. The main levers to fix this, roughly in order of
effort vs. payoff:
- **Batch inference** — the single biggest win. Group comments into
  chunks of ~32, tokenize together with dynamic padding (pad only to the
  *longest comment in that batch*, not always the fixed 128), and run one
  forward pass per batch instead of per comment — matrix math parallelizes
  across a batch almost for free.
- **GPU auto-detection** — `analyzer.py` currently always loads with
  `map_location="cpu"`; adding `device = "cuda" if torch.cuda.is_available()
  else "cpu"` and moving both model and input tensors there would give a
  large free speedup on any machine with a GPU.
- **Dedupe identical comments** before running the model at all (common
  with copy-pasted spam) and map results back to every duplicate.
- **`st.cache_resource`/`st.cache_data`** — Streamlit's own caching
  decorators, layered on top of the Redis caching already added, would
  avoid even the "check Redis" round-trip on every Streamlit re-run within
  the same session.
- **A smaller backbone** (e.g. `distilbert-base-uncased` instead of
  `bert-base-uncased`) trades a small amount of accuracy for roughly 40%
  faster inference — worth it if this needs to run live on CPU regularly.
- **Quantization** (`torch.quantization.quantize_dynamic(model, {nn.Linear},
  dtype=torch.qint8)`) shrinks the linear layers to 8-bit integers for a
  further 2-4x CPU speedup, with minor accuracy loss — a more advanced
  option for later.

### 11.4 Redis cache size limits
Upstash's free tier caps individual request/response sizes (commonly
~1MB). For a few hundred to a couple thousand comments the cached
analysis JSON fits comfortably, but a very large dataset could exceed
that — `set_raw()`/`get_raw()` will simply fail silently in that case
(wrapped in `try/except`), and the app falls back to re-analyzing rather
than crashing. For much larger datasets, gzip-compressing the JSON before
storing (or splitting it across multiple keys) would raise this ceiling.

### 11.5 The rule-based lexicons are English-only and hand-maintained
Every phrase in `SPAM_MARKERS`, `TOXIC_MARKERS_SEVERE/_MILD`,
`EMOTION_LEXICON`, and `INTENT_LEXICON` was written by hand — this catches
what someone thought to include, and nothing more. A comment in Hindi,
Spanish, or heavy internet slang/leetspeak the lexicon authors didn't
anticipate will simply not trigger any rule-based signal (though the ML
model, having learned from real training text, may still catch it via
sentiment/intent if the training data included similar examples). Growing
these lists based on real false-negatives found in production data (or
building a small multilingual toxicity classifier as a second ML signal
alongside the lexicons) is the natural next step.

---

## 12. Quick glossary (for terms used throughout this document)

| Term | Plain-English meaning |
|---|---|
| **Tokenization** | Chopping text into numbered sub-word pieces a model can process |
| **Logits** | A model's raw, un-normalized output scores, before turning them into probabilities |
| **Softmax** | Converts raw scores into probabilities that sum to 1 |
| **Argmax** | "Which option got the highest score" |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A small group of examples processed together, for speed and training stability |
| **Loss** | A single number measuring "how wrong was the model," which training tries to minimize |
| **Gradient** | For each weight, "if I nudge this up, does loss go up or down, and by how much" |
| **Backpropagation** | The calculation that computes all those gradients, using the chain rule |
| **Optimizer (AdamW)** | The algorithm that actually updates weights using the gradients |
| **Learning rate** | How big a step the optimizer takes on each update |
| **Overfitting** | The model memorized training examples instead of learning general patterns |
| **Dropout** | Randomly disabling some neurons during training, to reduce overfitting |
| **Transfer learning** | Starting from a model already trained on other data (like BERT), instead of from scratch |
| **`state_dict`** | A dictionary of a model's learned numbers, by layer name |
| **`.pt` file** | PyTorch's file format for saving a `state_dict` (or other tensors) to disk |
| **TTL (time-to-live)** | How long a cached/stored value is kept before automatically expiring |
| **Idempotent / dedupe** | Doing something twice has the same effect as doing it once — no duplicates |