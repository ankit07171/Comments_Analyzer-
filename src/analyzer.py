"""
analyzer.py
------------
Multi-dimensional comment analysis engine.

For every comment this module answers four questions a moderator/marketer
actually cares about:

  1. WHAT was said        -> sentiment, emotion, spam/toxicity
  2. WHY it matters        -> intent (complaint, support, purchase, etc.)
  3. WHO said it            -> author pass-through from the ingestion layer
  4. HOW urgent it is       -> priority score + explanation (key phrases,
                               confidence, model version) so a human can
                               audit every automated decision.

No heavyweight model downloads are required — sentiment uses VADER's
bundled lexicon and intent/emotion/spam use a curated keyword-rule engine.
If GEMINI_API_KEY is configured, `llm_explain()` can be used on top of this
for a natural-language "why" on the handful of comments that get escalated
(see src/alerts.py), keeping API usage cheap and predictable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_VERSION = "rule-engine-v1.3+vader-3.3"

_vader = SentimentIntensityAnalyzer()

# ------------------------------------------------------------------
# Lexicons — small, explainable, and easy to extend. Each entry maps
# a signal name to the phrases that trigger it, so every classification
# can cite the *exact words* that drove the decision (explainability).
# ------------------------------------------------------------------

EMOTION_LEXICON = {
    "anger": ["angry", "furious", "rage", "pissed", "outrageous", "unacceptable", "disgusting"],
    "frustration": ["frustrated", "annoyed", "fed up", "sick of", "tired of", "again and again", "still broken"],
    "joy": ["love", "amazing", "awesome", "fantastic", "great job", "best", "incredible", "obsessed"],
    "fear": ["worried", "scared", "afraid", "concerned", "unsafe", "dangerous"],
    "sarcasm_risk": ["yeah right", "sure jan", "totally", "oh great", "wow just wow", "as if", "lol ok"],
    "urgency": ["asap", "immediately", "right now", "urgent", "emergency", "before it's too late", "still waiting"],
}

INTENT_LEXICON = {
    "refund_risk": ["refund", "money back", "chargeback", "cancel my order", "cancel my subscription", "want my money"],
    "purchase_intent": ["where can i buy", "how much", "price", "cost", "link please", "where to purchase",
                         "is this available", "shop link", "want to order", "how do i order"],
    "complaint": ["worst", "terrible", "horrible", "hate this", "broken", "not working", "disappointed",
                  "waste of money", "scam", "ripped off", "never again"],
    "support_request": ["how do i", "how to", "can someone help", "need help", "not working", "getting an error",
                         "bug", "issue with", "doesn't work", "trouble with"],
    "feature_request": ["please add", "wish you had", "would be great if", "feature request", "can you add",
                         "suggestion:", "you should add"],
    "misinformation_risk": ["fake news", "this is fake", "hoax", "made up", "not true", "debunked", "clickbait lie"],
    "praise": ["love this", "amazing", "so good", "best video", "underrated", "well done", "keep it up"],
}

SPAM_MARKERS = ["follow me", "check my page", "dm me", "click the link", "free followers",
                "subscribe to my", "earn money fast", "work from home", "giveaway", "promo code", "bit.ly"]

TOXIC_MARKERS = ["idiot", "stupid", "shut up", "kill yourself", "trash", "garbage", "loser", "pathetic"]

GENERIC_SHORT = {"nice", "ok", "okay", "good", "cool", "wow", "great", "lol", "nice video", "first"}


def _find_hits(text: str, lexicon: dict) -> dict:
    """Return {label: [matched phrases]} for every label with a hit."""
    hits = {}
    for label, phrases in lexicon.items():
        matched = [p for p in phrases if p in text]
        if matched:
            hits[label] = matched
    return hits


def _find_flat_hits(text: str, phrases: list) -> list:
    return [p for p in phrases if p in text]


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

    intent: str = "general_discussion"
    intent_confidence: float = 0.0

    priority: str = "Low"
    priority_score: int = 0

    key_phrases: list = field(default_factory=list)
    confidence: float = 0.0
    model_version: str = MODEL_VERSION
    reason_summary: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["emotions"] = ", ".join(self.emotions) if self.emotions else "none"
        d["key_phrases"] = "; ".join(self.key_phrases) if self.key_phrases else "—"
        return d


def analyze_comment(text: str, author: str = "Unknown", platform: str = "unknown",
                     like_count: int = 0) -> CommentAnalysis:
    raw = text or ""
    norm = raw.lower().strip()

    result = CommentAnalysis(comment=raw, author=author or "Unknown", platform=platform)

    # ---------- spam ----------
    spam_hits = _find_flat_hits(norm, SPAM_MARKERS)
    result.is_spam = len(spam_hits) > 0
    result.spam_score = min(1.0, 0.4 * len(spam_hits) + (0.3 if result.is_spam else 0))

    # ---------- toxicity ----------
    toxic_hits = _find_flat_hits(norm, TOXIC_MARKERS)
    result.is_toxic = len(toxic_hits) > 0
    result.toxicity_score = min(1.0, 0.5 * len(toxic_hits))

    # ---------- sentiment (VADER, lexicon-based, no downloads) ----------
    vs = _vader.polarity_scores(raw)
    compound = vs["compound"]
    result.sentiment_score = round(compound, 3)
    if norm in GENERIC_SHORT or len(norm.split()) <= 2 and not spam_hits:
        result.sentiment = "Neutral"
    elif compound >= 0.25:
        result.sentiment = "Positive"
    elif compound <= -0.25:
        result.sentiment = "Negative"
    else:
        result.sentiment = "Neutral"

    # ---------- emotion ----------
    emotion_hits = _find_hits(norm, EMOTION_LEXICON)
    result.emotions = list(emotion_hits.keys())
    if emotion_hits:
        # primary = the emotion with the most matched phrases
        result.primary_emotion = max(emotion_hits, key=lambda k: len(emotion_hits[k]))
    else:
        result.primary_emotion = "none"

    # ---------- intent ----------
    intent_hits = _find_hits(norm, INTENT_LEXICON)
    if result.is_spam:
        result.intent = "spam"
        result.intent_confidence = round(min(0.95, 0.5 + 0.15 * len(spam_hits)), 2)
    elif intent_hits:
        # priority order matters: risk/complaint signals should win over praise
        for label in ["refund_risk", "complaint", "support_request", "misinformation_risk",
                       "purchase_intent", "feature_request", "praise"]:
            if label in intent_hits:
                result.intent = label
                result.intent_confidence = round(min(0.95, 0.45 + 0.15 * len(intent_hits[label])), 2)
                break
    else:
        result.intent = "general_discussion"
        result.intent_confidence = 0.35

    # ---------- priority score ----------
    score = 0
    reasons = []

    if result.is_toxic:
        score += 35
        reasons.append(f"toxic language ({', '.join(toxic_hits)})")
    if result.is_spam:
        score += 15
        reasons.append(f"spam markers ({', '.join(spam_hits)})")
    if result.sentiment == "Negative":
        score += 25
        reasons.append(f"negative sentiment ({result.sentiment_score})")
    if result.intent in ("complaint", "refund_risk"):
        score += 25
        reasons.append(f"{result.intent.replace('_', ' ')} detected")
    if result.intent == "support_request":
        score += 10
        reasons.append("support request detected")
    if "urgency" in result.emotions:
        score += 10
        reasons.append("urgency language")
    if "anger" in result.emotions or "frustration" in result.emotions:
        score += 10
        reasons.append(f"{result.primary_emotion} detected")
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

    # ---------- explainability ----------
    all_hits = list({*spam_hits, *toxic_hits})
    for label_hits in emotion_hits.values():
        all_hits.extend(label_hits)
    for label_hits in intent_hits.values():
        all_hits.extend(label_hits)
    result.key_phrases = list(dict.fromkeys(all_hits))[:6]  # dedupe, cap for readability

    result.confidence = round(min(0.97, 0.5 + 0.08 * len(reasons)), 2)

    if reasons:
        result.reason_summary = (
            f"Flagged {result.priority} priority — " + "; ".join(reasons) + "."
        )
    else:
        result.reason_summary = f"No risk signals found; routine {result.sentiment.lower()} comment."

    return result


def analyze_dataframe(df, text_col="comment", author_col="author", platform="unknown"):
    """Vectorized-ish wrapper: runs analyze_comment over a DataFrame and
    returns a new DataFrame with all analysis columns attached."""
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


def llm_explain(comment_text: str, analysis: CommentAnalysis) -> Optional[str]:
    """Optional: use Gemini 3.6 Flash to produce a natural-language explanation for
    an escalated comment. Only called for alert-worthy comments (see
    src/alerts.py) to keep API usage minimal. Returns None if no API key
    is configured or the call fails — callers should fall back to
    `analysis.reason_summary`.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    import requests

    try:
        # Using Gemini 3.6 Flash (latest and fastest available model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
        prompt = (
            "You are a content-moderation assistant. In 1-2 short sentences, explain in plain "
            "English why the following comment was flagged, referencing what was said and why it "
            "matters for the business. Be concrete, not generic.\n\n"
            f'Comment: "{comment_text}"\n'
            f"Detected sentiment: {analysis.sentiment} ({analysis.sentiment_score})\n"
            f"Detected intent: {analysis.intent}\n"
            f"Priority: {analysis.priority}\n"
            f"Key phrases: {', '.join(analysis.key_phrases) if analysis.key_phrases else 'none'}"
        )
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    return None
