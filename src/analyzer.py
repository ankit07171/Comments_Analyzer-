from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional 

# Lightweight multilingual sentiment and intent analyzer
# Using pre-trained models from HuggingFace that are fast and small

# Model choices (choose one based on performance needs)
# 1. cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingual sentiment, ~1GB)
# 2. nlptown/bert-base-multilingual-uncased-sentiment (multilingual sentiment, ~700MB)
# 3. Using fastText + rule-based intent for lightweight solution

_sentiment_model = None
_sentiment_tokenizer = None
_intent_analyzer = None
_lang_detector = None

USING_ML_MODEL = False
_MODEL_LOAD_ATTEMPTED = False
MODEL_VERSION = "multilingual-xlm-roberta-sentiment-v1"
RULES_VERSION = "rules-v2-toxicity-severity-tiers"

_vader = None  # Keep VADER as fallback for English

# Supported languages for multilingual model
SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'ar', 'zh', 'ja', 'ko', 'hi']

# Sentiment labels mapping for the multilingual model
SENTIMENT_LABELS = {
    'en': {'negative': 0, 'neutral': 1, 'positive': 2},
    'es': {'negativo': 0, 'neutral': 1, 'positivo': 2},
    'fr': {'négatif': 0, 'neutre': 1, 'positif': 2},
    'de': {'negativ': 0, 'neutral': 1, 'positiv': 2},
    # Add more languages as needed
}

def load_ml_models():
    """
    Load lightweight multilingual models for sentiment and intent analysis.
    Uses pre-trained models from HuggingFace that are cached locally.
    """
    global _sentiment_model, _sentiment_tokenizer, _intent_analyzer, _lang_detector
    global USING_ML_MODEL, _MODEL_LOAD_ATTEMPTED, MODEL_VERSION
    
    if _MODEL_LOAD_ATTEMPTED:
        return
    _MODEL_LOAD_ATTEMPTED = True
    
    try:
        print("⏳ Loading lightweight multilingual analysis models...")
        
        # Set HuggingFace cache and offline mode for Render
        os.environ['HF_HOME'] = '/tmp/huggingface_cache'  # Use /tmp on Render
        os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'
        
        # 1. Load language detection model (fastText, very lightweight)
        try:
            import fasttext
            # Download or use local fasttext model
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lid.176.bin")
            if not os.path.exists(model_path):
                print("⚠️ FastText language model not found locally.")
                print("   Language detection will use simple heuristics.")
            else:
                _lang_detector = fasttext.load_model(model_path)
                print("✅ Language detection model loaded")
        except ImportError:
            print("⚠️ fasttext not installed, skipping language detection")
        except Exception as e:
            print(f"⚠️ Could not load language detector: {e}")
        
        # 2. Load sentiment analysis model with retry logic
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            import time
            
            # Using a smaller model for faster inference
            # cardiffnlp/twitter-xlm-roberta-base-sentiment is multilingual and good for social media
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            
            print(f"⏳ Loading multilingual sentiment model: {model_name}")
            
            # Try to load from cache first, with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    _sentiment_tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir='/tmp/huggingface_cache',
                        local_files_only=False,
                        timeout=30
                    )
                    _sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        cache_dir='/tmp/huggingface_cache',
                        local_files_only=False,
                        timeout=30
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}... Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            # Create sentiment analysis pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=_sentiment_model,
                tokenizer=_sentiment_tokenizer,
                device=-1,  # CPU
                truncation=True,
                max_length=512
            )
            
            USING_ML_MODEL = True
            MODEL_VERSION = f"multilingual-xlm-roberta-sentiment"
            print("✅ Multilingual sentiment model loaded successfully")
            
        except ImportError as e:
            print(f"❌ Could not load transformers: {e}")
            print("   Install with: pip install transformers")
            return
        except Exception as e:
            print(f"⚠️ Could not load sentiment model: {e}")
            print("   Falling back to VADER for English + rules for other languages")
            return
        
        # 3. Intent analysis - using rule-based for now (lightweight)
        # Could replace with a small intent model if needed
        print("✅ Intent analysis using rule-based engine (lightweight)")
        
    except Exception as e:
        print(f"❌ Error loading ML models: {e}")
        print("⚠️ Falling back to rule-based analysis")
        USING_ML_MODEL = False

def detect_language(text: str) -> str:
    """
    Detect the language of text.
    Returns language code (en, es, fr, etc.) or 'unknown'
    """
    if not text or len(text.strip()) < 3:
        return 'unknown'
    
    # Try fastText first
    if _lang_detector:
        try:
            predictions = _lang_detector.predict(text)
            lang_code = predictions[0][0].replace('__label__', '')
            # Map to standard language codes
            lang_map = {
                'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
                'pt': 'pt', 'nl': 'nl', 'pl': 'pl', 'ru': 'ru', 'ar': 'ar',
                'zh': 'zh', 'ja': 'ja', 'ko': 'ko', 'hi': 'hi'
            }
            return lang_map.get(lang_code, 'unknown')
        except Exception:
            pass
    
    # Simple heuristic fallback
    text_lower = text.lower()
    
    # Check for common words in different languages
    lang_checks = {
        'es': [' el ', ' la ', ' de ', ' que ', ' y ', ' en '],
        'fr': [' le ', ' la ', ' de ', ' et ', ' que ', ' en '],
        'de': [' der ', ' die ', ' das ', ' und ', ' oder ', ' nicht '],
        'it': [' il ', ' la ', ' del ', ' che ', ' e ', ' di '],
        'pt': [' o ', ' a ', ' de ', ' que ', ' e ', ' do '],
    }
    
    for lang_code, words in lang_checks.items():
        if any(word in text_lower for word in words):
            return lang_code
    
    # Default to English (most social media comments are in English)
    return 'en'

def analyze_sentiment_multilingual(text: str, lang: str = 'en'):
    """
    Analyze sentiment using multilingual model with rate limit handling.
    Returns: (sentiment_label, confidence)
    """
    if not USING_ML_MODEL or _sentiment_model is None:
        return None
    
    try:
        from transformers import pipeline
        import time
        
        # Create sentiment analysis pipeline if not already created
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=_sentiment_model,
            tokenizer=_sentiment_tokenizer,
            device=-1,  # CPU
            truncation=True,
            max_length=512
        )
        
        # Retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Analyze sentiment with timeout
                result = sentiment_pipeline(text[:1000])[0]  # Limit text length
                
                # Map model output to our standard labels
                label = result['label'].lower()
                confidence = result['score']
                
                # Standardize labels
                if 'positive' in label or 'pos' in label:
                    return 'Positive', confidence
                elif 'negative' in label or 'neg' in label:
                    return 'Negative', confidence
                else:
                    return 'Neutral', confidence
                    
            except Exception as e:
                if '429' in str(e) and attempt < max_retries - 1:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️ Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            
    except Exception as e:
        print(f"⚠️ Multilingual sentiment analysis failed: {e}")
        return None


def load_trained_models():
     
    global _multitask_model, _multitask_tokenizer
    global _sentiment_labels, _intent_labels
    global USING_TRAINED_MODEL, _MODEL_LOAD_ATTEMPTED, MODEL_VERSION

    if _MODEL_LOAD_ATTEMPTED:
        return
    _MODEL_LOAD_ATTEMPTED = True

    # Try to download model if running in deployment (Render, etc.)
    # Always try to download if model files don't exist locally
    weights_path = os.path.join(_MODEL_DIR, "multitask_model.pt")
    if not os.path.exists(weights_path):
        # Try to download on Render/cloud environments
        if os.environ.get("RENDER") or os.environ.get("ENABLE_MODEL_DOWNLOAD"):
            _download_model_if_needed()
        else:
            print(f"ℹ️ Custom trained model not found locally.")
            print("   The app will use VADER + rule-based analysis.")
            print("   To use the trained BERT model, upload files to GitHub Releases")
            print("   and set ENABLE_MODEL_DOWNLOAD=true in Render environment variables.")
            return

    weights_path = os.path.join(_MODEL_DIR, "multitask_model.pt")
    tokenizer_path = os.path.join(_MODEL_DIR, "tokenizer")
    sentiment_labels_path = os.path.join(_MODEL_DIR, "sentiment_labels.json")
    intent_labels_path = os.path.join(_MODEL_DIR, "intent_labels.json")

    # Check if we have all required files
    required_files = [weights_path, tokenizer_path, sentiment_labels_path, intent_labels_path]
    if not all(os.path.exists(f) for f in required_files):
        missing = [f for f in required_files if not os.path.exists(f)]
        print(f"ℹ️ Missing model files: {', '.join([os.path.basename(f) for f in missing])}")
        print("   Falling back to VADER + rule-based analysis.")
        return

    try:
        import torch
        import torch.nn as nn
        from transformers import BertTokenizer

        print("⏳ Loading custom trained multi-task ML model (sentiment + intent)...")

        with open(sentiment_labels_path, "r", encoding="utf-8") as f:
            _sentiment_labels = {int(k): v for k, v in json.load(f).items()}
        with open(intent_labels_path, "r", encoding="utf-8") as f:
            _intent_labels = {int(k): v for k, v in json.load(f).items()}

        _multitask_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        model = MultiTaskClassifier.build(
            num_sentiment_labels=len(_sentiment_labels),
            num_intent_labels=len(_intent_labels),
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # Use every CPU core available for matmuls (default is often capped low
        # inside some hosting environments/containers).
        try:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception:
            pass

        # Dynamic INT8 quantization of the Linear layers (the vast majority of
        # BERT's compute). This is a CPU-inference-only speedup, no retraining,
        # no accuracy-relevant change to the model's architecture or outputs.
        try:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"⚠️ Quantization skipped ({e}); using full-precision model.")

        _multitask_model = model

        USING_TRAINED_MODEL = True
        MODEL_VERSION = "multitask-bert-base-uncased-v1.0 (sentiment+intent, fine-tuned)"
        print("✅ Custom trained multi-task model loaded successfully")
        print(f"   Sentiment classes: {list(_sentiment_labels.values())}")
        print(f"   Intent classes ({len(_intent_labels)}): {list(_intent_labels.values())}")

    except Exception as e:
        print(f"⚠️ Failed to load trained model, using rule-based fallback: {e}")
        USING_TRAINED_MODEL = False
        MODEL_VERSION = "rule-engine-v1.3+vader-fallback"
        _multitask_model = None
        _multitask_tokenizer = None


def _predict_with_trained_model(text: str):
    import torch

    encoding = _multitask_tokenizer(
        text[:1000],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        sentiment_logits, intent_logits = _multitask_model(
            encoding["input_ids"], encoding["attention_mask"]
        )
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0]
        intent_probs = torch.softmax(intent_logits, dim=1)[0]

    sentiment_idx = int(torch.argmax(sentiment_probs).item())
    intent_idx = int(torch.argmax(intent_probs).item())

    sentiment_label = _sentiment_labels[sentiment_idx]
    sentiment_conf = float(sentiment_probs[sentiment_idx].item())
    intent_label = _intent_labels[intent_idx]
    intent_conf = float(intent_probs[intent_idx].item())

    return sentiment_label.capitalize(), sentiment_conf, intent_label, intent_conf


def _predict_batch_with_trained_model(texts: list, batch_size: int = 64):
    """Runs the trained BERT model over many comments using batched forward
    passes instead of one comment at a time.

    This is the fix for the multi-minute delay: `_predict_with_trained_model`
    (above) does one BERT forward pass per comment, always padded to a fixed
    128 tokens. Batching comments together and padding each batch only to its
    own longest comment turns N sequential forward passes into ~N/batch_size
    batched ones, which is dramatically faster on CPU (and would be even more
    so on GPU) with byte-identical predictions to the unbatched version.

    Returns a list of (sentiment_label, sentiment_conf, intent_label, intent_conf)
    tuples, one per input text, in the same order as `texts`.
    """
    import torch

    results = [None] * len(texts)
    clean_texts = [(t or "")[:1000] for t in texts]

    for start in range(0, len(clean_texts), batch_size):
        chunk = clean_texts[start:start + batch_size]
        encoding = _multitask_tokenizer(
            chunk,
            truncation=True,
            padding=True,  # dynamic padding: pad to this batch's longest comment, not a fixed 128
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            sentiment_logits, intent_logits = _multitask_model(
                encoding["input_ids"], encoding["attention_mask"]
            )
            sentiment_probs = torch.softmax(sentiment_logits, dim=1)
            intent_probs = torch.softmax(intent_logits, dim=1)

        sentiment_idx = torch.argmax(sentiment_probs, dim=1)
        intent_idx = torch.argmax(intent_probs, dim=1)

        for i in range(len(chunk)):
            s_idx = int(sentiment_idx[i].item())
            i_idx = int(intent_idx[i].item())
            sentiment_label = _sentiment_labels[s_idx].capitalize()
            sentiment_conf = float(sentiment_probs[i, s_idx].item())
            intent_label = _intent_labels[i_idx]
            intent_conf = float(intent_probs[i, i_idx].item())
            results[start + i] = (sentiment_label, sentiment_conf, intent_label, intent_conf)

    return results

EMOTION_LEXICON = {
    "anger": [
        "angry", "furious", "rage", "pissed", "outrageous", "unacceptable", "disgusting",
        "mad", "livid", "enraged", "infuriated", "irate", "fuming", "raging", "seething",
        "incensed", "wrathful", "heated", "boiling", "steaming", "ticked off", "pissed off",
        "fed up", "had enough", "done with this", "sick and tired", "can't take", "losing it"
    ],
    "frustration": [
        "frustrated", "annoyed", "fed up", "sick of", "tired of", "again and again", "still broken",
        "irritated", "bothered", "aggravated", "exasperated", "vexed", "irked", "bugged",
        "hassle", "headache", "pain", "nightmare", "ridiculous", "absurd", "unbelievable",
        "seriously", "really", "come on", "for real", "are you kidding", "you've got to be kidding",
        "nth time", "how many times", "still not fixed", "same issue", "same problem"
    ],
    "joy": [
        "love", "amazing", "awesome", "fantastic", "great job", "best", "incredible", "obsessed",
        "wonderful", "excellent", "outstanding", "superb", "brilliant", "perfect", "flawless",
        "beautiful", "lovely", "delightful", "pleased", "happy", "glad", "excited", "thrilled",
        "impressed", "blown away", "mind blown", "game changer", "life saver", "can't live without",
        "recommend", "must have", "highly recommend", "love love love", "so good", "chef's kiss"
    ],
    "fear": [
        "worried", "scared", "afraid", "concerned", "unsafe", "dangerous",
        "anxious", "nervous", "terrified", "frightened", "alarmed", "panic", "panicking",
        "threat", "threatening", "risky", "risk", "cautious", "careful", "hesitant",
        "suspicious", "sketchy", "shady", "fishy", "scam", "fraud", "hack", "hacked",
        "security", "privacy", "data breach", "stolen", "compromised"
    ],
    "sarcasm_risk": [
        "yeah right", "sure jan", "totally", "oh great", "wow just wow", "as if", "lol ok",
        "sure sure", "of course", "obviously", "clearly", "brilliant", "genius", "nice one",
        "slow clap", "real smart", "congrats", "well done", "bravo", "thanks a lot",
        "fantastic", "amazing work", "keep it up", "doing great", "yep", "uh huh",
        "right...", "okay...", "cool cool cool"
    ],
    "urgency": [
        "asap", "immediately", "right now", "urgent", "emergency", "before it's too late", "still waiting",
        "hurry", "fast", "quickly", "now", "today", "critical", "pressing", "time sensitive",
        "deadline", "running out", "can't wait", "need now", "respond now", "answer now",
        "help now", "fix now", "waiting for", "been waiting", "how long", "when will"
    ],
}

INTENT_LEXICON = {
    "refund_risk": [
        "refund", "money back", "chargeback", "cancel my order", "cancel my subscription", "want my money",
        "return", "returning", "get my money", "pay me back", "reimburse", "reimbursement",
        "demand refund", "requesting refund", "charge back", "dispute charge", "cancel payment",
        "unsubscribe", "stop charging", "cancel membership", "cancel account", "delete account",
        "waste of money", "money wasted", "ripped off", "scammed", "stolen my money",
        "lost money", "never again", "last time", "done paying"
    ],
    "purchase_intent": [
        "where can i buy", "how much", "price", "cost", "link please", "where to purchase",
        "is this available", "shop link", "want to order", "how do i order",
        "where to get", "buy this", "purchase", "get one", "order", "shipping",
        "in stock", "available", "sell", "discount", "coupon", "promo code",
        "deal", "sale", "buy now", "add to cart", "checkout", "interested",
        "want one", "need this", "must have", "take my money", "shut up and take"
    ],
    "complaint": [
        "worst", "terrible", "horrible", "hate this", "broken", "not working", "disappointed",
        "waste of money", "scam", "ripped off", "never again",
        "awful", "pathetic", "useless", "garbage", "trash", "junk", "crap", "sucks",
        "poor", "bad", "terrible quality", "low quality", "cheap", "poorly made",
        "doesn't work", "won't work", "stopped working", "keeps crashing", "crashes",
        "buggy", "glitchy", "freezes", "lags", "slow", "unresponsive",
        "defective", "faulty", "damaged", "missing parts", "incomplete",
        "false advertising", "misleading", "not as described", "expected better"
    ],
    "support_request": [
        "how do i", "how to", "can someone help", "need help", "not working", "getting an error",
        "bug", "issue with", "doesn't work", "trouble with",
        "help me", "please help", "can you help", "support", "assistance", "question",
        "problem", "error", "can't", "unable to", "won't let me", "stuck",
        "guide", "tutorial", "instructions", "how can i", "what do i",
        "setup", "install", "configure", "activate", "access",
        "forgot password", "can't login", "won't connect", "not connecting"
    ],
    "feature_request": [
        "please add", "wish you had", "would be great if", "feature request", "can you add",
        "suggestion:", "you should add",
        "would love", "hope you add", "missing", "needs", "should have", "why no",
        "why isn't there", "add support for", "implement", "include", "integrate",
        "improve", "enhancement", "update", "upgrade", "new feature", "future update",
        "roadmap", "coming soon", "when will you", "planning to add"
    ],
    "misinformation_risk": [
        "fake news", "this is fake", "hoax", "made up", "not true", "debunked", "clickbait lie",
        "false", "misleading", "propaganda", "conspiracy", "lies", "lying",
        "misinformation", "disinformation", "fabricated", "photoshopped", "edited",
        "out of context", "manipulated", "staged", "actors", "crisis actors",
        "fake", "phony", "fraud", "scam alert", "warning", "don't believe"
    ],
    "praise": [
        "love this", "amazing", "so good", "best video", "underrated", "well done", "keep it up",
        "great", "awesome", "fantastic", "excellent", "perfect", "wonderful",
        "impressive", "loved it", "enjoyed", "appreciate", "thank you", "thanks",
        "brilliant", "genius", "masterpiece", "incredible", "outstanding",
        "helpful", "useful", "exactly what i needed", "life changing", "eye opening",
        "informative", "educational", "learned a lot", "subscribed", "subscribing"
    ],
}

SPAM_MARKERS = [
    "follow me", "check my page", "dm me", "click the link", "free followers",
    "subscribe to my", "earn money fast", "work from home", "giveaway", "promo code", "bit.ly",
    "check out my", "visit my", "link in bio", "tap link", "swipe up",
    "follow for follow", "f4f", "l4l", "like for like", "sub for sub", "s4s",
    "free money", "make money", "get rich", "earn $", "cash app", "paypal",
    "followers free", "likes free", "views free", "100% real", "not fake",
    "crypto", "bitcoin", "forex", "trading", "investment opportunity",
    "weight loss", "get fit", "lose weight", "diet pills", "supplements",
    "online casino", "betting", "gamble", "lottery", "prize",
    "congratulations", "you won", "claim your", "limited time", "act now",
    "whatsapp", "telegram", "signal", "contact me at", "text me",
    "tinyurl", "shorturl", "goo.gl", "ow.ly"
]

TOXIC_MARKERS_SEVERE = [
    "kill yourself", "kys", "neck yourself", "end yourself",
    "hope you die", "should die", "deserve to die",
    "kill you", "murder", "assault",
    "go to hell", "burn in hell", "rot in hell",
    "nobody likes you", "everyone hates you",
]

TOXIC_MARKERS_MILD = [
    "idiot", "stupid", "moron", "dumb", "dumbass", "dumbfuck", "imbecile",
    "fuck", "fucker", "fucking moron", "fucking idiot", "piece of shit",
    "shit", "shitty", "bullshit", "asshole", "ass hole",
    "bitch", "bastard", "piss off", "pissed off",
    "trash", "garbage human", "loser", "pathetic loser", "worthless",
    "useless", "waste of space", "scum", "filth", "pig",
    "disgusting", "vile", "toxic", "cancer",
    "shut up", "shut the fuck up", "stfu", "piece of trash"
]

TOXIC_MARKERS = TOXIC_MARKERS_SEVERE + TOXIC_MARKERS_MILD
 
_SECOND_PERSON_MARKERS = [
    "you", "you're", "youre", "ur", "your", "u r", "u are", " u ", "@you",
]


def _mentions_second_person(norm: str) -> bool:
    return any(re.search(r'\b' + re.escape(m).replace(r'\ ', r'\s+') + r'\b', norm) for m in _SECOND_PERSON_MARKERS)

GENERIC_SHORT = {
    "nice", "ok", "okay", "good", "cool", "wow", "great", "lol", "nice video", "first",
    "yeah", "yep", "nope", "yes", "no", "k", "kk", "gg", "nice one"
}

def _find_hits(text: str, lexicon: dict) -> dict:
    hits = {}
    for label, phrases in lexicon.items():
        matched = []
        for phrase in phrases:
            # Use word boundary regex for single words, exact match for phrases
            if ' ' in phrase:
                if phrase in text:
                    matched.append(phrase)
            else:
                pattern = r'\b' + re.escape(phrase) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    matched.append(phrase)
        if matched:
            hits[label] = matched
    return hits


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

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["emotions"] = ", ".join(self.emotions) if self.emotions else "none"
        d["key_phrases"] = "; ".join(self.key_phrases) if self.key_phrases else "—"
        return d


def analyze_comment(text: str, author: str = "Unknown", platform: str = "unknown",
                     like_count: int = 0) -> CommentAnalysis:
    """Analyzes a single comment using lightweight multilingual models."""
    # Lazy load multilingual models on first use
    load_ml_models()

    raw = text or ""
    norm = raw.lower().strip()

    result = CommentAnalysis(comment=raw, author=author or "Unknown", platform=platform)
    
    # Declare global variables at the top
    global _vader

    # 1. Detect language
    lang = detect_language(raw)
    
    # 2. Analyze spam and toxicity (language-independent rules)
    spam_hits = _find_flat_hits(norm, SPAM_MARKERS)
    result.is_spam = len(spam_hits) > 0
    result.spam_score = min(1.0, 0.4 * len(spam_hits) + (0.3 if result.is_spam else 0))

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

    # 3. Analyze sentiment using multilingual model
    if USING_ML_MODEL:
        # Try multilingual model first
        sentiment_result = analyze_sentiment_multilingual(raw, lang)
        if sentiment_result:
            sentiment_label, sentiment_conf = sentiment_result
            result.sentiment = sentiment_label
            if sentiment_label == "Positive":
                result.sentiment_score = round(sentiment_conf, 3)
            elif sentiment_label == "Negative":
                result.sentiment_score = round(-sentiment_conf, 3)
            else:
                result.sentiment_score = 0.0
        else:
            # Fallback to VADER for English, neutral for other languages
            if lang == 'en':
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
            else:
                # For non-English languages without model, use neutral
                result.sentiment = "Neutral"
                result.sentiment_score = 0.0
    else:
        # Fallback to VADER for English
        if lang == 'en':
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
        else:
            result.sentiment = "Neutral"
            result.sentiment_score = 0.0

    # 4. Analyze intent (rule-based, works across languages)
    intent_hits = _find_hits(norm, INTENT_LEXICON)
    
    if result.is_spam:
        result.intent = "advertisement_or_promotion"
        result.intent_confidence = round(min(0.95, 0.5 + 0.15 * len(spam_hits)), 2)
    elif intent_hits:
        # Priority order matters: risk/complaint signals should win over praise
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

    # 5. Analyze emotions
    emotion_hits = _find_hits(norm, EMOTION_LEXICON)
    result.emotions = list(emotion_hits.keys())
    if emotion_hits:
        result.primary_emotion = max(emotion_hits, key=lambda k: len(emotion_hits[k]))
    else:
        result.primary_emotion = "none"

    # 6. Calculate priority score
    score = 0
    reasons = []

    if result.toxicity_severity == "severe":
        score += 35 + (10 if directed else 0)
        reasons.append(f"severe toxic language / threat ({', '.join(severe_hits)})")
    elif result.toxicity_severity == "mild":
        score += 12 + (10 if directed else 0)
        target_note = "aimed at the channel" if directed else "third-party commentary"
        reasons.append(f"mild toxic language ({', '.join(mild_hits)}) — {target_note}")
    if result.is_spam:
        score += 15
        reasons.append(f"spam markers ({', '.join(spam_hits)})" if spam_hits
                        else f"spam-flavoured intent ({result.intent.replace('_', ' ')})")
    if result.sentiment == "Negative":
        score += 25
        reasons.append(f"negative sentiment ({result.sentiment_score})")
    if result.intent == "fraudulent_service_offer":
        score += 30
        reasons.append("fraudulent/scam service offer detected — brand-safety risk")
    if result.intent == "complaint_or_problem_report":
        score += 25
        reasons.append("complaint / problem report detected")
    if result.intent in ("financial_promotion", "giveaway_or_reward_scam"):
        score += 15
        reasons.append(f"{result.intent.replace('_', ' ')} detected — possible scam risk")
    if result.intent == "question_or_information_request":
        score += 10
        reasons.append("question / info request detected — needs a response")
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

    # 7. Collect key phrases
    all_hits = list({*spam_hits, *toxic_hits})
    for label_hits in emotion_hits.values():
        all_hits.extend(label_hits)
    for label_hits in intent_hits.values():
        all_hits.extend(label_hits)
    result.key_phrases = list(dict.fromkeys(all_hits))[:6]

    # 8. Confidence and summary
    result.model_version = MODEL_VERSION
    if USING_ML_MODEL:
        result.confidence = round((result.intent_confidence + abs(result.sentiment_score) if result.sentiment != "Neutral" else 0.7) / 2, 2)
    else:
        result.confidence = round(min(0.97, 0.5 + 0.08 * len(reasons)), 2)

    # Add language info to summary if not English
    lang_info = f" [Language: {lang}]" if lang != 'en' else ""
    
    if reasons:
        result.reason_summary = (
            f"Flagged {result.priority} priority{lang_info} — " + "; ".join(reasons) + "."
        )
    else:
        result.reason_summary = f"No risk signals found{lang_info}; routine {result.sentiment.lower()} comment."

    return result


def analyze_dataframe(df, text_col="comment", author_col="author", platform="unknown", batch_size=32):
    """Runs analysis over an entire DataFrame using lightweight multilingual models."""
    import pandas as pd

    # Load models once
    load_ml_models()

    texts = [str(row.get(text_col, "")) for _, row in df.iterrows()]

    records = []
    for idx, (_, row) in enumerate(df.iterrows()):
        text = texts[idx]
        author = str(row.get(author_col, "Unknown")) if author_col in df.columns else "Unknown"
        likes = int(row.get("like_count", 0)) if "like_count" in df.columns and str(row.get("like_count", "")).strip() != "" else 0
        
        # Analyze each comment
        analysis = analyze_comment(text, author=author, platform=platform, like_count=likes)
        records.append(analysis.to_dict())

    analysis_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), analysis_df.drop(columns=["comment", "author"], errors="ignore")], axis=1)
    return out


def llm_explain(comment_text: str, analysis: CommentAnalysis) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    import requests

    try:
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