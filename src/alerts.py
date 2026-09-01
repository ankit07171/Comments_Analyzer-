"""
alerts.py
----------
Turns analyzed comments into action:

  1. Filters comments at/above the configured priority threshold.
  2. De-duplicates against Upstash Redis so the same comment doesn't
     trigger a duplicate Jira ticket on every re-run of the dashboard.
  3. Files a Jira ticket for genuinely new, high-risk comments.
  4. Tracks a rolling "negative comments today" counter in Redis so the
     dashboard can show a live risk pulse across sessions.
  5. Persists a local alert log (data/alerts_log.json) as a fallback /
     audit trail even when Jira or Redis aren't configured.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from . import jira_client, redis_client
from .analyzer import llm_explain

PRIORITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

# Intents that represent something a human/engineer/support agent can
# actually *act on* — a real bug report, a scam attempt, a refund risk.
# General political/off-topic name-calling, praise, ads, or engagement-bait
# are not actionable even if they score High on the dashboard's priority
# scale (which also weighs things like reach/urgency language that matter
# for triage, but not for "should this open a Jira ticket").
ACTIONABLE_INTENTS = {
    "complaint_or_problem_report",
    "fraudulent_service_offer",
    "financial_promotion",
    "giveaway_or_reward_scam",
}


def _is_jira_worthy(row) -> bool:
    """Stricter gate than the dashboard's priority score. A comment only
    escalates to Jira if there's something for a human to actually act on:
      - severe toxicity (real threats / targeted hate speech) — always,
        regardless of who it's aimed at or what else is in the comment
      - one of the actionable ML intents (bug reports, scam attempts, etc.)
      - negative user-experience feedback specifically
    Mild name-calling ("idiot" about a public figure, generic political
    banter) alone no longer qualifies — it stays visible in the dashboard
    for a moderator to skim, but doesn't spam the Jira backlog.
    """
    if row.get("toxicity_severity") == "severe":
        return True
    intent = row.get("intent", "")
    if intent in ACTIONABLE_INTENTS:
        return True
    if intent == "user_experience_feedback" and row.get("sentiment") == "Negative":
        return True
    return False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ALERT_LOG_PATH = os.path.join(DATA_DIR, "alerts_log.json")


def _comment_hash(author: str, comment: str) -> str:
    raw = f"{author.strip().lower()}::{comment.strip().lower()}"
    return "alert:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


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


def process_alerts(analyzed_df, platform: str, use_llm: bool = True,
                    max_llm_calls: int = 5) -> list:
    """
    analyzed_df: DataFrame that already has analyzer output columns
                 (priority, sentiment, intent, reason_summary, key_phrases, ...)
    Returns a list of alert dicts (also persisted to data/alerts_log.json).
    """
    threshold = os.getenv("ALERT_PRIORITY_THRESHOLD", "High")
    ttl_hours = int(os.getenv("ALERT_DEDUPE_TTL_HOURS", "24"))
    threshold_rank = PRIORITY_ORDER.get(threshold, 2)

    rc = redis_client.get_client()
    jc = jira_client.get_client()

    candidates = analyzed_df[analyzed_df["priority"].map(lambda p: PRIORITY_ORDER.get(p, 0) >= threshold_rank)]
    # Priority threshold narrows it down to "worth a human's attention."
    # _is_jira_worthy() then narrows further to "worth an actual ticket" —
    # severe toxicity or a genuinely actionable ML intent, not just any
    # negative/toxic-flavoured comment (see docstring above).
    candidates = candidates[candidates.apply(_is_jira_worthy, axis=1)]

    alerts = []
    llm_calls_used = 0
    log = _load_local_log()

    for _, row in candidates.iterrows():
        author = str(row.get("author", "Unknown"))
        comment = str(row.get("comment", ""))
        key = _comment_hash(author, comment)

        is_new = rc.setnx_with_ttl(key, "1", ex_seconds=ttl_hours * 3600)

        explanation = row.get("reason_summary", "")
        if use_llm and is_new and llm_calls_used < max_llm_calls:
            from .analyzer import CommentAnalysis
            pseudo = CommentAnalysis(
                comment=comment, author=author, platform=platform,
                sentiment=row.get("sentiment", "Negative"),
                sentiment_score=row.get("sentiment_score", 0.0),
                intent=row.get("intent", "complaint"),
                priority=row.get("priority", "High"),
                key_phrases=str(row.get("key_phrases", "")).split("; ") if row.get("key_phrases") else [],
            )
            llm_text = llm_explain(comment, pseudo)
            if llm_text:
                explanation = llm_text
                llm_calls_used += 1

        jira_result = {"ok": False, "key": None, "url": None, "error": "skipped (duplicate or dry-run)"}
        if is_new:
            summary = f"[{platform.title()}] {row.get('priority', 'High')} priority comment from @{author}"
            description = (
                f"Platform: {platform}\n"
                f"Author: {author}\n"
                f"Sentiment: {row.get('sentiment')} ({row.get('sentiment_score')})\n"
                f"Intent: {row.get('intent')}\n"
                f"Priority score: {row.get('priority_score')}/100\n\n"
                f'Comment: "{comment}"\n\n'
                f"Why flagged: {explanation}"
            )
            jira_result = jc.create_issue(
                summary=summary,
                description=description,
                priority=row.get("priority", "High"),
                labels=["comment-intelligence", platform, str(row.get("intent", "complaint"))],
            )
            rc.incr(f"stats:negative_alerts:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}")

        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform,
            "author": author,
            "comment": comment,
            "sentiment": row.get("sentiment"),
            "intent": row.get("intent"),
            "priority": row.get("priority"),
            "priority_score": int(row.get("priority_score", 0)),
            "explanation": explanation,
            "is_new": is_new,
            "jira_ok": jira_result["ok"],
            "jira_key": jira_result["key"],
            "jira_url": jira_result["url"],
            "jira_error": jira_result["error"],
        }
        alerts.append(alert)
        if is_new:
            log.append(alert)

    _save_local_log(log)
    return alerts


def get_today_alert_count() -> int | None:
    rc = redis_client.get_client()
    val = rc.get(f"stats:negative_alerts:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def get_recent_alerts(limit: int = 50) -> list:
    log = _load_local_log()
    return list(reversed(log))[:limit]