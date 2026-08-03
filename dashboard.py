import streamlit as st
import pandas as pd
import subprocess
import os
import sys
import glob
import json
import warnings

# Suppress Windows asyncio warnings (harmless but noisy)
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*proactor.*')

from dotenv import load_dotenv
load_dotenv()

# ---------------- PATH SETUP ----------------
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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Comment Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
)
ui_theme.inject_css()

# ---------------- SESSION STATE ----------------
for key, default in [
    ("csv_path", None), ("df", None), ("post_metadata", None), ("ai_summary", None),
    ("analyzed_df", None), ("analyzed_for", None), ("last_alerts", None), ("platform_used", "unknown"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

redis_client = get_redis_client()
jira_client = get_jira_client()

PLATFORM_CONFIG = {
    "YouTube": {"script": os.path.join(BASE_DIR, "youtube", "fetch_comments.py"), "placeholder": "YouTube video URL or ID"},
    "Instagram": {"script": os.path.join(BASE_DIR, "instagram", "fetch_comments.py"), "placeholder": "Instagram reel/post URL"},
    "Bluesky": {"script": os.path.join(BASE_DIR, "bluesky", "fetch_comments.py"), "placeholder": "bsky.app post URL"},
}

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 🛰️ Comment Intelligence")
    st.caption("Multi-platform ingestion → deep analysis → automated routing")
    st.markdown("---")

    platform = st.selectbox("Platform", ["Select", "YouTube", "Instagram", "Bluesky"])
    url = st.text_input(
        "Post URL / ID",
        placeholder=PLATFORM_CONFIG.get(platform, {}).get("placeholder", "Enter a URL"),
    )
    fetch_clicked = st.button("🚀 Fetch & Analyze", use_container_width=True)

    st.markdown("---")
    st.markdown("**Integrations**")
    st.markdown(
        f"- Jira ticketing: {'🟢 connected' if jira_client.is_configured() else '⚪ not configured'}\n"
        f"- Upstash Redis: {'🟢 connected' if redis_client.is_configured() else '⚪ not configured'}\n"
        f"- Gemini explanations: {'🟢 enabled' if os.getenv('GEMINI_API_KEY') else '⚪ disabled'}"
    )
    if not jira_client.is_configured() or not redis_client.is_configured():
        st.caption("Add credentials to `.env` to enable auto-ticketing & alert de-dupe. See `.env.example`.")

    today_count = alerts_engine.get_today_alert_count()
    st.markdown("---")
    st.metric("🚨 Negative alerts today", today_count if today_count is not None else "—")

# ---------------- FETCH ----------------
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
        else:  # Bluesky
            cmd = [sys.executable, script, "--url", url]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR, env=env)

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

        st.session_state.csv_path = csv_path
        st.session_state.df = pd.read_csv(csv_path)
        st.session_state.platform_used = platform.lower()
        st.session_state.analyzed_df = None  # force re-analysis

        metadata_path = csv_path.replace("comments_", "metadata_").replace(".csv", ".json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                st.session_state.post_metadata = meta.get("metadata")
                st.session_state.ai_summary = meta.get("ai_summary")

        st.success(f"Loaded {len(st.session_state.df)} comments from {platform}")

# ---------------- LOAD PREVIOUS DATASET ----------------
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "comments_*.csv")), reverse=True)
if csv_files:
    selected = st.selectbox("📂 Or load a previous dataset", ["—"] + csv_files)
    if selected != "—" and selected != st.session_state.csv_path:
        st.session_state.csv_path = selected
        st.session_state.df = pd.read_csv(selected)
        st.session_state.analyzed_df = None
        metadata_path = selected.replace("comments_", "metadata_").replace(".csv", ".json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                st.session_state.post_metadata = meta.get("metadata")
                st.session_state.ai_summary = meta.get("ai_summary")

# ---------------- HERO ----------------
ui_theme.hero(
    "Comment Intelligence Platform",
    "Ingest comments from YouTube, Instagram &amp; Bluesky, classify sentiment, emotion, intent and "
    "priority with a fully explainable engine, and auto-route the risky ones to Jira in real time.",
    {
        "YouTube": True, "Instagram": True, "Bluesky": True,
        "Jira": jira_client.is_configured(), "Upstash": redis_client.is_configured(),
    },
)

# ---------------- POST / VIDEO INFO ----------------
if st.session_state.post_metadata:
    meta = st.session_state.post_metadata
    is_youtube = "video_id" in meta
    is_instagram = "reel_id" in meta
    is_bluesky = "post_uri" in meta

    st.markdown('<div class="section-title">📌 Source</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    if is_youtube:
        ui_theme.kpi_card("Views", f"{meta.get('view_count', 0):,}", col=c1)
        ui_theme.kpi_card("Likes", f"{meta.get('like_count', 0):,}", col=c2)
        ui_theme.kpi_card("Comments", f"{meta.get('comment_count', 0):,}", col=c3)
        ui_theme.kpi_card("Channel", meta.get("channel_title", "—"), col=c4)
        if meta.get("title"):
            st.caption(f"🎬 {meta['title']}")
    elif is_instagram or is_bluesky:
        ui_theme.kpi_card("Likes", f"{meta.get('like_count', 0):,}", col=c1)
        ui_theme.kpi_card("Comments", f"{meta.get('comment_count', 0):,}", col=c2)
        ui_theme.kpi_card("Views" if is_instagram else "Reposts",
                           f"{meta.get('view_count' if is_instagram else 'repost_count', 0):,}", col=c3)
        ui_theme.kpi_card("Author", f"@{meta.get('username', 'Unknown')}", col=c4)
        if meta.get("caption"):
            with st.expander("📝 Caption / post text"):
                st.write(meta["caption"])

    if st.session_state.ai_summary:
        with st.expander("🤖 AI-generated summary"):
            st.write(st.session_state.ai_summary)

# ---------------- ANALYSIS ----------------
if st.session_state.df is not None:
    df_raw = st.session_state.df.copy()
    df_raw["comment"] = df_raw["comment"].astype(str)
    if "author" not in df_raw.columns:
        df_raw["author"] = "Unknown"

    # Analyze only once per dataset (cached in session state)
    if st.session_state.analyzed_df is None or st.session_state.analyzed_for != st.session_state.csv_path:
        with st.spinner("Running multi-dimensional analysis (sentiment, emotion, intent, priority)..."):
            analyzed = analyze_dataframe(df_raw, platform=st.session_state.platform_used)
            st.session_state.analyzed_df = analyzed
            st.session_state.analyzed_for = st.session_state.csv_path

    df = st.session_state.analyzed_df.copy()
    df["cleaned"] = df["comment"].apply(clean_text)

    # ---- campaign / spam-cluster signals (kept from the original moderation engine) ----
    spam_texts = df[df["is_spam"] == True]["cleaned"].tolist()  # noqa: E712
    similarity_score, spam_clusters = spam_similarity_score(spam_texts)
    burst_flag, burst_series = burst_detect(df.get("published_at", pd.Series([None] * len(df))))
    spam_ratio = (df["is_spam"] == True).mean()  # noqa: E712
    risk_score = campaign_score(similarity_score, burst_flag, spam_ratio)
    campaign_reasons = explain_campaign(similarity_score, burst_flag, spam_ratio)

    tab_overview, tab_deep, tab_alerts, tab_moderation, tab_data = st.tabs(
        ["📊 Overview", "🔬 Deep Analysis", "🚨 Alerts & Actions", "🛡️ Moderation", "🗂️ Raw Data"]
    )

    # ================= OVERVIEW =================
    with tab_overview:
        st.markdown('<div class="section-title">Key metrics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        ui_theme.kpi_card("Total comments", len(df), col=c1)
        ui_theme.kpi_card("😊 Positive", int((df["sentiment"] == "Positive").sum()),
                           f"{(df['sentiment'] == 'Positive').mean():.0%}", col=c2)
        ui_theme.kpi_card("😠 Negative", int((df["sentiment"] == "Negative").sum()),
                           f"{(df['sentiment'] == 'Negative').mean():.0%}", col=c3)
        ui_theme.kpi_card("🚫 Spam blocked", int((df["is_spam"] == True).sum()), col=c4)  # noqa: E712
        ui_theme.kpi_card("🔥 Critical priority", int((df["priority"] == "Critical").sum()), col=c5)

        st.markdown("<br/>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">Sentiment distribution</div>', unsafe_allow_html=True)
            st.bar_chart(df["sentiment"].value_counts())
        with col_b:
            st.markdown('<div class="section-title">Intent breakdown</div>', unsafe_allow_html=True)
            st.bar_chart(df["intent"].value_counts())

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="section-title">Priority mix</div>', unsafe_allow_html=True)
            order = ["Low", "Medium", "High", "Critical"]
            counts = df["priority"].value_counts().reindex(order).fillna(0)
            st.bar_chart(counts)
        with col_d:
            st.markdown('<div class="section-title">Top voices (by comment count)</div>', unsafe_allow_html=True)
            top_authors = df["author"].value_counts().head(8)
            st.bar_chart(top_authors)

        if "like_count" in df.columns:
            st.markdown('<div class="section-title">🔥 Top engaged comments</div>', unsafe_allow_html=True)
            top_comments = df.nlargest(6, "like_count")
            for _, row in top_comments.iterrows():
                with st.expander(f"👍 {row['like_count']} · @{row.get('author', 'Unknown')} · {row['sentiment']} / {row['intent']}"):
                    st.write(row["comment"])
                    st.caption(row["reason_summary"])

    # ================= DEEP ANALYSIS =================
    with tab_deep:
        st.markdown(
            '<div class="section-title">Per-comment explainability — what, why, and by whom</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model: `{df['model_version'].iloc[0] if len(df) else 'n/a'}` — every classification below cites the exact phrases that drove it.")

        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            f_sentiment = st.multiselect("Sentiment", sorted(df["sentiment"].unique()), default=list(df["sentiment"].unique()))
        with colf2:
            f_intent = st.multiselect("Intent", sorted(df["intent"].unique()), default=list(df["intent"].unique()))
        with colf3:
            f_priority = st.multiselect("Priority", ["Critical", "High", "Medium", "Low"],
                                         default=["Critical", "High", "Medium", "Low"])

        filtered = df[df["sentiment"].isin(f_sentiment) & df["intent"].isin(f_intent) & df["priority"].isin(f_priority)]
        filtered = filtered.sort_values("priority_score", ascending=False)

        st.caption(f"{len(filtered)} of {len(df)} comments match filters")

        for _, row in filtered.head(40).iterrows():
            header = (
                f"{ui_theme.priority_pill(row['priority'])} {ui_theme.sentiment_pill(row['sentiment'])} "
                f"&nbsp; <b>@{row['author']}</b> &nbsp;·&nbsp; intent: <code>{row['intent']}</code> "
                f"&nbsp;·&nbsp; confidence {row['confidence']:.0%}"
            )
            st.markdown(header, unsafe_allow_html=True)
            st.markdown(f"“{row['comment']}”")
            st.caption(f"Why: {row['reason_summary']}  ·  Key phrases: {row['key_phrases']}  ·  Emotion: {row['emotions']}")
            st.markdown("---")

        if len(filtered) > 40:
            st.info(f"Showing top 40 of {len(filtered)} by priority score. Refine filters or use the Raw Data tab to export everything.")

    # ================= ALERTS & ACTIONS =================
    with tab_alerts:
        st.markdown('<div class="section-title">Automated routing</div>', unsafe_allow_html=True)
        st.caption(
            "Negative or toxic comments at/above the configured priority threshold are de-duplicated "
            "via Upstash Redis and filed as Jira tickets automatically. Re-running analysis on the same "
            "dataset will not create duplicate tickets."
        )

        threshold = os.getenv("ALERT_PRIORITY_THRESHOLD", "High")
        n_candidates = int(((df["sentiment"] == "Negative") | (df["is_toxic"] == True)).sum())  # noqa: E712
        st.write(f"Threshold: **{threshold}+** priority · Candidates in this dataset: **{n_candidates}**")

        run_alerts = st.button("⚡ Run alert & ticketing pass")
        if run_alerts:
            with st.spinner("Evaluating alerts, checking dedupe cache, filing Jira tickets..."):
                st.session_state.last_alerts = alerts_engine.process_alerts(
                    df, platform=st.session_state.platform_used
                )

        if st.session_state.last_alerts:
            new_alerts = [a for a in st.session_state.last_alerts if a["is_new"]]
            dup_alerts = [a for a in st.session_state.last_alerts if not a["is_new"]]
            st.success(f"{len(new_alerts)} new alert(s) raised · {len(dup_alerts)} already seen (deduped)")
            for a in st.session_state.last_alerts:
                ui_theme.alert_card(a)
        else:
            st.info("No alert pass has been run yet for this dataset.")

        st.markdown('<div class="section-title">Recent alert history</div>', unsafe_allow_html=True)
        history = alerts_engine.get_recent_alerts(limit=15)
        if history:
            for a in history:
                ui_theme.alert_card(a)
        else:
            st.caption("No alerts logged yet.")

    # ================= MODERATION =================
    with tab_moderation:
        st.markdown('<div class="section-title">Coordinated campaign / spam signals</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        ui_theme.kpi_card("Spam ratio", f"{spam_ratio:.1%}", col=c1)
        ui_theme.kpi_card("Similarity score", similarity_score, col=c2)
        ui_theme.kpi_card("Burst detected", "Yes" if burst_flag else "No", col=c3)

        st.markdown('<div class="section-title">Campaign risk score</div>', unsafe_allow_html=True)
        st.metric("Risk score", f"{risk_score} / 100")
        if campaign_reasons:
            for r in campaign_reasons:
                st.warning(r)
        else:
            st.success("No coordinated campaign detected")

        st.markdown('<div class="section-title">Repeated spam clusters</div>', unsafe_allow_html=True)
        if spam_clusters:
            for i, cluster in enumerate(spam_clusters, 1):
                with st.expander(f"Cluster {i} · {len(cluster)} comments"):
                    for idx in cluster[:5]:
                        st.write("•", spam_texts[idx])
        else:
            st.success("No repeated spam patterns found")

        if burst_series is not None and not burst_series.empty and len(burst_series) > 1:
            st.markdown('<div class="section-title">Comment activity over time</div>', unsafe_allow_html=True)
            st.line_chart(burst_series)

    # ================= RAW DATA =================
    with tab_data:
        st.markdown('<div class="section-title">Filter & export</div>', unsafe_allow_html=True)
        display_cols = ["comment", "author", "sentiment", "sentiment_score", "emotions", "intent",
                         "intent_confidence", "priority", "priority_score", "is_spam", "is_toxic",
                         "key_phrases", "confidence", "reason_summary"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], width='stretch', height=420)

        st.download_button(
            "📥 Download full analysis (CSV)",
            df.to_csv(index=False),
            "analyzed_comments.csv",
            "text/csv",
        )

else:
    st.info("👈 Select a platform and fetch comments to start analysis, or load a previous dataset above.")
