import streamlit as st

PRIORITY_COLORS = {
    "Critical": "#ef4444",
    "High": "#f97316",
    "Medium": "#eab308",
    "Low": "#22c55e",
}

SENTIMENT_COLORS = {
    "Positive": "#22c55e",
    "Neutral": "#94a3b8",
    "Negative": "#ef4444",
}

CSS = """
<style>
:root {
    --bg-0: #0b0f1a;
    --bg-1: #10162a;
    --card: #141b30;
    --card-border: #232c47;
    --accent: #7c6cf6;
    --accent-2: #22d3ee;
    --text-hi: #eef1fb;
    --text-lo: #9aa4c2;
}

.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #1a2140 0%, var(--bg-0) 55%) fixed;
}

section[data-testid="stSidebar"] {
    background: var(--bg-1);
    border-right: 1px solid var(--card-border);
}

h1, h2, h3, h4 { color: var(--text-hi) !important; letter-spacing: -0.01em; }
p, span, label, .stMarkdown { color: var(--text-lo); }

.hero {
    padding: 28px 32px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(124,108,246,0.18), rgba(34,211,238,0.08));
    border: 1px solid var(--card-border);
    margin-bottom: 18px;
}
.hero h1 { font-size: 30px; margin: 0 0 4px 0; }
.hero p { font-size: 15px; margin: 0; color: var(--text-lo); }

.badge-row { display:flex; gap:8px; margin-top:14px; flex-wrap: wrap; }
.badge {
    padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600;
    border: 1px solid var(--card-border); background: rgba(255,255,255,0.03); color: var(--text-lo);
}
.badge.on { color: #0b0f1a; background: var(--accent-2); border-color: var(--accent-2); }
.badge.off { opacity: 0.55; }

.kpi-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 18px 20px;
    height: 100%;
}
.kpi-label { font-size: 12px; color: var(--text-lo); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px;}
.kpi-value { font-size: 28px; font-weight: 700; color: var(--text-hi); }
.kpi-sub { font-size: 12px; color: var(--text-lo); margin-top: 4px; }

.pill {
    display:inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 700; color: #0b0f1a;
}

.alert-card {
    background: var(--card);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.alert-meta { font-size: 12px; color: var(--text-lo); margin-bottom: 4px; }
.alert-comment { font-size: 14px; color: var(--text-hi); margin: 6px 0; }
.alert-why { font-size: 13px; color: var(--text-lo); font-style: italic; }

.section-title { font-size: 18px; font-weight: 700; color: var(--text-hi); margin: 6px 0 14px 0; }

div[data-testid="stMetricValue"] { color: var(--text-hi); }
hr { border-color: var(--card-border) !important; }

.stButton>button {
    background: linear-gradient(135deg, var(--accent), #5b4fd6);
    color: white; border: none; border-radius: 10px; font-weight: 600;
    padding: 0.55em 1.4em;
}
.stButton>button:hover { filter: brightness(1.1); }

div[data-baseweb="select"] > div { background: var(--card); border-color: var(--card-border); }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, integration_flags: dict):
    badges = "".join(
        f'<span class="badge {"on" if v else "off"}">{k} {"●" if v else "○"}</span>'
        for k, v in integration_flags.items()
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="badge-row">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value, sub: str = "", col=None):
    target = col if col is not None else st
    target.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def priority_pill(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority, "#94a3b8")
    return f'<span class="pill" style="background:{color};">{priority}</span>'


def sentiment_pill(sentiment: str) -> str:
    color = SENTIMENT_COLORS.get(sentiment, "#94a3b8")
    return f'<span class="pill" style="background:{color};">{sentiment}</span>'


def alert_card(alert: dict):
    jira_line = ""
    if alert.get("jira_ok") and alert.get("jira_url"):
        jira_line = f'🎫 <a href="{alert["jira_url"]}" target="_blank">{alert["jira_key"]}</a>'
    elif alert.get("jira_error"):
        jira_line = f'⚠️ {alert["jira_error"]}'

    st.markdown(
        f"""
        <div class="alert-card">
            <div class="alert-meta">
                {priority_pill(alert.get('priority','High'))} &nbsp;
                {sentiment_pill(alert.get('sentiment','Negative'))} &nbsp;
                <b>@{alert.get('author','Unknown')}</b> &nbsp;•&nbsp; {alert.get('platform','').title()} &nbsp;•&nbsp; {jira_line}
            </div>
            <div class="alert-comment">“{alert.get('comment','')}”</div>
            <div class="alert-why">Why: {alert.get('explanation','')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
