# Setup guide

Copy `.env.example` to `.env` and fill in whichever of these you want to use.
Every integration is optional — the dashboard degrades gracefully and tells
you in the sidebar which ones are active.

## 1. YouTube ingestion
1. Go to the [Google Cloud Console](https://console.cloud.google.com/) → enable **YouTube Data API v3**.
2. Create an API key under *Credentials*.
3. Set `YOUTUBE_API_KEY` in `.env`.

## 2. Instagram ingestion (via Apify)
1. Create a free account at [apify.com](https://apify.com).
2. Go to *Settings → Integrations* and copy your API token.
3. Set `APIFY_TOKEN` in `.env`.
   - Uses the `apify/instagram-comment-scraper` actor under the hood.

## 3. Bluesky ingestion (AT Protocol)
1. In the Bluesky app, go to **Settings → App Passwords → Add App Password**.
   Never use your main account password here.
2. Set `BLUESKY_HANDLE` (e.g. `yourname.bsky.social`) and `BLUESKY_PASSWORD`
   (the app password) in `.env`.
3. Pass any `https://bsky.app/profile/<handle>/post/<id>` URL into the
   dashboard's "Bluesky" platform option.

## 4. Jira ticketing (alert routing)
1. In Jira Cloud, go to **Account Settings → Security → API tokens** and
   create a token.
2. Set:
   - `JIRA_DOMAIN` — e.g. `yourcompany.atlassian.net`
   - `JIRA_EMAIL` — the email tied to the token
   - `JIRA_API_TOKEN` 
   - `JIRA_PROJECT_KEY` — the project comments should be filed under (e.g. `CS`)
   - `JIRA_ISSUE_TYPE` — must match a valid issue type in that project (e.g. `Task`, `Bug`)

## 5. Upstash Redis (alert de-duplication + counters)
1. Create a free database at [upstash.com](https://upstash.com) (Redis, REST API).
2. Copy the **REST URL** and **REST Token** from the database details page.
3. Set `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` in `.env`.

No server, no Docker container, no persistent connection required — it's a
plain HTTPS REST API, which is why it's used here instead of a self-hosted
Redis instance.

## 6. Gemini (optional — natural-language "why" explanations)
1. Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Set `GEMINI_API_KEY` in `.env`.
3. Used for: the AI post summary, and up to 5 natural-language deep
   explanations per alert run (capped to keep usage predictable). Everything
   else works fine without it via the rule-based explanation engine.

## Tuning alert behavior

```
ALERT_PRIORITY_THRESHOLD=High     # Low | Medium | High | Critical
ALERT_DEDUPE_TTL_HOURS=24         # hours before the same comment can re-alert
```
