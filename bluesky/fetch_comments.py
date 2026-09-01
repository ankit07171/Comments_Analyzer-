"""
bluesky/fetch_comments.py
----------------------------
Fetches replies ("comments") on a Bluesky post via the AT Protocol, using
the official `atproto` Python SDK. Requires a Bluesky app password
(BLUESKY_HANDLE / BLUESKY_PASSWORD) — never use your main account password,
generate one at bsky.app -> Settings -> App Passwords.

Contract (used by dashboard.py):
  - prints ONLY the resulting CSV path to stdout on success
  - writes progress/diagnostics to stderr
  - writes data/comments_<ts>.csv and data/metadata_<ts>.json

Usage:
    python bluesky/fetch_comments.py --url "https://bsky.app/profile/<handle>/post/<rkey>"
"""

import os
import re
import sys
import io
import json
import time
import argparse

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")


def parse_post_url(url: str):
    """Accepts a bsky.app URL or a raw at:// URI."""
    if url.startswith("at://"):
        return url, None

    match = re.search(r"bsky\.app/profile/([^/]+)/post/([^/?#]+)", url)
    if not match:
        return None, None
    handle, rkey = match.group(1), match.group(2)
    return None, (handle, rkey)


def flatten_replies(thread_node, out: list, depth: int = 0):
    """Recursively walk the reply tree returned by getPostThread."""
    replies = getattr(thread_node, "replies", None) or []
    for reply in replies:
        post = getattr(reply, "post", None)
        if post is None:
            continue
        record = getattr(post, "record", None)
        author = getattr(post, "author", None)
        out.append({
            "comment": getattr(record, "text", "") if record else "",
            "author": getattr(author, "handle", "Unknown") if author else "Unknown",
            "like_count": getattr(post, "like_count", 0) or 0,
            "published_at": getattr(record, "created_at", "") if record else "",
            "depth": depth,
        })
        flatten_replies(reply, out, depth + 1)


def scrape_bluesky(post_url: str):
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        print("ERROR: BLUESKY_HANDLE / BLUESKY_PASSWORD not set", file=sys.stderr)
        sys.exit(1)

    try:
        from atproto import Client
    except ImportError:
        print("ERROR: install the 'atproto' package (pip install atproto)", file=sys.stderr)
        sys.exit(1)

    print("Authenticating with Bluesky...", file=sys.stderr)
    client = Client()
    client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)

    at_uri, handle_rkey = parse_post_url(post_url)

    if at_uri is None:
        handle, rkey = handle_rkey
        print(f"Resolving handle @{handle}...", file=sys.stderr)
        did_resp = client.com.atproto.identity.resolve_handle({"handle": handle})
        did = did_resp.did
        at_uri = f"at://{did}/app.bsky.feed.post/{rkey}"

    print("Fetching post thread...", file=sys.stderr)
    thread_resp = client.app.bsky.feed.get_post_thread({"uri": at_uri, "depth": 10})
    thread = thread_resp.thread

    root_post = getattr(thread, "post", None)
    root_record = getattr(root_post, "record", None)
    root_author = getattr(root_post, "author", None)

    post_metadata = {
        "post_uri": at_uri,
        "url": post_url,
        "username": getattr(root_author, "handle", "Unknown") if root_author else "Unknown",
        "caption": getattr(root_record, "text", "") if root_record else "",
        "like_count": getattr(root_post, "like_count", 0) or 0,
        "repost_count": getattr(root_post, "repost_count", 0) or 0,
        "comment_count": getattr(root_post, "reply_count", 0) or 0,
        "view_count": 0,  # not exposed by AAT Protocol for posts
        "timestamp": getattr(root_record, "created_at", "") if root_record else "",
    }

    comments_data = []
    flatten_replies(thread, comments_data)

    if not comments_data:
        print("ERROR: No replies found on this post", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(comments_data).sort_values("like_count", ascending=False).reset_index(drop=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, f"comments_{ts}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    metadata_path = os.path.join(DATA_DIR, f"metadata_{ts}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": post_metadata, "ai_summary": None}, f, indent=2, ensure_ascii=False)

    print(f"Fetched {len(df)} replies from @{post_metadata['username']}", file=sys.stderr)
    print(csv_path)  # REQUIRED stdout contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="bsky.app post URL or at:// URI")
    args = parser.parse_args()
    scrape_bluesky(args.url)
