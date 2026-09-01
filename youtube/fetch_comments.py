import requests
import pandas as pd
import os
import time
import sys
import io
import json
import warnings
import re

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    print("ERROR: YOUTUBE_API_KEY not found", file=sys.stderr)
    sys.exit(1)

def extract_video_id(url_or_id):
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtu\.be/([^?&/]+)",
        r"youtube\.com/shorts/([^?&/]+)",
        r"youtube\.com/embed/([^?&/]+)",
        r"youtube\.com/v/([^?&/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


def fetch_video_metadata(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "snippet,statistics", "id": video_id, "key": API_KEY}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        return {}
    items = resp.json().get("items", [])
    if not items:
        return {}
    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "view_count": int(stats.get("viewCount", 0)),
        "like_count": int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
    }


def fetch_youtube_comments(video_input):
    video_id = extract_video_id(video_input)
    if not video_id:
        print("ERROR: Invalid YouTube URL or ID", file=sys.stderr)
        sys.exit(1)

    print("Fetching video metadata...", file=sys.stderr)
    metadata = fetch_video_metadata(video_id)

    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments_data = []
    next_page = None

    print("Fetching comments...", file=sys.stderr)
    while True:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "pageToken": next_page,
            "order": "relevance",
            "key": API_KEY,
        }
        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            print(f"ERROR: YouTube API failed ({response.status_code})", file=sys.stderr)
            sys.exit(1)

        data = response.json()
        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments_data.append({
                "comment": top.get("textDisplay", ""),
                "author": top.get("authorDisplayName", "Unknown"),
                "like_count": top.get("likeCount", 0),
                "published_at": top.get("publishedAt", ""),
            })

        next_page = data.get("nextPageToken")
        if not next_page or len(comments_data) >= 1000:
            break

    if not comments_data:
        print("ERROR: No comments found (comments may be disabled)", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(comments_data).sort_values("like_count", ascending=False).reset_index(drop=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, f"comments_{ts}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    metadata_path = os.path.join(DATA_DIR, f"metadata_{ts}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "ai_summary": None}, f, indent=2, ensure_ascii=False)

    print(f"Fetched {len(df)} comments for '{metadata.get('title', video_id)}'", file=sys.stderr)
    print(csv_path)  # REQUIRED stdout contract


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: No video URL provided", file=sys.stderr)
        sys.exit(1)
    fetch_youtube_comments(sys.argv[1])
