# from apify_client import ApifyClient
# import pandas as pd
# import time, os, argparse, sys, io

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# def scrape_instagram(reel_url, token):
#     if not token or not token.strip():
#         print("ERROR: Empty APIFY token", file=sys.stderr)
#         sys.exit(1)

#     client = ApifyClient(token)
#     comments = []

#     run = client.actor("apify/instagram-comment-scraper").call(
#         run_input={
#             "directUrls": [reel_url],
#             "maxRequestsPerCrawl": 1,
#             "maxComments": 100
#         }
#     )

#     while True:
#         status = client.run(run["id"]).get()["status"]
#         if status in ("SUCCEEDED", "FAILED"):
#             break
#         time.sleep(5)
#         print(f"Status: {status}", file=sys.stderr)

#     if status != "SUCCEEDED":
#         print(f"Run failed: {status}", file=sys.stderr)
#         sys.exit(1)

#     dataset = client.dataset(run["defaultDatasetId"]).list_items().items
#     for item in dataset:
#         if item.get("text"):
#             comments.append(item["text"].strip())

#     if not comments:
#         print("No comments found", file=sys.stderr)
#         sys.exit(1)

#     df = pd.DataFrame({"comment": comments})

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DATA_DIR = os.path.join(BASE_DIR, "../data")
#     os.makedirs(DATA_DIR, exist_ok=True)

#     ts = time.strftime("%Y%m%d_%H%M%S")
#     csv_path = os.path.abspath(os.path.join(DATA_DIR, f"comments_{ts}.csv"))

#     df.to_csv(csv_path, index=False, encoding="utf-8-sig")
#     print(csv_path)  # ONLY stdout

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("--url", required=True)
#     args = parser.parse_args()
#     print(args)
#     print(args.url)
#     APIFY_TOKEN = os.getenv("")
#     scrape_instagram(args.url, APIFY_TOKEN)

 
import pandas as pd
import time, os, argparse, sys, io 
from apify_client import ApifyClient
import requests
import json
import re

from dotenv import load_dotenv 
load_dotenv()

# UTF-8 output (Windows fix)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# API Keys
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- HELPERS ----------
def extract_reel_id(url):
    """Extract reel ID from Instagram URL"""
    patterns = [
        r"instagram\.com/reel/([A-Za-z0-9_-]+)",
        r"instagram\.com/p/([A-Za-z0-9_-]+)",
        r"instagram\.com/reels/([A-Za-z0-9_-]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def get_gemini_summary(reel_metadata):
    """Get AI-generated summary of Instagram reel using Gemini API"""
    if not GEMINI_API_KEY:
        return "Gemini API key not configured"
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        
        caption = reel_metadata.get('caption', '')
        username = reel_metadata.get('username', 'Unknown')
        
        prompt = f"""Analyze this Instagram reel and provide a concise summary:

Username: @{username}
Caption: {caption[:500] if caption else 'No caption'}

Provide a brief 2-3 sentence summary of what this reel is about based on the caption and context."""

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            summary = result["candidates"][0]["content"]["parts"][0]["text"]
            return summary
        else:
            return f"Failed to generate summary (Status: {response.status_code})"
            
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def scrape_instagram(reel_url, token):
    if not token:
        print("ERROR: APIFY_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    print("Fetching Instagram reel data...", file=sys.stderr)
    
    reel_id = extract_reel_id(reel_url)
    
    client = ApifyClient(token)

    # Enhanced scraper with more data
    run = client.actor("apify/instagram-comment-scraper").call(
        run_input={
            "directUrls": [reel_url],
            "maxComments": 500,
            "resultsLimit": 500
        }
    )

    # Wait until finished
    print("Scraping in progress...", file=sys.stderr)
    while True:
        status = client.run(run["id"]).get()["status"]
        if status in ("SUCCEEDED", "FAILED", "ABORTED"):
            break
        time.sleep(3)

    if status != "SUCCEEDED":
        print(f"ERROR: Instagram scrape failed with status: {status}", file=sys.stderr)
        sys.exit(1)

    # Fetch dataset
    dataset = client.dataset(run["defaultDatasetId"]).list_items().items
    
    if not dataset:
        print("ERROR: No data returned from scraper", file=sys.stderr)
        sys.exit(1)

    # Extract reel metadata from first item
    first_item = dataset[0] if dataset else {}
    
    reel_metadata = {
        "reel_id": reel_id or "unknown",
        "url": reel_url,
        "username": first_item.get("ownerUsername", "Unknown"),
        "caption": first_item.get("caption", ""),
        "like_count": first_item.get("likesCount", 0),
        "comment_count": first_item.get("commentsCount", 0),
        "view_count": first_item.get("videoViewCount", 0),
        "timestamp": first_item.get("timestamp", "")
    }
    
    # Get Gemini summary
    print("Generating AI summary...", file=sys.stderr)
    ai_summary = get_gemini_summary(reel_metadata)
    
    # Print reel info
    print("\n" + "="*60, file=sys.stderr)
    print(f"REEL: @{reel_metadata['username']}", file=sys.stderr)
    print(f"VIEWS: {reel_metadata['view_count']:,}", file=sys.stderr)
    print(f"LIKES: {reel_metadata['like_count']:,}", file=sys.stderr)
    print(f"COMMENTS: {reel_metadata['comment_count']:,}", file=sys.stderr)
    if reel_metadata['caption']:
        print(f"\nCAPTION: {reel_metadata['caption'][:100]}...", file=sys.stderr)
    print(f"\nAI SUMMARY:\n{ai_summary}", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    # Extract comments with metadata
    comments_data = []
    for item in dataset:
        if item.get("text"):
            comments_data.append({
                "comment": item["text"].strip(),
                "author": item.get("ownerUsername", "Unknown"),
                "like_count": item.get("likesCount", 0),
                "timestamp": item.get("timestamp", "")
            })

    if not comments_data:
        print("ERROR: No comments found", file=sys.stderr)
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(comments_data)
    
    # Sort by likes
    df = df.sort_values("like_count", ascending=False).reset_index(drop=True)
    
    # Add metadata
    df.attrs["metadata"] = reel_metadata
    df.attrs["ai_summary"] = ai_summary

    # Save CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, f"comments_{ts}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # Save metadata
    metadata_path = os.path.join(DATA_DIR, f"metadata_{ts}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": reel_metadata,
            "ai_summary": ai_summary
        }, f, indent=2, ensure_ascii=False)
    
    # Print top 5 most liked comments
    print("\nTOP 5 MOST LIKED COMMENTS:", file=sys.stderr)
    for idx, row in df.head(5).iterrows():
        print(f"\n[{row['like_count']} likes] @{row['author']}:", file=sys.stderr)
        print(f"  {row['comment'][:100]}...", file=sys.stderr)

    print(csv_path)  # REQUIRED OUTPUT  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    APIFY_TOKEN = os.getenv("APIFY_TOKEN")
    if not APIFY_TOKEN:
        print("ERROR: APIFY_TOKEN env var missing", file=sys.stderr)
        sys.exit(1) 
    scrape_instagram(args.url, APIFY_TOKEN)