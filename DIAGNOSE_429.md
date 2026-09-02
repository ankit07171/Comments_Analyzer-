# Diagnose 429 Error on Render

## Where is the 429 Error Coming From?

The 429 error can come from **3 different sources**:

### 1. YouTube API (Most Likely)
**Quota**: 10,000 units/day (free tier)
- Each comment fetch = ~100 units
- You can fetch ~100 videos/day
- Resets at midnight PST

**How to check:**
1. Go to: https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas
2. Log in with your Google account
3. Check "Queries per day" quota
4. If at or near 10,000 - **This is your issue**

**Solution:**
- Wait until midnight PST for quota reset
- Or request quota increase from Google
- Or reduce comments per video (change `maxResults` to 50)

### 2. Instagram API (Apify)
**Quota**: Depends on your Apify plan
- Free: 5 runs/month
- Paid: Varies by plan

**How to check:**
1. Go to: https://console.apify.com/
2. Check your usage limits
3. Check if you've exceeded runs

**Solution:**
- Upgrade Apify plan
- Or use fewer Instagram fetches

### 3. HuggingFace Model Download
**Quota**: Usually not an issue, but possible
- Rate limited on model downloads
- Should only happen on first run

**How to check:**
Look at Render logs for:
```
⚠️ Could not load sentiment model: 429
```

**Solution:**
- Already fixed with retry logic in latest code
- Should auto-retry and succeed

## How to Diagnose on Render

### Step 1: Check Render Logs

1. Go to Render Dashboard
2. Select your service  
3. Click **"Logs"** tab
4. Look for error messages

### Look For These Patterns:

**YouTube 429:**
```
ERROR: YouTube API failed (429)
ERROR: YouTube API rate limit exceeded
```
➡️ **Fix**: Wait for quota reset or upgrade YouTube API quota

**Instagram 429:**
```
ERROR: Apify actor failed
ERROR: Instagram scraping failed (429)
```
➡️ **Fix**: Check Apify quota

**HuggingFace 429:**
```
⚠️ Could not load sentiment model
HTTPError: 429 Client Error
```
➡️ **Fix**: Already fixed, redeploy if you haven't

**Bluesky 429:**
```
ERROR: Bluesky API rate limited
```
➡️ **Fix**: Add delays between requests

## Quick Diagnostic Test

Add this to your Render environment variables to see detailed logs:

```
LOG_LEVEL=DEBUG
STREAMLIT_LOG_LEVEL=debug
```

Then check logs again for exact error source.

## Most Common Cause: YouTube API Quota

**YouTube API Free Tier Limits:**
- **10,000 units per day**
- Fetching comments = ~1 unit per comment
- Video metadata = ~1 unit
- **Typical usage**: 100 comments = ~100 units

**Daily limit example:**
- 100 videos × 100 comments = 10,000 units
- After that, you get 429 until midnight PST

### How to Fix YouTube Quota Issues:

#### Option 1: Wait for Reset
- Quota resets at midnight PST
- Check current quota: https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas
- Try again tomorrow

#### Option 2: Request Quota Increase
1. Go to: https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas
2. Click "QUOTAS" tab
3. Select "Queries per day"
4. Click "EDIT QUOTAS"
5. Request increase to 1,000,000 (usually granted)
6. Wait 1-2 business days for approval

#### Option 3: Reduce Comments Per Request
Edit `youtube/fetch_comments.py`:
```python
# Change line with maxResults
"maxResults": 50,  # Was 100, now 50 = half the quota usage
```

#### Option 4: Use Multiple API Keys
1. Create additional Google Cloud projects
2. Enable YouTube API on each
3. Get API keys from each
4. Rotate between keys in your code

## Instagram/Apify Quota Issues

**Apify Free Tier:**
- 5 actor runs per month
- Each Instagram fetch = 1 run

**Fix:**
- Upgrade to paid plan ($49/month for 100 runs)
- Or reduce Instagram usage
- Or use alternative scraping method

## Test After Fixing

1. **Try a single video** with few comments first
2. **Check Render logs** for success messages
3. **Try 50 comments** if single video works
4. **Monitor quota usage** at Google Console

## Render-Specific Environment Variables

Add these to Render to help with rate limiting:

```bash
# YouTube API
YOUTUBE_API_KEY=your_key_here

# If you have multiple keys for rotation:
YOUTUBE_API_KEY_2=your_second_key
YOUTUBE_API_KEY_3=your_third_key

# Rate limiting
YOUTUBE_MAX_RESULTS=50  # Reduce from 100 to save quota
COMMENTS_DELAY_MS=500   # Delay between pages
```

## Summary: What's Most Likely

Based on typical usage:

1. **90% chance**: YouTube API quota exceeded
   - **Fix**: Wait for reset or request increase
   
2. **5% chance**: Apify/Instagram quota exceeded
   - **Fix**: Upgrade Apify plan
   
3. **5% chance**: HuggingFace rate limit
   - **Fix**: Already fixed in latest code

## Need More Help?

Share your **Render logs** (the error messages) to identify exact issue.

Look for lines containing:
- `ERROR`
- `429`
- `rate limit`
- `quota exceeded`

This will tell us exactly which API is causing the problem.
