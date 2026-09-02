# Render Deployment - Final Fix for 429 Errors

## The Real Problem

You're getting **429 errors during ANALYSIS**, not during data fetching. This means:

❌ Data fetches successfully (YouTube API works)  
❌ Analysis starts but fails with 429  
❌ The issue is the **HuggingFace model downloading during runtime**

## The Solution

Models must be **pre-downloaded during build**, not during runtime/analysis.

### What I Changed:

1. **`download_models.py`** - New script that downloads models during build
2. **`build.sh`** - Custom build script for Render
3. **`src/analyzer.py`** - Now tries offline mode first (cache-only)
4. **`render.yaml`** - Updated to use new build process

### How It Works:

```
Build Phase (happens once):
1. Install packages
2. Download models → /tmp/huggingface_cache
3. Verify models work
4. Start app

Runtime (every request):
1. Load models from cache (offline mode)
2. No download needed
3. No 429 errors
4. Fast analysis
```

## Deploy on Render - Step by Step

### Step 1: Clear Build Cache
1. Go to Render Dashboard
2. Select your service
3. Go to **Settings**
4. Scroll down to **Build Cache**
5. Click **"Clear Build Cache"**

### Step 2: Manual Deploy
1. Still in Settings, go to **Manual Deploy** section
2. Click **"Deploy latest commit"**
3. Or click **"Clear build cache & deploy"**

### Step 3: Watch Build Logs
The build will show:
```bash
🚀 Starting Render build process...
📦 Installing Python packages...
🔥 Pre-downloading AI models...
📥 Downloading cardiffnlp/twitter-xlm-roberta-base-sentiment
   ⏳ Downloading tokenizer...
   ✅ Tokenizer downloaded
   ⏳ Downloading model (this may take 2-5 minutes)...
   ✅ Model downloaded
   🧪 Testing model...
   ✅ Model verified working
✅ All models downloaded and cached successfully!
✅ Build complete! Starting application...
```

**Build time**: 10-15 minutes (first time only)

### Step 4: Test
1. Wait for app to start (you'll see "Live" status)
2. Try analyzing 50 comments
3. Should complete in 1-2 minutes
4. ✅ No more 429 errors!

## What If Build Fails?

### Scenario 1: 429 During Build

**Symptoms:**
```
⚠️ Rate limited (429). Waiting 2s before retry...
⚠️ Rate limited (429). Waiting 4s before retry...
❌ Failed after 5 attempts due to rate limiting
```

**Solution:**
Wait 1 hour, then redeploy. HuggingFace has hourly rate limits.

**Or:**
Add `HF_TOKEN` environment variable:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```
Get token from: https://huggingface.co/settings/tokens

### Scenario 2: Build Succeeds but Still 429 at Runtime

**Check Render Logs for:**
```
⏳ Loading multilingual sentiment model...
   Attempting to load from cache (offline mode)...
   ⚠️ Cache miss: [...]
   Downloading from HuggingFace Hub...
```

This means the cache didn't work. **Solution:**

1. Check `/tmp/huggingface_cache` exists
2. Environment variables set correctly:
   ```
   HF_HOME=/tmp/huggingface_cache
   TRANSFORMERS_CACHE=/tmp/huggingface_cache
   ```
3. Redeploy with cleared cache

### Scenario 3: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
Make sure `requirements.txt` has:
```
transformers>=4.35.0
torch>=2.0.0
torchvision>=0.15.0
sentencepiece>=0.1.99
tiktoken>=0.5.0
```

## Render Configuration

### Build Command (in render.yaml or dashboard):
```bash
bash build.sh
```

### Start Command:
```bash
streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0 --logger.level=error
```

### Environment Variables (Required):
```
HF_HOME=/tmp/huggingface_cache
TRANSFORMERS_CACHE=/tmp/huggingface_cache
HF_HUB_ETAG_TIMEOUT=60
HF_HUB_DISABLE_SYMLINKS_WARNING=1
TOKENIZERS_PARALLELISM=false
```

Plus your API keys:
```
YOUTUBE_API_KEY=your_key
APIFY_TOKEN=your_token
BLUESKY_HANDLE=your_handle
BLUESKY_PASSWORD=your_password
```

## Testing After Deployment

### 1. Check Build Logs
Look for:
- ✅ "All models downloaded and cached successfully!"
- ✅ "Build complete!"

### 2. Check Runtime Logs
When you analyze comments, look for:
- ✅ "Loaded from cache successfully!"
- ❌ NOT "Downloading from HuggingFace Hub" (this means cache failed)

### 3. Test Analysis
1. Fetch 50 comments from YouTube
2. Analysis should complete in 1-2 minutes
3. Results should appear
4. No 429 errors

## Performance Metrics

| Metric | Expected |
|--------|----------|
| Build Time (first) | 10-15 min |
| Build Time (subsequent) | 2-3 min |
| App Startup | 30-60 sec |
| Model Load | 5-10 sec (from cache) |
| 50 Comments Analysis | 1-2 min |
| 100 Comments Analysis | 2-3 min |

## Troubleshooting Checklist

- [ ] Cleared build cache before deploying
- [ ] Build completed successfully
- [ ] Saw "All models downloaded and cached" in build logs
- [ ] Environment variables set correctly
- [ ] App shows "Live" status
- [ ] Tried analyzing comments
- [ ] Checked runtime logs for "Loaded from cache"

## If Still Getting 429

### Option 1: Use Smaller Model
Edit `src/analyzer.py` line ~75:
```python
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # 700MB instead of 1GB
```

### Option 2: Use HuggingFace Token
Add to Render environment:
```
HF_TOKEN=hf_your_token_here
```
This gives higher rate limits.

### Option 3: Disable ML Model
Add to Render environment:
```
USE_ML_MODEL=false
```
Falls back to VADER (English only, but faster).

## Success Indicators

✅ Build logs show models cached  
✅ Runtime logs show "Loaded from cache"  
✅ Analysis completes without 429  
✅ Results appear correctly  
✅ Multiple analyses work consistently  

## Next Steps

1. **Clear Render build cache**
2. **Redeploy**
3. **Wait 10-15 minutes** for build
4. **Test with 50 comments**
5. **Check logs** to verify cache working
6. **Celebrate!** 🎉

Your app should now work perfectly on Render!
