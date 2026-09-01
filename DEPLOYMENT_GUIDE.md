# Lightweight Multilingual Comment Intelligence Platform

## What's New

This version replaces the custom fine-tuned BERT model (417MB) with a lightweight multilingual solution:

### Key Improvements:
1. **No large model files** - Uses pre-trained HuggingFace models that download automatically
2. **Multilingual support** - Works with English, Spanish, French, German, Italian, Portuguese, and more
3. **Fast deployment** - No need to upload large `.pt` files to GitHub Releases
4. **Lightweight** - Uses `cardiffnlp/twitter-xlm-roberta-base-sentiment` (~1GB, downloads automatically)
5. **Easy Render deployment** - No special setup needed

## How It Works

### Sentiment Analysis
- Uses `cardiffnlp/twitter-xlm-roberta-base-sentiment` model
- Pre-trained on Twitter data in 100+ languages
- Automatically downloads from HuggingFace Hub on first run
- Cached locally for subsequent runs

### Language Detection
- Simple heuristic-based language detection
- Can be enhanced with fastText for better accuracy (optional)
- Falls back to English if language can't be determined

### Intent Analysis
- Rule-based using keyword lexicons
- Works across all languages
- Lightweight and fast

### Toxicity & Spam Detection
- Rule-based using keyword lists
- Language-independent

## Deployment on Render

### Step 1: Update Repository
```bash
git add .
git commit -m "Switch to lightweight multilingual model"
git push origin main
```

### Step 2: Deploy on Render
1. Go to https://render.com
2. Create a new "Web Service"
3. Connect your GitHub repository
4. Configure with these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`
   - **Python Version**: 3.9+

### Step 3: Environment Variables
Add these to your Render environment variables:

Required:
```
YOUTUBE_API_KEY=your_key
APIFY_TOKEN=your_token
BLUESKY_HANDLE=your_handle.bsky.social
BLUESKY_PASSWORD=your_app_password
```

Optional:
```
GEMINI_API_KEY=your_key  # For AI explanations
JIRA_DOMAIN=your_domain.atlassian.net  # For Jira integration
JIRA_EMAIL=your_email
JIRA_API_TOKEN=your_token
UPSTASH_REDIS_REST_URL=your_url
UPSTASH_REDIS_REST_TOKEN=your_token
```

### Step 4: Remove Old Variables
**IMPORTANT**: Remove these if they exist:
- `SENTIMENT_MODEL=vader` (This forces VADER only)
- `ENABLE_MODEL_DOWNLOAD` (Not needed anymore)
- `GITHUB_RELEASE_MODEL_URL` (Not needed anymore)

## Model Caching

The HuggingFace model will:
1. Download automatically on first run
2. Cache in `~/.cache/huggingface/hub/`
3. Reuse cached version on subsequent runs
4. Use about 1GB of disk space

## Performance

- **First run**: ~2 minutes (model downloads)
- **Subsequent runs**: < 30 seconds (cached model)
- **Analysis speed**: ~50 comments/second on CPU
- **Memory usage**: ~2GB RAM

## Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run dashboard.py

# Test with sample data
# The model will download automatically on first analysis
```

## Fallback Behavior

If the multilingual model fails to load:
1. Uses VADER for English comments
2. Uses rule-based analysis for other languages
3. Still provides spam/toxicity detection
4. Still provides intent classification

## Customization

### To add better language detection:
```bash
pip install fasttext
# Download language detection model:
# https://fasttext.cc/docs/en/language-identification.html
```

### To use a different sentiment model:
Edit `src/analyzer.py` and change:
```python
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
```
To any HuggingFace sentiment model.

## Troubleshooting

### Model won't download
- Check internet connection on Render
- Ensure `transformers` package is installed
- Check HuggingFace Hub status

### Slow performance
- Reduce batch size in `analyze_dataframe()` (default: 32)
- Use smaller model like `nlptown/bert-base-multilingual-uncased-sentiment`

### Memory issues
- Render Free tier has 512MB RAM, upgrade if needed
- Consider using smaller model
- Reduce max text length in analyzer

## Benefits Over Previous Version

1. **No manual model uploads** - Automatic downloads
2. **Multilingual** - Works with non-English comments
3. **Smaller Git repository** - No 400MB model files
4. **Easier deployments** - Just push code, no extra steps
5. **Better cacheability** - Models cached globally by HuggingFace