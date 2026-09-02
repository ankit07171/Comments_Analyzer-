# Setup Instructions - Lightweight Multilingual Model

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **Streamlit** - UI framework
- **Transformers** - HuggingFace models
- **PyTorch** - ML framework
- **Torchvision** - Computer vision utilities (required by transformers)
- **Scikit-learn** - For spam detection
- **VADER** - Sentiment analysis fallback
- **Ingestion APIs** - YouTube, Instagram, Bluesky

### 2. Setup Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Then edit `.env` with:
- `YOUTUBE_API_KEY` - YouTube Data API key
- `APIFY_TOKEN` - Apify API for Instagram scraping
- `BLUESKY_HANDLE` & `BLUESKY_PASSWORD` - Bluesky credentials
- (Optional) `GEMINI_API_KEY` - For AI-powered explanations
- (Optional) Jira and Redis credentials for integrations

### 3. Run Locally

```bash
streamlit run dashboard.py
```

The app will:
1. Load on `http://localhost:8501`
2. Download the multilingual sentiment model on first run (~1GB, takes 2-5 minutes)
3. Cache the model locally for future runs
4. Start analyzing comments

## Model Details

### Sentiment Analysis Model
- **Name**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **Size**: ~1GB (downloaded on first use)
- **Languages**: 100+ languages (multilingual)
- **Training Data**: Twitter comments
- **Download Location**: `~/.cache/huggingface/hub/`

### How Language Detection Works
1. Automatic detection using character patterns
2. Falls back to English if unsure
3. (Optional) Can use fastText for better accuracy

### Analysis Flow
1. **Detect language** of comment
2. **Sentiment**: Use multilingual transformer model
3. **Intent**: Rule-based keyword analysis (works all languages)
4. **Toxicity/Spam**: Rule-based keyword detection
5. **Priority**: Score based on sentiment + intent + toxicity

## Deployment on Render

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Lightweight multilingual model"
git push origin main
```

### Step 2: Create Render Service
1. Go to https://render.com
2. New → Web Service
3. Connect GitHub repo
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`
   - **Python Version**: 3.9+

### Step 3: Environment Variables on Render
Add all your API keys from `.env` to Render's environment settings.

### Important: Remove These Variables if Present
- ❌ `SENTIMENT_MODEL=vader` (Forces VADER only)
- ❌ `ENABLE_MODEL_DOWNLOAD` (Not needed)
- ❌ `GITHUB_RELEASE_MODEL_URL` (Not needed)

### Step 4: Deploy
Render will automatically:
1. Install dependencies from `requirements.txt`
2. Download the multilingual model on first run
3. Cache it for future restarts
4. Use ~2GB RAM for the model

## Performance

| Metric | Value |
|--------|-------|
| First Run (with download) | ~2-5 min |
| Subsequent Runs | < 30 sec |
| Analysis Speed | ~50 comments/sec on CPU |
| Memory Usage | ~2GB |
| Disk Space (model cache) | ~1GB |

## Troubleshooting

### Model Download Fails
- Check internet connection
- Verify `transformers` is installed
- Check HuggingFace Hub status

### Slow Performance
- First run is slow due to model download
- Subsequent runs use cached model
- Reduce batch size in analyzer if needed

### Out of Memory
- Render Free tier: 512MB RAM (insufficient)
- Upgrade to Starter plan (1GB RAM)
- Or use smaller model

### Torchvision Missing Error
- Run: `pip install torchvision`
- Already in `requirements.txt`

## File Structure

```
.
├── requirements.txt              # Dependencies (includes torchvision, scikit-learn)
├── .gitignore                    # Excludes model cache, .pt files
├── dashboard.py                  # Main Streamlit app
├── src/
│   ├── analyzer.py              # Multilingual sentiment/intent analysis
│   ├── simi.py                  # Spam similarity detection (needs scikit-learn)
│   ├── ui_theme.py              # UI components
│   └── ...
├── models/
│   └── models/                  # (No large files - models download at runtime)
└── data/
    └── comments_*.csv           # Analysis results
```

## What's Different from Previous Version

| Feature | Old | New |
|---------|-----|-----|
| Model Size | 417 MB | ~1 GB (auto-download) |
| Multilingual | ❌ English only | ✅ 100+ languages |
| Deployment | Complex (GitHub Releases) | Simple (just push code) |
| Model Files | Committed to Git | Downloaded at runtime |
| Sentiment Analysis | Custom BERT | XLM-RoBERTa (pre-trained) |
| Intent Analysis | Custom BERT | Rule-based (fast) |
| Toxicity Detection | Custom rules | Rule-based (fast) |
| Render Deployment | Manual uploads needed | Automatic |

## Support

For issues:
1. Check that all packages installed: `pip list | grep -E "transformers|torch|scikit-learn"`
2. Check Python version: `python --version` (should be 3.9+)
3. Try reinstalling requirements: `pip install -r requirements.txt --force-reinstall`
