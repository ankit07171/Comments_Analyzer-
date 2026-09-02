#!/bin/bash
# Render build script - downloads models BEFORE starting the app

set -e  # Exit on error

echo "🚀 Starting Render build process..."

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set HuggingFace cache location to persistent storage
export HF_HOME=/opt/render/project/.cache/huggingface
export TRANSFORMERS_CACHE=/opt/render/project/.cache/huggingface
mkdir -p $HF_HOME

echo "📥 Pre-downloading multilingual sentiment model..."
echo "   This happens during build, not at runtime (prevents 429 errors)"

# Download the model using Python
python3 << 'PYTHON_SCRIPT'
import os
import sys

# Configure cache
cache_dir = "/opt/render/project/.cache/huggingface"
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.makedirs(cache_dir, exist_ok=True)

print("⏳ Downloading cardiffnlp/twitter-xlm-roberta-base-sentiment...")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    
    # Download tokenizer (small, fast)
    print("   📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("   ✅ Tokenizer downloaded")
    
    # Download model (larger, ~1GB)
    print("   📥 Downloading model (this takes 3-5 minutes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("   ✅ Model downloaded and cached")
    
    print(f"\n✅ Model successfully cached at: {cache_dir}")
    print("   The app will load instantly from cache at runtime!")
    
except Exception as e:
    print(f"\n❌ ERROR downloading model: {e}")
    print("   The app will try to download at runtime (may cause 429 errors)")
    sys.exit(1)

PYTHON_SCRIPT

echo ""
echo "✅ Build complete! Models are cached and ready."
echo "   Cache location: $HF_HOME"
echo "   App startup will be fast (loads from cache)"
