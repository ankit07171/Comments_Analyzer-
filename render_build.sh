#!/bin/bash

# Render build script - ensures models are cached before app starts
echo "🚀 Render Build Script Starting..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --no-cache-dir

# Pre-warm the HuggingFace model cache
echo "🔥 Pre-warming HuggingFace model cache..."
python3 << 'EOF'
import os
import sys

# Set cache location
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

print("⏳ Downloading and caching multilingual sentiment model...")
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    print(f"   Model: {model_name}")
    
    # Download tokenizer
    print("   Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='/tmp/huggingface_cache',
        timeout=60
    )
    print("   ✅ Tokenizer cached")
    
    # Download model
    print("   Downloading model (this may take 2-5 minutes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir='/tmp/huggingface_cache',
        timeout=60
    )
    print("   ✅ Model cached")
    
    print("✅ Models successfully cached for Render deployment!")
    
except Exception as e:
    print(f"⚠️ Failed to pre-warm models: {e}")
    print("   The app will try to download on first run (slower startup)")
    sys.exit(0)  # Don't fail the build, just log the warning

EOF

echo "✅ Build complete! App is ready to start."
