#!/usr/bin/env python3
"""
Pre-download models during Render build to avoid 429 errors during runtime.
Run this script during build phase, not at request time.
"""

import os
import sys
import time

# Set cache locations for Render
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print("🚀 Pre-downloading models for Render deployment...")

def download_with_retry(model_name, max_retries=5):
    """Download model with retry logic to handle rate limits."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    for attempt in range(max_retries):
        try:
            print(f"\n📥 Downloading {model_name} (attempt {attempt + 1}/{max_retries})...")
            
            # Download tokenizer
            print("   ⏳ Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir='/tmp/huggingface_cache',
                resume_download=True,
                force_download=False,
                local_files_only=False
            )
            print("   ✅ Tokenizer downloaded")
            
            # Download model
            print("   ⏳ Downloading model (this may take 2-5 minutes)...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir='/tmp/huggingface_cache',
                resume_download=True,
                force_download=False,
                local_files_only=False
            )
            print("   ✅ Model downloaded")
            
            # Verify it works
            print("   🧪 Testing model...")
            test_text = "This is a test comment"
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            print("   ✅ Model verified working")
            
            return True
            
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'rate' in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    print(f"   ⚠️ Rate limited (429). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"   ❌ Failed after {max_retries} attempts due to rate limiting")
                    return False
            else:
                print(f"   ❌ Error: {error_str}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in 2s...")
                    time.sleep(2)
                    continue
                return False
    
    return False

def main():
    try:
        # Ensure cache directory exists
        os.makedirs('/tmp/huggingface_cache', exist_ok=True)
        
        # Download the multilingual sentiment model
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        
        success = download_with_retry(model_name, max_retries=5)
        
        if success:
            print("\n✅ All models downloaded and cached successfully!")
            print(f"   Cache location: /tmp/huggingface_cache")
            print(f"   Models will be used at runtime without re-downloading")
            return 0
        else:
            print("\n⚠️ Model download failed. The app will try to download at runtime.")
            print("   This may be slower and could hit rate limits.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   The app will fall back to VADER sentiment analysis")
        return 1

if __name__ == "__main__":
    sys.exit(main())
