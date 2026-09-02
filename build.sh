#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting Render build process..."

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Pre-download models to avoid 429 errors at runtime
echo "🔥 Pre-downloading AI models..."
python3 download_models.py

echo "✅ Build complete! Starting application..."
