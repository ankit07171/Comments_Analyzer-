"""
Check which Gemini models are available with your API key
"""

import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No API key found")
    exit(1)

print("Checking available Gemini models...\n")

# List all available models
url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"

try:
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        models = data.get("models", [])
        
        print(f"Found {len(models)} available models:\n")
        
        # Filter for models that support generateContent
        generate_content_models = []
        
        for model in models:
            name = model.get("name", "")
            supported_methods = model.get("supportedGenerationMethods", [])
            
            if "generateContent" in supported_methods:
                generate_content_models.append(name)
                display_name = name.replace("models/", "")
                print(f"✅ {display_name}")
        
        print(f"\n{len(generate_content_models)} models support generateContent")
        
        # Recommend best option
        print("\n" + "="*60)
        print("RECOMMENDATION:")
        print("="*60)
        
        if any("gemini-1.5-flash" in m for m in generate_content_models):
            print("Use: gemini-1.5-flash (fastest, cheapest)")
        elif any("gemini-1.5-pro" in m for m in generate_content_models):
            print("Use: gemini-1.5-pro (balanced)")
        elif any("gemini-pro" in m for m in generate_content_models):
            print("Use: gemini-pro (stable, widely available)")
        elif any("gemini-1.0-pro" in m for m in generate_content_models):
            print("Use: gemini-1.0-pro")
        else:
            print("No suitable models found")
            
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")
