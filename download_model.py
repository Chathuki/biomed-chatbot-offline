from sentence_transformers import SentenceTransformer
import os
import sys

# Target path
model_name = 'all-MiniLM-L6-v2'
model_path = os.path.join("models", model_name)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Check if model already exists
if os.path.isdir(model_path):
    print(f"✅ Model already exists at: {model_path}")
    sys.exit(0)

try:
    print(f"📥 Downloading '{model_name}'...")
    model = SentenceTransformer(model_name)
    model.save(model_path)
    print(f"✅ Model downloaded and saved to: {model_path}")
except Exception as e:
    print(f"❌ Error downloading model: {str(e)}")
    print("\nPlease try the following manually:")
    print(f"1. Visit: https://huggingface.co/sentence-transformers/{model_name}")
    print("2. Download the entire repo (including .bin and config files)")
    print(f"3. Extract the folder to: {model_path}")
    sys.exit(1)
