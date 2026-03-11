"""
Download model checkpoint from GitHub releases or HuggingFace Hub
"""
import os
import urllib.request
from pathlib import Path

MODEL_URL = os.environ.get(
    'MODEL_URL',
    'https://github.com/QEbellavita/quantara-nanoGPT/releases/download/v1.0/ckpt.pt'
)
MODEL_DIR = Path('out-quantara-emotion-fast')
MODEL_PATH = MODEL_DIR / 'ckpt.pt'

def download_model():
    """Download model if not present"""
    if MODEL_PATH.exists():
        print(f"[Model] Found existing checkpoint at {MODEL_PATH}")
        return str(MODEL_PATH)

    print(f"[Model] Downloading checkpoint from {MODEL_URL}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[Model] Downloaded to {MODEL_PATH}")
        return str(MODEL_PATH)
    except Exception as e:
        print(f"[Model] Download failed: {e}")
        print("[Model] Will use sentence-transformers only (no generation)")
        return None

if __name__ == '__main__':
    download_model()
