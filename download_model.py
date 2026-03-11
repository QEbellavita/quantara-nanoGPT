"""
Download model checkpoint from GitHub releases (supports private repos)
"""
import os
import urllib.request
import subprocess
from pathlib import Path

# Use GitHub API for private repo access
GITHUB_REPO = 'QEbellavita/quantara-nanoGPT'
RELEASE_TAG = 'v1.0.0'
ASSET_NAME = 'ckpt.pt'

MODEL_DIR = Path('out-quantara-emotion-fast')
MODEL_PATH = MODEL_DIR / 'ckpt.pt'

def download_with_gh_cli():
    """Download using gh CLI (handles authentication)"""
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            'gh', 'release', 'download', RELEASE_TAG,
            '--repo', GITHUB_REPO,
            '--pattern', ASSET_NAME,
            '--dir', str(MODEL_DIR),
            '--clobber'
        ], capture_output=True, text=True, timeout=600)

        if result.returncode == 0 and MODEL_PATH.exists():
            print(f"[Model] Downloaded via gh CLI to {MODEL_PATH}")
            return str(MODEL_PATH)
        else:
            print(f"[Model] gh CLI failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"[Model] gh CLI error: {e}")
        return None

def download_with_token():
    """Download using GITHUB_TOKEN environment variable"""
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not token:
        print("[Model] No GITHUB_TOKEN found")
        return None

    try:
        import requests
    except ImportError:
        print("[Model] requests not installed, trying urllib with token")
        return download_with_urllib_token(token)

    # Get release assets
    api_url = f'https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{RELEASE_TAG}'
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        resp = requests.get(api_url, headers=headers)
        resp.raise_for_status()
        release = resp.json()

        # Find asset
        asset = next((a for a in release.get('assets', []) if a['name'] == ASSET_NAME), None)
        if not asset:
            print(f"[Model] Asset {ASSET_NAME} not found in release")
            return None

        # Download asset
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        download_headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/octet-stream'
        }

        print(f"[Model] Downloading {ASSET_NAME} ({asset['size'] / 1024 / 1024:.1f} MB)...")
        with requests.get(asset['url'], headers=download_headers, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"[Model] Downloaded to {MODEL_PATH}")
        return str(MODEL_PATH)
    except Exception as e:
        print(f"[Model] Token download failed: {e}")
        return None

def download_with_urllib_token(token):
    """Fallback: download with urllib and token"""
    api_url = f'https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{RELEASE_TAG}'

    try:
        req = urllib.request.Request(api_url, headers={
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        })
        with urllib.request.urlopen(req) as resp:
            import json
            release = json.loads(resp.read())

        asset = next((a for a in release.get('assets', []) if a['name'] == ASSET_NAME), None)
        if not asset:
            return None

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        download_req = urllib.request.Request(asset['url'], headers={
            'Authorization': f'token {token}',
            'Accept': 'application/octet-stream'
        })

        print(f"[Model] Downloading {ASSET_NAME}...")
        with urllib.request.urlopen(download_req) as resp:
            with open(MODEL_PATH, 'wb') as f:
                f.write(resp.read())

        print(f"[Model] Downloaded to {MODEL_PATH}")
        return str(MODEL_PATH)
    except Exception as e:
        print(f"[Model] urllib token download failed: {e}")
        return None

def download_model():
    """Download model if not present"""
    if MODEL_PATH.exists():
        print(f"[Model] Found existing checkpoint at {MODEL_PATH}")
        return str(MODEL_PATH)

    print(f"[Model] Checkpoint not found, attempting download...")

    # Try gh CLI first (best for private repos)
    result = download_with_gh_cli()
    if result:
        return result

    # Try with GitHub token
    result = download_with_token()
    if result:
        return result

    print("[Model] All download methods failed")
    print("[Model] Will use sentence-transformers only (no generation)")
    return None

if __name__ == '__main__':
    download_model()
