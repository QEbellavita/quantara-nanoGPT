"""
===============================================================================
QUANTARA EMOTION API - Test Script
===============================================================================
Tests all emotion API endpoints.

Usage:
  # Start server first:
  python emotion_api_server.py --port 5050

  # Then run tests:
  python test_emotion_api.py
===============================================================================
"""

import requests
import json

BASE_URL = "http://localhost:5050"


def test_status():
    """Test status endpoint"""
    print("\n[1] Testing /api/emotion/status")
    r = requests.get(f"{BASE_URL}/api/emotion/status")
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(r.json(), indent=2)}")
    return r.status_code == 200


def test_emotions():
    """Test emotions list endpoint"""
    print("\n[2] Testing /api/emotion/emotions")
    r = requests.get(f"{BASE_URL}/api/emotion/emotions")
    print(f"    Status: {r.status_code}")
    print(f"    Emotions: {r.json().get('emotions')}")
    return r.status_code == 200


def test_analyze():
    """Test emotion analysis"""
    print("\n[3] Testing /api/emotion/analyze")

    tests = [
        "I'm so happy today! Everything is wonderful!",
        "I feel really sad and lonely right now.",
        "This makes me so angry, I can't believe it!",
        "I'm worried about what might happen tomorrow."
    ]

    for text in tests:
        r = requests.post(
            f"{BASE_URL}/api/emotion/analyze",
            json={"text": text}
        )
        result = r.json()
        print(f"    Text: '{text[:40]}...'")
        print(f"    Detected: {result.get('dominant_emotion')} ({result.get('confidence', 0):.0%})")

    return True


def test_generate():
    """Test emotion-aware generation"""
    print("\n[4] Testing /api/emotion/generate")

    r = requests.post(
        f"{BASE_URL}/api/emotion/generate",
        json={
            "prompt": "Today I feel",
            "emotion": "joy",
            "max_tokens": 50
        }
    )
    result = r.json()
    print(f"    Prompt: 'Today I feel' (emotion: joy)")
    print(f"    Generated: {result.get('response', '')[:100]}...")
    return r.status_code == 200


def test_coach():
    """Test coaching response"""
    print("\n[5] Testing /api/emotion/coach")

    r = requests.post(
        f"{BASE_URL}/api/emotion/coach",
        json={
            "message": "I've been feeling overwhelmed with work lately",
            "biometric": {
                "heart_rate": 92,
                "hrv": 35
            }
        }
    )
    result = r.json()
    print(f"    Message: 'I've been feeling overwhelmed with work lately'")
    print(f"    Detected emotion: {result.get('detected_emotion')}")
    print(f"    Biometric insight: {result.get('biometric_insight')}")
    print(f"    Response: {result.get('response', '')[:150]}...")
    return r.status_code == 200


def test_therapy():
    """Test therapy technique"""
    print("\n[6] Testing /api/emotion/therapy")

    for emotion in ['sadness', 'anger', 'fear']:
        r = requests.post(
            f"{BASE_URL}/api/emotion/therapy",
            json={"emotion": emotion}
        )
        result = r.json()
        print(f"    {emotion.upper()}: {result.get('technique')}")

    return True


def test_neural_workflow():
    """Test Neural Ecosystem integration"""
    print("\n[7] Testing /api/neural/emotion-workflow")

    r = requests.post(
        f"{BASE_URL}/api/neural/emotion-workflow",
        json={
            "text": "I'm feeling stressed and anxious about the deadline",
            "biometric": {
                "heart_rate": 95,
                "hrv": 28
            }
        }
    )
    result = r.json()
    print(f"    Emotion: {result.get('emotion_analysis', {}).get('dominant_emotion')}")
    print(f"    Workflow trigger: {result.get('workflow_trigger')}")
    print(f"    Actions: {result.get('recommended_actions')}")
    return r.status_code == 200


def main():
    print("=" * 60)
    print("  QUANTARA EMOTION API - Test Suite")
    print("=" * 60)

    try:
        # Check if server is running
        r = requests.get(f"{BASE_URL}/api/emotion/status", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"\n[!] Cannot connect to {BASE_URL}")
        print("[!] Start the server first:")
        print("    python emotion_api_server.py --port 5050")
        return

    tests = [
        ("Status", test_status),
        ("Emotions", test_emotions),
        ("Analyze", test_analyze),
        ("Generate", test_generate),
        ("Coach", test_coach),
        ("Therapy", test_therapy),
        ("Neural Workflow", test_neural_workflow),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"    [!] Error: {e}")

    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
