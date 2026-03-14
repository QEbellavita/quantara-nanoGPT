"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - nanoGPT Integration
===============================================================================
Connects trained emotion models to the Quantara ecosystem.

Integration Points:
- Neural Workflow AI Engine
- AI Conversational Coach
- Emotion-Aware Training Engine
- Psychology Emotion Database
- Biometric Integration Engine
- Real-time Dashboard Data

API Endpoints:
- /api/emotion/generate - Generate emotion-aware text
- /api/emotion/analyze - Analyze emotional content
- /api/emotion/coach - Get coaching response based on emotion
===============================================================================
"""

import os
import pickle
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import tiktoken

# nanoGPT imports
from model import GPT, GPTConfig

class QuantaraEmotionGPT:
    """
    Quantara Emotion GPT Model Interface

    Connects to:
    - Neural Workflow AI Engine
    - ML Training & Prediction Systems
    - Backend APIs (cases, workflows, analytics)
    - Dashboard Data Integration
    - Real-time customer service data
    """

    def __init__(
        self,
        checkpoint_path: str = 'out-quantara-emotion/ckpt.pt',
        device: str = 'auto',
        seed: int = 42
    ):
        self.device = self._detect_device(device)
        self.seed = seed

        # Load model
        self._load_model(checkpoint_path)

        # 32-Emotion taxonomy (9 families)
        self.emotion_families = {
            'Joy': ['joy', 'excitement', 'enthusiasm', 'fun', 'gratitude', 'pride'],
            'Sadness': ['sadness', 'grief', 'boredom', 'nostalgia'],
            'Anger': ['anger', 'frustration', 'hate', 'contempt', 'disgust', 'jealousy'],
            'Fear': ['fear', 'anxiety', 'worry', 'overwhelmed', 'stressed'],
            'Love': ['love', 'compassion'],
            'Calm': ['calm', 'relief', 'mindfulness', 'resilience', 'hope'],
            'Self-Conscious': ['guilt', 'shame'],
            'Surprise': ['surprise'],
            'Neutral': ['neutral'],
        }
        self.emotions = [e for ems in self.emotion_families.values() for e in ems]
        self._emotion_to_family = {}
        for fam, ems in self.emotion_families.items():
            for e in ems:
                self._emotion_to_family[e] = fam

        print(f"[Quantara] Emotion GPT loaded on {self.device}")

    def _detect_device(self, device: str) -> str:
        """Auto-detect best available device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device

    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        torch.manual_seed(self.seed)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Initialize model
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf)

        # Load weights
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        # Load tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: self.enc.encode(s, allowed_special={"<|endoftext|>"})
        self.decode = lambda l: self.enc.decode(l)

        # Check for char-level meta
        meta_path = Path('data/quantara_emotion/meta.pkl')
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.stoi = meta['stoi']
            self.itos = meta['itos']
            self.encode = lambda s: [self.stoi[c] for c in s]
            self.decode = lambda l: ''.join([self.itos[i] for i in l])

    def generate(
        self,
        prompt: str,
        emotion: str = None,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 200
    ) -> str:
        """
        Generate emotion-aware text response.

        Connected to:
        - AI Conversational Coach
        - Neural Workflow AI Engine

        Args:
            prompt: Input text/question
            emotion: Target emotion (sadness, joy, anger, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_k: Top-k sampling

        Returns:
            Generated response text
        """
        # Format prompt with emotion tag
        if emotion:
            formatted = f"<{emotion}>{prompt}"
        else:
            formatted = prompt

        # Encode
        start_ids = self.encode(formatted)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        # Generate
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16) if self.device == 'cuda' else nullcontext():
                y = self.model.generate(x, max_tokens, temperature=temperature, top_k=top_k)

        # Decode
        response = self.decode(y[0].tolist())

        # Clean up emotion tags if present
        if emotion and f"</{emotion}>" in response:
            response = response.split(f"</{emotion}>")[0]
            response = response.replace(f"<{emotion}>", "")

        return response.strip()

    def analyze_emotion(self, text: str) -> dict:
        """
        Analyze emotional content of text.

        Connected to:
        - Emotion-Aware Training Engine
        - Psychology Emotion Database
        - Biometric Integration (cross-reference)

        Returns:
            Dict with emotion scores and dominant emotion
        """
        # Generate completion for each emotion category
        scores = {}

        for emotion in self.emotions:
            prompt = f"<{emotion}>{text}</{emotion}>"
            start_ids = self.encode(prompt)
            x = torch.tensor(start_ids[:100], dtype=torch.long, device=self.device)[None, ...]

            with torch.no_grad():
                logits, _ = self.model(x)
                # Use mean logit as rough score
                scores[emotion] = logits.mean().item()

        # Normalize to probabilities
        total = sum(abs(s) for s in scores.values())
        if total > 0:
            scores = {k: abs(v) / total for k, v in scores.items()}

        # Find dominant emotion
        dominant = max(scores, key=scores.get)

        family = self._emotion_to_family.get(dominant, 'Neutral')

        return {
            'scores': scores,
            'dominant_emotion': dominant,
            'family': family,
            'confidence': scores[dominant],
            'text': text[:100] + '...' if len(text) > 100 else text
        }

    def get_coaching_response(
        self,
        user_message: str,
        detected_emotion: str = None,
        biometric_data: dict = None
    ) -> dict:
        """
        Generate empathetic coaching response.

        Connected to:
        - AI Conversational Coach
        - Neural Workflow AI Engine
        - Biometric Integration Engine
        - Real-time Dashboard Data

        Args:
            user_message: User's input message
            detected_emotion: Pre-detected emotion (or auto-detect)
            biometric_data: Optional biometric context (HR, HRV, etc.)

        Returns:
            Coaching response with emotion context
        """
        # Auto-detect emotion if not provided
        if not detected_emotion:
            analysis = self.analyze_emotion(user_message)
            detected_emotion = analysis['dominant_emotion']

        # Generate empathetic response
        prompt = f"<empathy>User feels: {detected_emotion} | Message: {user_message} | Response:"
        response = self.generate(prompt, max_tokens=200, temperature=0.7)

        # Add biometric context if available
        biometric_insight = None
        if biometric_data:
            hr = biometric_data.get('heart_rate', 0)
            hrv = biometric_data.get('hrv', 0)

            if hr > 100:
                biometric_insight = "Your elevated heart rate suggests heightened arousal. Let's try a breathing exercise."
            elif hrv < 30:
                biometric_insight = "Your HRV indicates some stress. A moment of mindfulness could help."

        return {
            'response': response,
            'detected_emotion': detected_emotion,
            'biometric_insight': biometric_insight,
            'user_message': user_message,
            'model': 'quantara-emotion-gpt'
        }

    def get_therapy_technique(self, emotion: str) -> str:
        """
        Get appropriate therapy technique for emotion.

        Connected to:
        - Therapist Dashboard Engine
        - Psychology Emotion Database
        """
        prompt = f"<therapy>For {emotion}, the recommended technique is:"
        return self.generate(prompt, max_tokens=150, temperature=0.6)


# Flask API integration (for neural ecosystem backend)
def create_api_routes(app, model: QuantaraEmotionGPT):
    """
    Create Flask API routes for Quantara integration.

    Endpoints:
    - POST /api/emotion/generate
    - POST /api/emotion/analyze
    - POST /api/emotion/coach
    """
    from flask import request, jsonify

    @app.route('/api/emotion/generate', methods=['POST'])
    def api_generate():
        data = request.json
        result = model.generate(
            prompt=data.get('prompt', ''),
            emotion=data.get('emotion'),
            max_tokens=data.get('max_tokens', 256),
            temperature=data.get('temperature', 0.8)
        )
        return jsonify({'response': result, 'status': 'success'})

    @app.route('/api/emotion/analyze', methods=['POST'])
    def api_analyze():
        data = request.json
        result = model.analyze_emotion(data.get('text', ''))
        return jsonify({**result, 'status': 'success'})

    @app.route('/api/emotion/coach', methods=['POST'])
    def api_coach():
        data = request.json
        result = model.get_coaching_response(
            user_message=data.get('message', ''),
            detected_emotion=data.get('emotion'),
            biometric_data=data.get('biometric')
        )
        return jsonify({**result, 'status': 'success'})

    return app


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Quantara Emotion GPT')
    parser.add_argument('--checkpoint', default='out-quantara-emotion/ckpt.pt', help='Model checkpoint path')
    parser.add_argument('--device', default='auto', help='Device (cuda, mps, cpu, auto)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    args = parser.parse_args()

    # Load model
    model = QuantaraEmotionGPT(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    if args.interactive:
        print("\n" + "=" * 60)
        print("  QUANTARA EMOTION GPT - Interactive Mode")
        print("=" * 60)
        print("  Commands: /emotion <name>, /analyze, /coach, /quit")
        print("=" * 60 + "\n")

        current_emotion = None

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == '/quit':
                break
            elif user_input.startswith('/emotion '):
                current_emotion = user_input.split(' ')[1]
                print(f"  [Emotion set to: {current_emotion}]")
                continue
            elif user_input == '/analyze':
                text = input("  Text to analyze: ")
                result = model.analyze_emotion(text)
                print(f"  Dominant: {result['dominant_emotion']} ({result['confidence']:.2%})")
                continue
            elif user_input == '/coach':
                message = input("  Your message: ")
                result = model.get_coaching_response(message)
                print(f"\n  Coach: {result['response']}")
                print(f"  [Detected: {result['detected_emotion']}]\n")
                continue

            # Generate response
            response = model.generate(user_input, emotion=current_emotion)
            print(f"\nAssistant: {response}\n")

    else:
        # Demo
        print("\n[Demo] Generating emotion-aware response...")
        response = model.generate("I've been feeling overwhelmed lately", emotion="empathy")
        print(f"Response: {response}")
