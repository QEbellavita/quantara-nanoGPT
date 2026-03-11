"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Emotion GPT API Server
===============================================================================
REST API for emotion-aware text generation and analysis.

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Emotion-Aware Training Engine
- Psychology Emotion Database
- Biometric Integration Engine
- Therapist Dashboard Engine

Endpoints:
  POST /api/emotion/generate   - Generate emotion-aware text
  POST /api/emotion/analyze    - Analyze emotional content
  POST /api/emotion/coach      - Get empathetic coaching response
  POST /api/emotion/therapy    - Get therapy technique recommendation
  GET  /api/emotion/status     - API health check
  GET  /api/emotion/emotions   - List supported emotions

Start server:
  python emotion_api_server.py --port 5050
===============================================================================
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import torch

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT, GPTConfig

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("[!] Flask not installed. Run: pip install flask flask-cors")

try:
    from emotion_classifier import MultimodalEmotionAnalyzer
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False


class EmotionGPTModel:
    """Emotion GPT Model wrapper for API"""

    EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'neutral']

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        self.device = self._detect_device(device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.encode = None
        self.decode = None
        self._load_model()

    def _detect_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        return device

    def _init_multimodal_analyzer(self):
        """Initialize multimodal analyzer if available"""
        self.multimodal_analyzer = None
        classifier_path = Path(__file__).parent / 'checkpoints' / 'emotion_fusion_head.pt'
        if HAS_MULTIMODAL and classifier_path.exists():
            try:
                self.multimodal_analyzer = MultimodalEmotionAnalyzer(
                    gpt_checkpoint=self.checkpoint_path,
                    classifier_checkpoint=str(classifier_path),
                    device=self.device
                )
                print("[EmotionGPT] Multimodal analyzer loaded")
            except Exception as e:
                print(f"[EmotionGPT] Multimodal analyzer failed: {e}")

    def _load_model(self):
        """Load trained model from checkpoint"""
        print(f"[EmotionGPT] Loading model from {self.checkpoint_path}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        # Setup tokenizer
        meta_path = Path(__file__).parent / 'data' / 'quantara_emotion' / 'meta.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi.get(c, 0) for c in s]
            self.decode = lambda l: ''.join([itos.get(i, '') for i in l])
            print("[EmotionGPT] Using character-level tokenizer")
        elif HAS_TIKTOKEN:
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: enc.decode(l)
            print("[EmotionGPT] Using GPT-2 BPE tokenizer")
        else:
            raise RuntimeError("No tokenizer available")

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[EmotionGPT] Model loaded: {param_count/1e6:.2f}M parameters on {self.device}")

        # Initialize multimodal analyzer if available
        self._init_multimodal_analyzer()

    def generate(
        self,
        prompt: str,
        emotion: str = None,
        max_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 200
    ) -> dict:
        """Generate emotion-aware text"""

        # Format prompt with emotion tag
        if emotion and emotion in self.EMOTIONS:
            formatted = f"<{emotion}>{prompt}"
        else:
            formatted = prompt

        start_ids = self.encode(formatted)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        with torch.no_grad():
            ctx = torch.amp.autocast(device_type=self.device, dtype=torch.float16) if self.device == 'cuda' else nullcontext()
            with ctx:
                y = self.model.generate(x, max_tokens, temperature=temperature, top_k=top_k)

        output = self.decode(y[0].tolist())

        # Clean up emotion tags
        if emotion and f"</{emotion}>" in output:
            output = output.split(f"</{emotion}>")[0]
            output = output.replace(f"<{emotion}>", "").strip()

        return {
            'response': output,
            'emotion': emotion,
            'prompt': prompt,
            'tokens_generated': len(y[0]) - len(start_ids)
        }

    def analyze(self, text: str, biometrics: dict = None) -> dict:
        """Analyze emotional content of text with optional biometrics."""

        # Use multimodal analyzer if available
        if self.multimodal_analyzer is not None:
            return self.multimodal_analyzer.analyze(text, biometrics)

        # Fallback to keyword-based analysis
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'blessed'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'crying', 'tears', 'grief', 'lonely'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'frustrated', 'hate', 'rage'],
            'fear': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'terrified', 'panic'],
            'love': ['love', 'adore', 'cherish', 'caring', 'affection', 'romantic'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow', 'astonished'],
        }

        text_lower = text.lower()
        scores = {}

        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[emotion] = count

        total = sum(scores.values()) or 1
        scores = {k: v / total for k, v in scores.items()}

        if max(scores.values()) < 0.3:
            scores['neutral'] = 0.5

        dominant = max(scores, key=scores.get)

        return {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'scores': scores,
            'dominant_emotion': dominant,
            'confidence': scores[dominant],
            'status': 'success'
        }

    # Fast empathetic responses for instant feedback
    FAST_RESPONSES = {
        'joy': [
            "It's wonderful to hear you're feeling joyful! This positive energy is precious - savor it.",
            "Your happiness is radiating through your words. What's bringing you this joy today?",
            "Feeling joyful is a gift. Take a moment to really embrace this feeling.",
        ],
        'sadness': [
            "I hear you, and it's okay to feel sad. Your feelings are valid and I'm here with you.",
            "Sadness can feel heavy. Remember, it's okay to take things one moment at a time.",
            "I'm sorry you're going through this. Would you like to talk more about what's on your mind?",
        ],
        'anger': [
            "I can sense your frustration. It's natural to feel angry sometimes - your feelings matter.",
            "Anger often signals that something important to us has been crossed. What's behind this feeling?",
            "Take a deep breath. Your anger is valid, and we can work through this together.",
        ],
        'fear': [
            "Feeling anxious or worried is completely understandable. You're not alone in this.",
            "Fear can be overwhelming. Let's take this one step at a time together.",
            "I hear your concerns. Sometimes naming our fears helps diminish their power.",
        ],
        'neutral': [
            "I'm here to listen. How can I support you today?",
            "Thank you for sharing. What's on your mind right now?",
            "I'm here whenever you need to talk. What would be most helpful for you?",
        ],
        'love': [
            "Love is such a beautiful emotion. It's wonderful that you're experiencing this connection.",
            "The warmth in your words is touching. Love enriches our lives in so many ways.",
            "Feeling loved and giving love are among life's greatest gifts.",
        ],
        'surprise': [
            "That sounds unexpected! How are you processing this surprise?",
            "Life can certainly catch us off guard. How are you feeling about this?",
            "Surprises can be exciting or unsettling. I'm here to help you work through it.",
        ],
    }

    def coach(
        self,
        message: str,
        emotion: str = None,
        biometric_data: dict = None,
        use_model: bool = False
    ) -> dict:
        """Generate empathetic coaching response (fast mode by default)"""
        import random

        # Auto-detect emotion if not provided
        if not emotion:
            analysis = self.analyze(message)
            emotion = analysis['dominant_emotion']

        # Get pre-written responses for this emotion
        responses = self.FAST_RESPONSES.get(emotion, self.FAST_RESPONSES['neutral'])

        # Fast mode: use pre-written empathetic responses (default)
        if not use_model:
            response = random.choice(responses)
        else:
            # Model mode: generate with reduced tokens for speed
            prompt = f"<empathy>User feels: {emotion} | Message: {message} | Response:"
            result = self.generate(prompt, max_tokens=75, temperature=0.7)
            response = result.get('response', random.choice(responses))

        # Add biometric insight if available
        biometric_insight = None
        if biometric_data:
            hr = biometric_data.get('heart_rate', 0)
            hrv = biometric_data.get('hrv', 0)

            if hr > 100:
                biometric_insight = "Your elevated heart rate suggests heightened arousal. A breathing exercise may help."
            elif hrv < 30:
                biometric_insight = "Your HRV indicates stress. Consider a moment of mindfulness."
            elif hr < 60 and emotion == 'sadness':
                biometric_insight = "Your low heart rate combined with mood may indicate low energy. Gentle movement could help."

        return {
            'response': response,
            'detected_emotion': emotion,
            'biometric_insight': biometric_insight,
            'user_message': message,
            'model': 'quantara-emotion-gpt'
        }

    def get_therapy_technique(self, emotion: str) -> dict:
        """Get therapy technique for emotion"""

        techniques = {
            'sadness': {
                'name': 'Behavioral Activation',
                'description': 'Schedule small, pleasurable activities. Even brief moments of engagement can shift mood.',
                'exercise': 'List 3 activities that brought you joy in the past. Choose one to do today, even for 5 minutes.'
            },
            'anger': {
                'name': 'Cognitive Reframing',
                'description': 'Examine the thoughts driving your anger. Are there alternative interpretations?',
                'exercise': 'Write down the situation, your automatic thought, and then 2-3 alternative perspectives.'
            },
            'fear': {
                'name': 'Grounding (5-4-3-2-1)',
                'description': 'Bring yourself to the present moment using your senses.',
                'exercise': 'Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.'
            },
            'joy': {
                'name': 'Savoring',
                'description': 'Extend positive emotions by fully engaging with the experience.',
                'exercise': 'Pause and notice all aspects of this positive moment. Share it with someone if possible.'
            },
            'love': {
                'name': 'Loving-Kindness Meditation',
                'description': 'Extend feelings of love and compassion to yourself and others.',
                'exercise': 'Repeat: "May I be happy. May I be healthy. May I be at peace." Then extend to loved ones.'
            },
            'surprise': {
                'name': 'Mindful Observation',
                'description': 'Stay present with unexpected experiences without immediate judgment.',
                'exercise': 'Notice your physical sensations and thoughts without acting. Breathe and observe.'
            },
            'neutral': {
                'name': 'Values Clarification',
                'description': 'Connect with what matters most to guide your next action.',
                'exercise': 'Ask yourself: What do I want to stand for today? What small step aligns with that?'
            }
        }

        technique = techniques.get(emotion, techniques['neutral'])

        return {
            'emotion': emotion,
            'technique': technique['name'],
            'description': technique['description'],
            'exercise': technique['exercise']
        }


def create_app(model: EmotionGPTModel) -> Flask:
    """Create Flask application with emotion API routes"""

    app = Flask(__name__)
    CORS(app)  # Enable CORS for frontend integration

    @app.route('/api/emotion/status', methods=['GET'])
    def status():
        """Health check endpoint"""
        return jsonify({
            'status': 'online',
            'model': 'quantara-emotion-gpt',
            'device': model.device,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })

    @app.route('/api/emotion/emotions', methods=['GET'])
    def list_emotions():
        """List supported emotions"""
        return jsonify({
            'emotions': model.EMOTIONS,
            'count': len(model.EMOTIONS)
        })

    @app.route('/api/emotion/generate', methods=['POST'])
    def generate():
        """Generate emotion-aware text

        Request body:
        {
            "prompt": "I feel...",
            "emotion": "joy",  // optional
            "max_tokens": 150,  // optional
            "temperature": 0.8  // optional
        }
        """
        try:
            data = request.json or {}
            prompt = data.get('prompt', '')

            if not prompt:
                return jsonify({'error': 'prompt is required'}), 400

            result = model.generate(
                prompt=prompt,
                emotion=data.get('emotion'),
                max_tokens=data.get('max_tokens', 150),
                temperature=data.get('temperature', 0.8)
            )

            return jsonify({**result, 'status': 'success'})

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/emotion/analyze', methods=['POST'])
    def analyze():
        """Analyze emotional content

        Request body:
        {
            "text": "I'm feeling great today!",
            "biometrics": {  // optional
                "heart_rate": 80,
                "hrv": 50,
                "eda": 3.0
            }
        }
        """
        try:
            data = request.json or {}
            text = data.get('text', '')

            if not text:
                return jsonify({'error': 'text is required'}), 400

            biometrics = data.get('biometrics')
            result = model.analyze(text, biometrics)

            return jsonify({**result, 'status': 'success'})

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/emotion/coach', methods=['POST'])
    def coach():
        """Get empathetic coaching response

        Request body:
        {
            "message": "I've been feeling overwhelmed",
            "emotion": "sadness",  // optional, auto-detected if missing
            "use_model": false,  // optional, use fast responses by default
            "biometric": {  // optional
                "heart_rate": 85,
                "hrv": 45
            }
        }
        """
        try:
            data = request.json or {}
            message = data.get('message', '')

            if not message:
                return jsonify({'error': 'message is required'}), 400

            result = model.coach(
                message=message,
                emotion=data.get('emotion'),
                biometric_data=data.get('biometric'),
                use_model=data.get('use_model', False)
            )

            return jsonify({**result, 'status': 'success'})

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/emotion/therapy', methods=['POST'])
    def therapy():
        """Get therapy technique recommendation

        Request body:
        {
            "emotion": "sadness"
        }
        """
        try:
            data = request.json or {}
            emotion = data.get('emotion', 'neutral')

            result = model.get_therapy_technique(emotion)

            return jsonify({**result, 'status': 'success'})

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    # Neural Ecosystem integration endpoint
    @app.route('/api/neural/emotion-workflow', methods=['POST'])
    def neural_workflow():
        """Neural Ecosystem workflow integration

        Processes emotion data for workflow triggers.
        Connected to Neural Workflow AI Engine.
        """
        try:
            data = request.json or {}

            # Analyze incoming data
            text = data.get('text', data.get('message', ''))
            biometric = data.get('biometric', {})

            # Get emotion analysis
            analysis = model.analyze(text) if text else {'dominant_emotion': 'neutral'}

            # Generate workflow recommendation
            emotion = analysis.get('dominant_emotion', 'neutral')

            workflow_actions = {
                'sadness': ['trigger_support_check', 'schedule_wellness_activity'],
                'anger': ['enable_cool_down_mode', 'suggest_break'],
                'fear': ['activate_grounding_exercise', 'notify_support_contact'],
                'joy': ['log_positive_moment', 'suggest_sharing'],
                'neutral': ['continue_normal_workflow']
            }

            return jsonify({
                'emotion_analysis': analysis,
                'biometric_data': biometric,
                'recommended_actions': workflow_actions.get(emotion, []),
                'workflow_trigger': f'emotion_{emotion}',
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            })

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description='Quantara Emotion GPT API Server')
    parser.add_argument('--checkpoint', default='out-quantara-emotion-fast/ckpt.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=5050, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--device', default='auto', help='Device (cuda, mps, cpu, auto)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    if not HAS_FLASK:
        print("[!] Flask required. Install with: pip install flask flask-cors")
        sys.exit(1)

    print("=" * 60)
    print("  QUANTARA EMOTION GPT API SERVER")
    print("=" * 60)

    # Load model
    model = EmotionGPTModel(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Create app
    app = create_app(model)

    print(f"\n  API Endpoints:")
    print(f"  - GET  /api/emotion/status")
    print(f"  - GET  /api/emotion/emotions")
    print(f"  - POST /api/emotion/generate")
    print(f"  - POST /api/emotion/analyze")
    print(f"  - POST /api/emotion/coach")
    print(f"  - POST /api/emotion/therapy")
    print(f"  - POST /api/neural/emotion-workflow")
    print(f"\n  Starting server on http://{args.host}:{args.port}")
    print("=" * 60 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
