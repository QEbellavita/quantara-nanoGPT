"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Emotion GPT API Server (32-Emotion Taxonomy)
===============================================================================
REST API for emotion-aware text generation and analysis.
Supports 32 emotions across 9 families with hierarchical classification.

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Emotion-Aware Training Engine
- Psychology Emotion Database
- Biometric Integration Engine
- Therapist Dashboard Engine

Endpoints:
  POST /api/emotion/generate        - Generate emotion-aware text
  POST /api/emotion/analyze         - Analyze emotional content
  POST /api/emotion/coach           - Get empathetic coaching response
  POST /api/emotion/therapy         - Get therapy technique recommendation
  GET  /api/emotion/status          - API health check
  GET  /api/emotion/emotions        - List supported emotions
  GET  /api/emotion/family          - List all families with emotions
  GET  /api/emotion/family/<name>   - Get emotions in a specific family
  POST /api/neural/emotion-workflow - Neural workflow integration
  POST /api/neural/emotion-transition - Emotion transition pathway
  GET  /api/ruview/status            - RuView WiFi sensing status
  POST /api/ruview/connect           - Connect to RuView service
  GET  /api/ruview/biometrics        - Get WiFi-sensed biometrics
  GET  /api/ruview/presence          - Get presence/occupancy data
  POST /api/ruview/analyze           - Emotion analysis with WiFi biometrics
  POST /api/emotion/transition/record              - Record emotion for tracking
  GET  /api/emotion/transition/trajectory/<user_id> - Get emotion timeline
  GET  /api/emotion/transition/patterns/<user_id>   - Detect concerning patterns
  GET  /api/emotion/transition/dashboard/<user_id>  - Therapist dashboard summary

Start server:
  python emotion_api_server.py --port 5050
===============================================================================
"""

import os
import sys
import json
import logging
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

logger = logging.getLogger(__name__)

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
    from emotion_classifier import MultimodalEmotionAnalyzer, EMOTION_FAMILIES, FAMILY_NAMES, apply_profile_personalization
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False

from emotion_gpt import EmotionGPT

from metrics_collector import MetricsCollector

try:
    from external_context import ExternalContextProvider
    HAS_EXTERNAL_CONTEXT = True
except ImportError:
    HAS_EXTERNAL_CONTEXT = False

try:
    from emotion_transition_engine import EmotionTransitionEngine
    HAS_TRANSITION_ENGINE = True
except ImportError:
    HAS_TRANSITION_ENGINE = False

try:
    from emotion_websocket import init_websocket, emit_emotion_update as ws_emit_emotion
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

try:
    from auto_retrain import AutoRetrainManager
    HAS_AUTO_RETRAIN = True
except ImportError:
    HAS_AUTO_RETRAIN = False

try:
    from emotion_transition_tracker import EmotionTransitionTracker
    HAS_TRANSITION_TRACKER = True
except ImportError:
    HAS_TRANSITION_TRACKER = False

# Profile engine integration
try:
    from user_profile_engine import UserProfileEngine
    from profile_api import create_profile_blueprint
    HAS_PROFILE_ENGINE = True
except ImportError:
    HAS_PROFILE_ENGINE = False

profile_engine = None


# ─── 32-Emotion Taxonomy ────────────────────────────────────────────────────

_EMOTION_FAMILIES = {
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

_EMOTION_TO_FAMILY = {}
for _fam, _ems in _EMOTION_FAMILIES.items():
    for _e in _ems:
        _EMOTION_TO_FAMILY[_e] = _fam


# ─── Therapy Techniques (all 32 emotions) ────────────────────────────────────

THERAPY_TECHNIQUES = {
    # Joy family
    'joy': {
        'name': 'Savoring',
        'description': 'Extend positive emotions by fully engaging with the experience.',
        'exercise': 'Pause and notice all aspects of this positive moment. Share it with someone if possible.',
    },
    'excitement': {
        'name': 'Mindful Savoring',
        'description': 'Channel high energy into focused awareness of the positive experience.',
        'exercise': 'Notice the physical sensations of excitement. What specifically sparked this? Anchor it.',
    },
    'enthusiasm': {
        'name': 'Goal Anchoring',
        'description': 'Connect motivated energy to deeper values and long-term goals.',
        'exercise': 'Write down what you\'re enthusiastic about and connect it to one core value.',
    },
    'fun': {
        'name': 'Positive Journaling',
        'description': 'Capture playful moments to build a reservoir of positive memories.',
        'exercise': 'Write about this moment of fun. What made it special? How can you create more of this?',
    },
    'gratitude': {
        'name': 'Gratitude Letter',
        'description': 'Deepen appreciation through structured expression.',
        'exercise': 'Write a letter to someone who made a difference. Be specific about what they did and its impact.',
    },
    'pride': {
        'name': 'Achievement Integration',
        'description': 'Integrate accomplishments into your self-concept for lasting confidence.',
        'exercise': 'Name 3 strengths that led to this achievement. How can you apply them next?',
    },
    # Sadness family
    'sadness': {
        'name': 'Behavioral Activation',
        'description': 'Schedule small, pleasurable activities. Even brief moments of engagement can shift mood.',
        'exercise': 'List 3 activities that brought you joy in the past. Choose one to do today, even for 5 minutes.',
    },
    'grief': {
        'name': 'Grief Journaling + Continuing Bonds',
        'description': 'Honor loss through written expression and maintaining meaningful connections to what was lost.',
        'exercise': 'Write a letter to what/who you\'ve lost. Share a favorite memory and what it meant to you.',
    },
    'boredom': {
        'name': 'Values Exploration',
        'description': 'Use boredom as a signal to reconnect with what matters.',
        'exercise': 'Ask: What would feel meaningful right now? Set one micro-goal for the next 30 minutes.',
    },
    'nostalgia': {
        'name': 'Narrative Integration',
        'description': 'Bridge past positive experiences with present meaning.',
        'exercise': 'Share a cherished memory. What quality from that time can you bring into today?',
    },
    # Anger family
    'anger': {
        'name': 'Cognitive Reframing',
        'description': 'Examine the thoughts driving your anger. Are there alternative interpretations?',
        'exercise': 'Write down the situation, your automatic thought, and then 2-3 alternative perspectives.',
    },
    'frustration': {
        'name': 'Problem-Solving Therapy',
        'description': 'Channel frustration into structured problem-solving.',
        'exercise': 'List what you can and can\'t control. Pick one controllable item and write 3 action steps.',
    },
    'hate': {
        'name': 'Cognitive Defusion (ACT)',
        'description': 'Create distance between yourself and intense negative thoughts.',
        'exercise': 'Notice the thought without acting on it. Say "I\'m having the thought that..." and observe it pass.',
    },
    'contempt': {
        'name': 'Empathy Building',
        'description': 'Develop understanding through perspective-taking exercises.',
        'exercise': 'Imagine 3 possible reasons for the behavior you find contemptible. Which feels most human?',
    },
    'disgust': {
        'name': 'Exposure Hierarchy',
        'description': 'Gradually increase tolerance through structured exposure.',
        'exercise': 'Rate the intensity 1-10. What\'s a 2/10 version you could sit with for 2 minutes?',
    },
    'jealousy': {
        'name': 'Self-Worth Inventory',
        'description': 'Redirect comparison energy toward self-appreciation.',
        'exercise': 'List 5 things you\'ve accomplished this year. Compare yourself to who you were 12 months ago.',
    },
    # Fear family
    'fear': {
        'name': 'Grounding (5-4-3-2-1)',
        'description': 'Bring yourself to the present moment using your senses.',
        'exercise': 'Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.',
    },
    'anxiety': {
        'name': 'Progressive Muscle Relaxation',
        'description': 'Release anxiety through systematic body relaxation.',
        'exercise': 'Tense each muscle group for 5 seconds, then release. Start with feet, work up to head.',
    },
    'worry': {
        'name': 'Worry Time Scheduling',
        'description': 'Contain worry by giving it a structured boundary.',
        'exercise': 'Schedule 15 minutes of "worry time." Outside that window, write worries down and postpone them.',
    },
    'overwhelmed': {
        'name': 'Task Chunking + Triage',
        'description': 'Break overwhelm into manageable pieces.',
        'exercise': 'Write everything down. Circle the ONE thing that matters most. Do just that for 5 minutes.',
    },
    'stressed': {
        'name': 'Box Breathing (4-4-4-4)',
        'description': 'Reset your autonomic nervous system through controlled breathing.',
        'exercise': 'Breathe in for 4 counts, hold 4, out 4, hold 4. Repeat 4 cycles.',
    },
    # Love family
    'love': {
        'name': 'Loving-Kindness Meditation',
        'description': 'Extend feelings of love and compassion to yourself and others.',
        'exercise': 'Repeat: "May I be happy. May I be healthy. May I be at peace." Then extend to loved ones.',
    },
    'compassion': {
        'name': 'Compassion-Focused Therapy',
        'description': 'Deepen empathetic caring through structured practice.',
        'exercise': 'Place hand on heart. Breathe slowly. Send warmth to someone who is suffering. Notice how it feels.',
    },
    # Calm family
    'calm': {
        'name': 'Body Scan Meditation',
        'description': 'Deepen calm awareness through systematic body attention.',
        'exercise': 'Starting at your head, slowly scan down to your toes. Notice sensations without changing them.',
    },
    'relief': {
        'name': 'Integration Practice',
        'description': 'Acknowledge and integrate the shift from difficulty to ease.',
        'exercise': 'Notice what helped you get through. Name it. This is a resource you can draw on again.',
    },
    'mindfulness': {
        'name': 'Open Monitoring Meditation',
        'description': 'Expand awareness to include all experience without attachment.',
        'exercise': 'Sit quietly. Notice whatever arises — thoughts, feelings, sounds — without following any of them.',
    },
    'resilience': {
        'name': 'Strength Spotting',
        'description': 'Identify and reinforce personal strengths through past-success reflection.',
        'exercise': 'Name a time you overcame something hard. What strength did you use? How can you apply it now?',
    },
    'hope': {
        'name': 'Future Self Visualization',
        'description': 'Strengthen hope through concrete future imagery.',
        'exercise': 'Close your eyes. Picture your life 6 months from now at its best. What one step gets you there?',
    },
    # Self-Conscious family
    'guilt': {
        'name': 'Self-Compassion Practice (Neff)',
        'description': 'Balance accountability with self-kindness.',
        'exercise': 'Acknowledge what happened. Ask: What would making it right look like? Then: What kindness do I need?',
    },
    'shame': {
        'name': 'Shame Resilience (Brene Brown)',
        'description': 'Separate behavior from identity to rebuild self-worth.',
        'exercise': 'Name the shame story. Ask: Is this about what I did, or who I am? Challenge the identity narrative.',
    },
    # Atomic
    'surprise': {
        'name': 'Mindful Observation',
        'description': 'Stay present with unexpected experiences without immediate judgment.',
        'exercise': 'Notice your physical sensations and thoughts without acting. Breathe and observe.',
    },
    'neutral': {
        'name': 'Values Clarification',
        'description': 'Connect with what matters most to guide your next action.',
        'exercise': 'Ask yourself: What do I want to stand for today? What small step aligns with that?',
    },
}


# ─── Transition Pathways ─────────────────────────────────────────────────────

TRANSITION_PATHWAYS = {
    'joy': 'joy → sustained wellbeing via gratitude practice',
    'excitement': 'excitement → focused calm via grounding',
    'enthusiasm': 'enthusiasm → sustained motivation via values alignment',
    'fun': 'fun → gratitude via reflection',
    'gratitude': 'gratitude → compassion via loving-kindness',
    'pride': 'pride → resilience via strength inventory',
    'sadness': 'sadness → acceptance via behavioral activation',
    'grief': 'grief → acceptance via meaning-making',
    'boredom': 'boredom → engagement via micro-goals',
    'nostalgia': 'nostalgia → present gratitude via bridging past-present',
    'anger': 'anger → understanding via perspective-taking',
    'frustration': 'frustration → agency via action planning',
    'hate': 'hate → understanding via perspective-taking',
    'contempt': 'contempt → curiosity via perspective shift',
    'disgust': 'disgust → tolerance via graduated exposure',
    'jealousy': 'jealousy → self-acceptance via comparing to past self',
    'fear': 'fear → calm via grounding',
    'anxiety': 'anxiety → calm via body-down regulation',
    'worry': 'worry → problem-solving via containment',
    'overwhelmed': 'overwhelmed → manageable via smallest-next-step',
    'stressed': 'stressed → regulated via autonomic reset',
    'love': 'love → deepened connection via appreciation expression',
    'compassion': 'compassion → action via service',
    'calm': 'calm → mindfulness via deepening awareness',
    'relief': 'relief → gratitude via acknowledging the shift',
    'mindfulness': 'mindfulness → insight via non-reactive observation',
    'resilience': 'resilience → hope via past-success recall',
    'hope': 'hope → motivation via concrete next steps',
    'guilt': 'guilt → repair via amends planning',
    'shame': 'shame → self-worth via separating action from identity',
    'surprise': 'surprise → curiosity via mindful observation',
    'neutral': 'neutral → engagement via values clarification',
}


# ─── Coaching Prompts ─────────────────────────────────────────────────────────

COACHING_PROMPTS = {
    'joy': "It's wonderful to feel this way. What's contributing to this positive feeling?",
    'excitement': "That energy is powerful. Let's channel it — what's the one thing you most want to direct this toward?",
    'enthusiasm': "I love your drive. Let's connect it to your deeper goals so it carries you forward.",
    'fun': "This joy matters. What about this moment would you want to remember?",
    'gratitude': "That appreciation you feel — have you told them? Sometimes expressing it multiplies it.",
    'pride': "You earned this. Let's name the strengths that got you here — they'll serve you again.",
    'sadness': "I hear you, and it's okay to feel sad. Your feelings are valid and I'm here with you.",
    'grief': "There's no timeline for this. What you're feeling honors what you lost. I'm here.",
    'boredom': "Boredom is often a signal. What would feel meaningful to you right now, even something small?",
    'nostalgia': "Those memories shaped who you are. What from that time do you still carry with you today?",
    'anger': "I can sense your frustration. It's natural to feel angry — your feelings matter.",
    'frustration': "I hear the frustration. Let's separate what you can control from what you can't — and start there.",
    'hate': "That intensity is telling you something matters deeply. What need is underneath this?",
    'contempt': "What if there's a reason you haven't considered? Let's explore what might be driving their behavior.",
    'disgust': "That reaction is your boundaries speaking. Let's understand what's being violated and how to protect it.",
    'jealousy': "That comparison is stealing your peace. What have you accomplished that you're not giving yourself credit for?",
    'fear': "Feeling anxious or worried is completely understandable. You're not alone in this.",
    'anxiety': "Your mind is racing ahead. Let's bring you back to right now — what do you feel in your body?",
    'worry': "Let's give this worry a boundary. What's the actual worst case, and how likely is it really?",
    'overwhelmed': "Everything at once is too much. What's the one smallest thing you could do in the next 5 minutes?",
    'stressed': "Your system is running hot. Let's cool it down — breathe with me: in 4, hold 4, out 4, hold 4.",
    'love': "Love is such a beautiful emotion. It's wonderful that you're experiencing this connection.",
    'compassion': "Your empathy is a gift. How can we channel this caring into something meaningful?",
    'calm': "You're in a good space. Let's deepen it — notice how your body feels right now, without changing anything.",
    'relief': "You made it through. Take a moment to notice — what helped you get here?",
    'mindfulness': "Stay with this awareness. Whatever arises — thoughts, feelings — just notice without following.",
    'resilience': "You've weathered storms before. What got you through last time? That strength is still in you.",
    'hope': "Hold onto that vision. Now let's build a bridge to it — what's one step you can take this week?",
    'guilt': "Guilt means your values are intact. What would making it right look like?",
    'shame': "What happened doesn't define who you are. Can you tell me what you're telling yourself about this?",
    'surprise': "That sounds unexpected! How are you processing this surprise?",
    'neutral': "I'm here to listen. How can I support you today?",
}


class EmotionGPTModel:
    """Emotion GPT Model wrapper for API — 32-emotion taxonomy"""

    EMOTIONS = [
        # Joy family
        'joy', 'excitement', 'enthusiasm', 'fun', 'gratitude', 'pride',
        # Sadness family
        'sadness', 'grief', 'boredom', 'nostalgia',
        # Anger family
        'anger', 'frustration', 'hate', 'contempt', 'disgust', 'jealousy',
        # Fear family
        'fear', 'anxiety', 'worry', 'overwhelmed', 'stressed',
        # Love family
        'love', 'compassion',
        # Calm family
        'calm', 'relief', 'mindfulness', 'resilience', 'hope',
        # Self-Conscious family
        'guilt', 'shame',
        # Atomic
        'surprise', 'neutral',
    ]

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
        """Initialize multimodal analyzer if available.

        Auto-detects GoEmotions checkpoints and uses the best available encoder:
        GoEmotions RoBERTa (796-dim) > sentence-transformers (384-dim) > nanoGPT.
        """
        self.multimodal_analyzer = None
        classifier_path = Path(__file__).parent / 'checkpoints' / 'emotion_fusion_head.pt'
        if HAS_MULTIMODAL and classifier_path.exists():
            try:
                # Auto-detect if checkpoint was trained with GoEmotions
                use_go_emotions = False
                go_combined = True
                state = torch.load(classifier_path, map_location='cpu', weights_only=False)
                if state.get('embedding_type') == 'go-emotions':
                    use_go_emotions = True
                    go_combined = state.get('go_emotions_combined', True)
                    print("[EmotionGPT] Detected GoEmotions checkpoint")

                self.multimodal_analyzer = MultimodalEmotionAnalyzer(
                    gpt_checkpoint=self.checkpoint_path,
                    classifier_checkpoint=str(classifier_path),
                    device=self.device,
                    use_go_emotions=use_go_emotions,
                    go_emotions_combined=go_combined,
                )
                encoder_type = 'GoEmotions' if use_go_emotions else 'sentence-transformers'
                print(f"[EmotionGPT] Multimodal analyzer loaded (32 emotions, {encoder_type})")
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
            'family': _EMOTION_TO_FAMILY.get(emotion, None),
            'prompt': prompt,
            'tokens_generated': len(y[0]) - len(start_ids)
        }

    def analyze(self, text: str, biometrics: dict = None) -> dict:
        """Analyze emotional content of text with optional biometrics."""

        # Use multimodal analyzer if available (returns family + is_fallback)
        if self.multimodal_analyzer is not None:
            return self.multimodal_analyzer.analyze(text, biometrics)

        # Fallback to keyword-based analysis (expanded for 32 emotions)
        emotion_keywords = {
            'joy': ['happy', 'joy', 'great', 'wonderful', 'amazing', 'blessed'],
            'excitement': ['excited', 'thrilled', 'pumped', 'stoked', 'hyped'],
            'enthusiasm': ['enthusiastic', 'motivated', 'eager', 'passionate'],
            'fun': ['fun', 'playful', 'hilarious', 'entertaining', 'amusing'],
            'gratitude': ['grateful', 'thankful', 'appreciate', 'blessed', 'thanks'],
            'pride': ['proud', 'accomplished', 'achieved', 'earned', 'succeeded'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'crying'],
            'grief': ['lost', 'died', 'death', 'mourning', 'gone forever'],
            'boredom': ['bored', 'boring', 'dull', 'tedious', 'monotonous'],
            'nostalgia': ['remember', 'miss', 'used to', 'back then', 'memories'],
            'anger': ['angry', 'furious', 'mad', 'rage', 'outraged'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'ugh'],
            'hate': ['hate', 'despise', 'loathe', 'detest', 'abhor'],
            'contempt': ['pathetic', 'beneath', 'worthless', 'look down'],
            'disgust': ['disgusting', 'gross', 'revolting', 'nauseating'],
            'jealousy': ['jealous', 'envious', 'envy', 'unfair'],
            'fear': ['scared', 'afraid', 'terrified', 'panic'],
            'anxiety': ['anxious', 'nervous', 'worried', 'uneasy'],
            'worry': ['worried', 'concerning', 'what if', 'might happen'],
            'overwhelmed': ['overwhelmed', 'too much', 'can\'t handle', 'drowning'],
            'stressed': ['stressed', 'pressure', 'tension', 'burnout'],
            'love': ['love', 'adore', 'cherish', 'affection', 'romantic'],
            'compassion': ['compassion', 'empathy', 'caring', 'sympathy'],
            'calm': ['calm', 'peaceful', 'serene', 'relaxed', 'tranquil'],
            'relief': ['relieved', 'relief', 'finally', 'weight off'],
            'mindfulness': ['mindful', 'present', 'aware', 'conscious'],
            'resilience': ['resilient', 'strong', 'persevere', 'overcome'],
            'hope': ['hope', 'hopeful', 'optimistic', 'looking forward'],
            'guilt': ['guilty', 'regret', 'my fault', 'sorry', 'apologize'],
            'shame': ['ashamed', 'humiliated', 'embarrassed', 'disgrace'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow'],
        }

        text_lower = text.lower()
        scores = {}

        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[emotion] = count

        total = sum(scores.values()) or 1
        scores = {k: v / total for k, v in scores.items()}

        if max(scores.values()) < 0.1:
            scores['neutral'] = 0.5

        dominant = max(scores, key=scores.get)
        family = _EMOTION_TO_FAMILY.get(dominant, 'Neutral')

        return {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'scores': scores,
            'dominant_emotion': dominant,
            'family': family,
            'confidence': scores[dominant],
            'is_fallback': False,
            'status': 'success'
        }

    # Fast empathetic responses for all 32 emotions
    FAST_RESPONSES = {
        # Joy family
        'joy': [
            "It's wonderful to hear you're feeling joyful! This positive energy is precious - savor it.",
            "Your happiness is radiating through your words. What's bringing you this joy today?",
        ],
        'excitement': [
            "That excitement is contagious! What an incredible moment you're experiencing.",
            "I can feel your energy! Let's make sure to channel this into something meaningful.",
        ],
        'enthusiasm': [
            "Your enthusiasm is inspiring! That kind of motivation can move mountains.",
            "I love seeing this drive. What's fueling your passion right now?",
        ],
        'fun': [
            "It sounds like you're having a great time! These moments of play are so important.",
            "Joy and playfulness are gifts. What about this is making it so enjoyable?",
        ],
        'gratitude': [
            "That sense of appreciation is beautiful. Gratitude has a way of multiplying what's good.",
            "It's wonderful that you're recognizing what you have. That awareness is a strength.",
        ],
        'pride': [
            "You should be proud! That accomplishment reflects real effort and character.",
            "Celebrate this. You earned it, and acknowledging that matters.",
        ],
        # Sadness family
        'sadness': [
            "I hear you, and it's okay to feel sad. Your feelings are valid and I'm here with you.",
            "Sadness can feel heavy. Remember, it's okay to take things one moment at a time.",
        ],
        'grief': [
            "I'm so sorry for your loss. There's no right way to grieve — take all the time you need.",
            "Grief is love with nowhere to go. I'm here to sit with you in this.",
        ],
        'boredom': [
            "Boredom can be uncomfortable, but it's often a signal that something wants to change.",
            "Sometimes stillness reveals what we truly want. What's calling to you right now?",
        ],
        'nostalgia': [
            "Those memories clearly mean a lot to you. The past shaped who you are today.",
            "There's a bittersweet beauty in looking back. What would you want to carry forward?",
        ],
        # Anger family
        'anger': [
            "I can sense your frustration. It's natural to feel angry sometimes - your feelings matter.",
            "Anger often signals that something important to us has been crossed. What's behind this feeling?",
        ],
        'frustration': [
            "Being frustrated is exhausting. Let's figure out what we can do about this.",
            "I hear you. When things don't go as planned, that friction is completely valid.",
        ],
        'hate': [
            "Those are intense feelings. Something clearly matters deeply to you here.",
            "That level of emotion tells me something important is at stake. Let's explore what.",
        ],
        'contempt': [
            "When we feel contempt, it often reflects our values being violated. What standard matters here?",
            "That dismissiveness is a signal. What boundary has been crossed?",
        ],
        'disgust': [
            "That visceral reaction is your mind protecting you. What specifically triggered this?",
            "Disgust serves a purpose — it guards our boundaries. Let's understand yours.",
        ],
        'jealousy': [
            "Jealousy can feel consuming, but it often points to something we deeply want for ourselves.",
            "That comparison pain is real. But your journey has its own unique value.",
        ],
        # Fear family
        'fear': [
            "Feeling anxious or worried is completely understandable. You're not alone in this.",
            "Fear can be overwhelming. Let's take this one step at a time together.",
        ],
        'anxiety': [
            "Anxiety can make everything feel urgent. Let's slow down and find solid ground.",
            "Your nervous system is in overdrive. That's not your fault — let's help it settle.",
        ],
        'worry': [
            "Worrying shows you care, but it can also trap you. Let's separate real risks from what-ifs.",
            "I hear your concerns. Let's look at this clearly — what's actually in your control?",
        ],
        'overwhelmed': [
            "When everything hits at once, it's natural to feel this way. You don't have to solve it all now.",
            "Overwhelm is temporary, even though it doesn't feel that way. One small step at a time.",
        ],
        'stressed': [
            "Stress accumulates, and it's okay to acknowledge it. Your body is telling you something.",
            "You've been carrying a lot. Let's find one thing we can ease right now.",
        ],
        # Love family
        'love': [
            "Love is such a beautiful emotion. It's wonderful that you're experiencing this connection.",
            "The warmth in your words is touching. Love enriches our lives in so many ways.",
        ],
        'compassion': [
            "Your empathy speaks volumes about your character. The world needs more of this.",
            "Caring deeply is a strength, not a weakness. How can we channel this?",
        ],
        # Calm family
        'calm': [
            "That sense of peace is precious. Savor this moment of equilibrium.",
            "Being calm is a skill and a gift. Notice how your body feels right now.",
        ],
        'relief': [
            "That weight lifting is real. You made it through — that takes strength.",
            "Relief reminds us that difficult things do end. Take a moment to appreciate this.",
        ],
        'mindfulness': [
            "Being present is one of the most powerful things you can do. Stay with this awareness.",
            "This moment of mindfulness is a gift to yourself. What do you notice?",
        ],
        'resilience': [
            "Your ability to bounce back is remarkable. That strength is a core part of who you are.",
            "Resilience isn't about never falling — it's about getting back up. And you do.",
        ],
        'hope': [
            "Hope is powerful fuel. It means you believe things can get better — and they can.",
            "That spark of optimism matters more than you might think. Let's nurture it.",
        ],
        # Self-Conscious family
        'guilt': [
            "Guilt means your moral compass is working. That's actually a good sign.",
            "Feeling guilty is uncomfortable, but it also shows you care about doing right.",
        ],
        'shame': [
            "Shame tells us we're not enough — but that's a lie. You are more than your worst moment.",
            "What you're feeling is heavy, and you don't deserve to carry it alone.",
        ],
        # Atomic
        'surprise': [
            "That sounds unexpected! How are you processing this surprise?",
            "Life can certainly catch us off guard. How are you feeling about this?",
        ],
        'neutral': [
            "I'm here to listen. How can I support you today?",
            "Thank you for sharing. What's on your mind right now?",
        ],
    }

    def coach(
        self,
        message: str,
        emotion: str = None,
        biometric_data: dict = None,
        use_model: bool = False,
        context: dict = None,
        context_provider=None
    ) -> dict:
        """Generate empathetic coaching response (fast mode by default)"""
        import random

        # Auto-detect emotion if not provided
        if not emotion:
            analysis = self.analyze(message)
            emotion = analysis['dominant_emotion']

        family = _EMOTION_TO_FAMILY.get(emotion, 'Neutral')

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
            eda = biometric_data.get('eda', 0)

            if hr > 100:
                biometric_insight = "Your elevated heart rate suggests heightened arousal. A breathing exercise may help."
            elif hrv < 30:
                biometric_insight = "Your HRV indicates stress. Consider a moment of mindfulness."
            elif hr < 60 and emotion in ['sadness', 'grief', 'boredom']:
                biometric_insight = "Your low heart rate combined with mood may indicate low energy. Gentle movement could help."
            elif eda > 7:
                biometric_insight = "Your high EDA suggests system overload. Let's try a grounding exercise."
            elif hrv > 70:
                biometric_insight = "Your high HRV indicates good vagal tone. You're in a regulated state."

        # External context enrichment (only when context is provided)
        weather_insight = None
        nutrition_insight = None
        cross_validation = None
        if context and context_provider:
            enrichment = context_provider.enrich_coaching(context, message, local_family=family)
            weather_insight = enrichment.get('weather_insight')
            nutrition_insight = enrichment.get('nutrition_insight')
            cross_validation = enrichment.get('cross_validation')

        result = {
            'response': response,
            'detected_emotion': emotion,
            'family': family,
            'biometric_insight': biometric_insight,
            'coaching_prompt': COACHING_PROMPTS.get(emotion, ''),
            'user_message': message,
            'model': 'quantara-emotion-gpt'
        }

        if weather_insight is not None:
            result['weather_insight'] = weather_insight
        if nutrition_insight is not None:
            result['nutrition_insight'] = nutrition_insight
        if cross_validation is not None:
            result['cross_validation'] = cross_validation

        return result

    def get_therapy_technique(self, emotion: str) -> dict:
        """Get therapy technique, transition, and coaching prompt for emotion"""

        technique = THERAPY_TECHNIQUES.get(emotion, THERAPY_TECHNIQUES['neutral'])
        transition = TRANSITION_PATHWAYS.get(emotion, '')
        coaching = COACHING_PROMPTS.get(emotion, '')
        family = _EMOTION_TO_FAMILY.get(emotion, 'Neutral')

        return {
            'emotion': emotion,
            'family': family,
            'technique': technique['name'],
            'description': technique['description'],
            'exercise': technique['exercise'],
            'transition': transition,
            'coaching_prompt': coaching,
        }


def create_app(model: EmotionGPTModel) -> Flask:
    """Create Flask application with emotion API routes"""

    app = Flask(__name__)
    CORS(app)  # Enable CORS for frontend integration

    # Emotion Transition Engine (graph-based pathfinding + adaptive weights)
    transition_engine = EmotionTransitionEngine() if HAS_TRANSITION_ENGINE else None

    # WebSocket streaming (emotion/biometrics/system namespaces)
    socketio = init_websocket(app) if HAS_WEBSOCKET else None

    # External context provider (weather, nutrition, sentiment)
    context_provider = ExternalContextProvider() if HAS_EXTERNAL_CONTEXT else None

    # Multimodal analyzer (available even when main model is None)
    multimodal_analyzer = model.multimodal_analyzer if model and hasattr(model, 'multimodal_analyzer') else None

    # Auto-retrain manager (monitors calibration model drift and triggers retraining)
    auto_retrain_manager = None
    if HAS_AUTO_RETRAIN and context_provider and hasattr(context_provider, 'ruview') and context_provider.ruview:
        ruview = context_provider.ruview
        if hasattr(ruview, '_calibration_buffer') and ruview._calibration_buffer:
            try:
                auto_retrain_manager = AutoRetrainManager(
                    calibration_buffer=ruview._calibration_buffer,
                    model=ruview._calibration_model or ruview._calibration_buffer._model,
                    checkpoint_path='checkpoints/ruview_calibration.pt',
                    socketio=socketio,
                )
                auto_retrain_manager.start(check_interval=60)
                logger.info("[EmotionAPI] Auto-retrain manager started")
            except Exception as e:
                logger.error(f"[EmotionAPI] Auto-retrain init failed: {e}")

    # Emotion Transition Tracker (per-user emotion history + pattern detection)
    # ── Backend Integration Note ──────────────────────────────────────────────
    # The Quantara Backend (Node.js) can call the transition tracker endpoints
    # to build per-user emotion timelines and detect concerning patterns:
    #
    #   POST /api/emotion/transition/record
    #       Body: { user_id, emotion, family?, confidence?, timestamp? }
    #       Records a single emotion reading for a user.
    #
    #   GET  /api/emotion/transition/trajectory/<user_id>?window=24
    #       Returns the emotion timeline for the given window (hours).
    #
    #   GET  /api/emotion/transition/patterns/<user_id>
    #       Detects concerning patterns (rapid shifts, prolonged negative states).
    #
    #   GET  /api/emotion/transition/dashboard/<user_id>
    #       Full therapist dashboard summary (trajectory + patterns + stats).
    #
    # The unified_text_emotion_engine.js and emotion_aware_training_engine.js
    # in Quantara-Backend/ml-pipeline/ already connect to this server on port
    # 5050 (configurable via DISTILBERT_PORT / pythonApiUrl). After calling
    # /api/emotion/analyze, the backend can POST to /api/emotion/transition/record
    # with the same user_id to maintain a longitudinal emotion log. Alternatively,
    # when the analyze request includes a user_id field, this server will
    # automatically record the result in the transition tracker (see below).
    # ──────────────────────────────────────────────────────────────────────────
    transition_tracker = EmotionTransitionTracker() if HAS_TRANSITION_TRACKER else None
    if transition_tracker:
        logger.info("[EmotionAPI] Emotion Transition Tracker initialized")

    # Initialize profile engine
    global profile_engine
    if HAS_PROFILE_ENGINE:
        try:
            profile_engine = UserProfileEngine(db_path='data/profile.db')
            profile_bp = create_profile_blueprint(profile_engine)
            app.register_blueprint(profile_bp)
            print("✓ Profile engine initialized")
        except Exception as e:
            print(f"⚠ Profile engine failed to initialize: {e}")
            profile_engine = None

    # Initialize event bus + ecosystem integration
    if HAS_PROFILE_ENGINE and profile_engine:
        try:
            from profile_event_bus import ProfileEventBus
            from ecosystem_connector import EcosystemConnector
            from process_scheduler import ProcessScheduler

            event_bus = ProfileEventBus()
            ecosystem_connector = EcosystemConnector(event_bus, profile_engine.db)
            profile_engine._ecosystem_connector = ecosystem_connector
            profile_engine.set_event_bus(event_bus, ecosystem_connector)

            process_scheduler = ProcessScheduler(
                process_fn=profile_engine.process,
                debounce_seconds=30, count_threshold=20, periodic_seconds=300
            )
            process_scheduler.start()

            # WebSocket router
            try:
                from websocket_router import WebSocketRouter
                # Note: socketio may already be initialized, or we create it
                ws_router = WebSocketRouter(event_bus)
                print("✓ WebSocket router initialized")
            except Exception as e:
                print(f"⚠ WebSocket router failed: {e}")

            print("✓ Event bus + ecosystem connector initialized")
        except Exception as e:
            print(f"⚠ Event bus failed to initialize: {e}")

    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    # Wire metrics to subsystems
    if profile_engine:
        profile_engine.set_metrics(metrics_collector)
    if profile_engine and profile_engine._event_bus:
        profile_engine._event_bus.set_metrics(metrics_collector)
    if profile_engine and profile_engine._ecosystem_connector:
        profile_engine._ecosystem_connector.set_metrics(metrics_collector)

    # Initialize EmotionGPT coordinator
    emotion_gpt = None
    try:
        emotion_gpt = EmotionGPT(
            analyzer=model,
            transition_tracker=transition_tracker,
            auto_retrain_manager=auto_retrain_manager if 'auto_retrain_manager' in dir() else None,
            profile_engine=profile_engine,
            metrics=metrics_collector,
            bus=profile_engine._event_bus if profile_engine and profile_engine._event_bus else None,
        )
        print("✓ EmotionGPT coordinator initialized")
    except Exception as e:
        print(f"⚠ EmotionGPT coordinator failed: {e}")

    def require_model():
        """Check if model is loaded, return error response if not"""
        if model is None:
            return jsonify({
                'error': 'Model unavailable',
                'status': 'degraded',
                'message': 'GPT model not loaded. Multimodal analysis may still be available.',
            }), 503
        return None

    @app.route('/api/emotion/status', methods=['GET'])
    def status():
        """Health check endpoint"""
        resp = {
            'status': 'online' if model else 'degraded',
            'model': 'quantara-emotion-gpt',
            'device': model.device if model else 'none',
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0',
            'taxonomy': '32-emotion / 9-family',
            'external_context': {
                'available': HAS_EXTERNAL_CONTEXT,
                'weather': True,
                'nutrition': bool(os.environ.get('NUTRITIONIX_APP_ID')),
                'sentiment': bool(os.environ.get('NLPCLOUD_API_KEY')),
            }
        }
        if metrics_collector:
            resp['metrics'] = metrics_collector.get_all()
        return jsonify(resp)

    @app.route('/api/emotion/emotions', methods=['GET'])
    def list_emotions():
        """List supported emotions"""
        return jsonify({
            'emotions': model.EMOTIONS,
            'count': len(model.EMOTIONS),
            'families': _EMOTION_FAMILIES,
        })

    @app.route('/api/emotion/family', methods=['GET'])
    def list_families():
        """List all emotion families with their emotions"""
        return jsonify({
            'families': _EMOTION_FAMILIES,
            'count': len(_EMOTION_FAMILIES),
        })

    @app.route('/api/emotion/family/<name>', methods=['GET'])
    def get_family(name):
        """Get emotions in a specific family"""
        # Case-insensitive lookup
        for family_name, emotions in _EMOTION_FAMILIES.items():
            if family_name.lower() == name.lower():
                return jsonify({
                    'family': family_name,
                    'emotions': emotions,
                    'count': len(emotions),
                })
        return jsonify({'error': f'Family not found: {name}'}), 404

    @app.route('/api/emotion/generate', methods=['POST'])
    def generate():
        """Generate emotion-aware text"""
        err = require_model()
        if err: return err
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
        """Analyze emotional content — returns emotion, family, confidence, is_fallback"""
        err = require_model()
        if err: return err
        try:
            data = request.json or {}
            text = data.get('text', '')

            if not text:
                return jsonify({'error': 'text is required'}), 400

            biometrics = data.get('biometrics')
            result = model.analyze(text, biometrics)

            # Classifier metrics
            metrics_collector.increment('classifier.requests')
            metrics_collector.increment('classifier.confidence_sum',
                                       float(result.get('confidence', 0)))
            if result.get('is_fallback'):
                metrics_collector.increment('classifier.fallback_count')

            # Stream emotion update via WebSocket
            if HAS_WEBSOCKET:
                try:
                    ws_emit_emotion(result)
                except Exception:
                    pass

            # Auto-record to transition tracker when user_id is provided
            user_id = data.get('user_id')
            if user_id and transition_tracker:
                try:
                    transition_tracker.record(
                        user_id=user_id,
                        emotion=result.get('dominant_emotion', result.get('emotion', 'neutral')),
                        family=result.get('family'),
                        confidence=float(result.get('confidence', 1.0)),
                    )
                except Exception:
                    # Non-blocking: tracker errors must not affect the analysis response
                    pass

            # Log to profile engine
            # NOTE: Intentionally removed the 'default' user_id fallback that was
            # here. Using 'default' created a ghost user that accumulated all
            # anonymous requests, polluting profile data.
            if profile_engine and user_id:
                try:
                    profile_engine.log_event(user_id, 'emotional', 'emotion_classified', {
                        'emotion': result.get('dominant_emotion', result.get('emotion', 'neutral')),
                        'family': result.get('family', 'Neutral'),
                        'confidence': result.get('confidence', 0.5),
                    }, 'nanogpt', result.get('confidence'))
                    if text:
                        profile_engine.log_event(user_id, 'linguistic', 'text_analyzed', {
                            'text': text[:500],
                        }, 'nanogpt')
                    profile_engine.log_event(user_id, 'temporal', 'session_started', {
                        'endpoint': 'analyze',
                    }, 'nanogpt')
                except Exception:
                    pass

            # Apply profile personalization when user_id is present
            snapshot = None
            if user_id and profile_engine:
                try:
                    snapshot = profile_engine.get_profile_snapshot(user_id)
                    apply_profile_personalization(result, snapshot)

                    # Publish event bus event on tiebreak swap
                    if (result.get('personalized') and
                            'profile_tiebreak' in result.get('personalization_reason', '') and
                            profile_engine._event_bus):
                        try:
                            profile_engine._event_bus.publish('profile.personalization.applied', {
                                'user_id': user_id,
                                'personalized_family': result.get('family'),
                                'reason': result.get('personalization_reason'),
                            })
                        except Exception:
                            pass
                except Exception:
                    result['personalized'] = False
            elif user_id and not profile_engine:
                result['personalized'] = False

            # Personalization metrics
            if user_id and profile_engine:
                metrics_collector.increment('personalization.requests')
                if result.get('personalized'):
                    reason = result.get('personalization_reason', '')
                    if 'profile_tiebreak' in reason:
                        metrics_collector.increment('personalization.swapped')
                    else:
                        metrics_collector.increment('personalization.consulted')
                else:
                    metrics_collector.increment('personalization.skipped')

            # Opt-in profile context enrichment (reuses snapshot from above)
            include_profile = data.get('include_profile', False)
            if include_profile and user_id and profile_engine:
                try:
                    if snapshot is None:
                        snapshot = profile_engine.get_profile_snapshot(user_id)
                    if snapshot:
                        from datetime import datetime, timezone
                        result['profile_context'] = {
                            'evolution_stage': snapshot.evolution_stage,
                            'overall_confidence': snapshot.overall_confidence,
                            'dominant_family': snapshot.dominant_family,
                            'emotion_prior': snapshot.emotion_prior,
                            'event_count': snapshot.event_count,
                            'last_updated': datetime.fromtimestamp(
                                snapshot.last_updated, tz=timezone.utc
                            ).isoformat(),
                        }
                except Exception:
                    pass

            return jsonify({**result, 'status': 'success'})

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/emotion/coach', methods=['POST'])
    def coach():
        """Get empathetic coaching response"""
        err = require_model()
        if err: return err
        try:
            data = request.json or {}
            message = data.get('message', '')

            if not message:
                return jsonify({'error': 'message is required'}), 400

            result = model.coach(
                message=message,
                emotion=data.get('emotion'),
                biometric_data=data.get('biometric'),
                use_model=data.get('use_model', False),
                context=data.get('context'),
                context_provider=context_provider
            )

            return jsonify({**result, 'status': 'success'})

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/emotion/therapy', methods=['POST'])
    def therapy():
        """Get therapy technique, transition pathway, and coaching prompt"""
        err = require_model()
        if err: return err
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
        """Neural Ecosystem workflow integration — family-aware triggers"""
        err = require_model()
        if err: return err
        try:
            data = request.json or {}

            text = data.get('text', data.get('message', ''))
            biometric = data.get('biometric', {})

            analysis = model.analyze(text) if text else {'dominant_emotion': 'neutral', 'family': 'Neutral'}

            emotion = analysis.get('dominant_emotion', 'neutral')
            family = analysis.get('family', _EMOTION_TO_FAMILY.get(emotion, 'Neutral'))

            # Family-aware workflow actions
            workflow_actions = {
                'Joy': ['log_positive_moment', 'suggest_sharing', 'trigger_savoring_exercise'],
                'Sadness': ['trigger_support_check', 'schedule_wellness_activity', 'notify_care_team'],
                'Anger': ['enable_cool_down_mode', 'suggest_break', 'offer_reframing_exercise'],
                'Fear': ['activate_grounding_exercise', 'notify_support_contact', 'trigger_breathing_guide'],
                'Love': ['log_connection_moment', 'suggest_gratitude_expression'],
                'Calm': ['reinforce_positive_state', 'suggest_deepening_practice'],
                'Self-Conscious': ['trigger_self_compassion', 'offer_perspective_shift'],
                'Surprise': ['log_notable_event', 'offer_processing_space'],
                'Neutral': ['continue_normal_workflow'],
            }

            return jsonify({
                'emotion_analysis': analysis,
                'biometric_data': biometric,
                'family': family,
                'recommended_actions': workflow_actions.get(family, []),
                'workflow_trigger': f'emotion_{emotion}',
                'family_trigger': f'family_{family.lower()}',
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            })

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/neural/emotion-transition', methods=['POST'])
    def emotion_transition():
        """Get emotion transition pathway between two emotions.

        If to_emotion is provided AND transition engine is available,
        returns a multi-step graph-based pathway. Otherwise, falls back
        to legacy single-step TRANSITION_PATHWAYS lookup.
        """
        try:
            data = request.json or {}
            from_emotion = data.get('from_emotion', '').lower()
            to_emotion = data.get('to_emotion', '').lower() if data.get('to_emotion') else ''

            if not from_emotion:
                return jsonify({'error': 'from_emotion is required'}), 400

            family = _EMOTION_TO_FAMILY.get(from_emotion, 'Neutral')

            # Enhanced: multi-step pathway via graph engine
            if to_emotion and transition_engine:
                try:
                    path = transition_engine.find_path(from_emotion, to_emotion)
                    total_duration = sum(s.get('duration_min', 0) for s in path)
                    return jsonify({
                        'from_emotion': from_emotion,
                        'from_family': family,
                        'to_emotion': to_emotion,
                        'pathway': path,
                        'total_steps': len(path),
                        'total_duration_min': total_duration,
                        'mode': 'graph',
                        'status': 'success'
                    })
                except ValueError as ve:
                    return jsonify({'error': str(ve), 'status': 'error'}), 400

            # Legacy: single-step lookup
            transition = TRANSITION_PATHWAYS.get(from_emotion, '')
            technique = THERAPY_TECHNIQUES.get(from_emotion, THERAPY_TECHNIQUES['neutral'])
            coaching = COACHING_PROMPTS.get(from_emotion, '')

            return jsonify({
                'from_emotion': from_emotion,
                'from_family': family,
                'to_emotion': to_emotion or 'auto',
                'transition': transition,
                'technique': technique['name'],
                'exercise': technique['exercise'],
                'coaching_prompt': coaching,
                'mode': 'legacy',
                'status': 'success'
            })

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/neural/transition-session', methods=['POST'])
    def transition_session():
        """Manage transition sessions (start, advance, query).

        Actions:
          start   - Start a new session: {action: 'start', user_id, from_emotion, to_emotion}
          advance - Advance session step: {action: 'advance', session_id, biometrics: {hr_delta, hrv_delta}}
          query   - Get session state:   {action: 'query', session_id}
        """
        if not transition_engine:
            return jsonify({'error': 'Transition engine not available', 'status': 'error'}), 503

        try:
            data = request.json or {}
            action = data.get('action', '')

            if action == 'start':
                user_id = data.get('user_id', 'anonymous')
                from_emotion = data.get('from_emotion', '').lower()
                to_emotion = data.get('to_emotion', '').lower()
                if not from_emotion or not to_emotion:
                    return jsonify({'error': 'from_emotion and to_emotion are required'}), 400
                session = transition_engine.start_session(user_id, from_emotion, to_emotion)
                return jsonify({
                    'session': session.to_dict(),
                    'current_step': session.get_current_step(),
                    'status': 'success'
                })

            elif action == 'advance':
                session_id = data.get('session_id', '')
                if not session_id:
                    return jsonify({'error': 'session_id is required'}), 400
                session = transition_engine.get_session(session_id)
                if not session:
                    return jsonify({'error': 'Session not found'}), 404
                biometrics = data.get('biometrics', {})
                bio_check = session.check_biometric_criteria(biometrics)
                session.advance()
                return jsonify({
                    'session': session.to_dict(),
                    'current_step': session.get_current_step(),
                    'biometric_criteria_met': bio_check,
                    'status': 'success'
                })

            elif action == 'query':
                session_id = data.get('session_id', '')
                if not session_id:
                    return jsonify({'error': 'session_id is required'}), 400
                session = transition_engine.get_session(session_id)
                if not session:
                    return jsonify({'error': 'Session not found'}), 404
                return jsonify({
                    'session': session.to_dict(),
                    'current_step': session.get_current_step(),
                    'status': 'success'
                })

            else:
                return jsonify({'error': f"Unknown action: {action}. Use start, advance, or query"}), 400

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/neural/transition-feedback', methods=['POST'])
    def transition_feedback():
        """Log outcome feedback for a transition technique.

        Body: {from_emotion, to_emotion, technique, success: bool}
        """
        if not transition_engine:
            return jsonify({'error': 'Transition engine not available', 'status': 'error'}), 503

        try:
            data = request.json or {}
            from_emotion = data.get('from_emotion', '').lower()
            to_emotion = data.get('to_emotion', '').lower()
            technique = data.get('technique', '')
            success = data.get('success', True)

            if not from_emotion or not to_emotion or not technique:
                return jsonify({'error': 'from_emotion, to_emotion, and technique are required'}), 400

            transition_engine.log_feedback(from_emotion, to_emotion, technique, success=bool(success))
            return jsonify({
                'logged': True,
                'from_emotion': from_emotion,
                'to_emotion': to_emotion,
                'technique': technique,
                'success': bool(success),
                'status': 'success'
            })

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    # ─── External Context Endpoints ─────────────────────────────────────────

    @app.route('/api/context/weather', methods=['POST'])
    def context_weather():
        """Get weather and air quality for a location"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        try:
            data = request.json or {}
            lat = data.get('latitude')
            lon = data.get('longitude')
            if lat is None or lon is None:
                return jsonify({'error': 'latitude and longitude are required', 'status': 'error'}), 400

            result = context_provider.get_weather(float(lat), float(lon))
            if result is None:
                return jsonify({'error': 'Weather service unavailable', 'status': 'error'}), 502
            return jsonify({**result, 'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/context/nutrition', methods=['POST'])
    def context_nutrition():
        """Analyze food log for mood-relevant nutrition data"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        try:
            data = request.json or {}
            food_log = data.get('food_log')
            if not food_log or not isinstance(food_log, list):
                return jsonify({'error': 'food_log (list of strings) is required', 'status': 'error'}), 400

            result = context_provider.get_nutrition(food_log)
            if result is None:
                return jsonify({'error': 'Nutrition service unavailable', 'status': 'error'}), 502
            return jsonify({**result, 'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/context/sentiment', methods=['POST'])
    def context_sentiment():
        """Cross-validate text sentiment via NLP Cloud"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        try:
            data = request.json or {}
            text = data.get('text')
            if not text:
                return jsonify({'error': 'text is required', 'status': 'error'}), 400

            result = context_provider.validate_sentiment(text)
            if result is None:
                return jsonify({'error': 'Sentiment service unavailable', 'status': 'error'}), 502
            return jsonify({**result, 'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    # ─── RuView WiFi Sensing Endpoints (Ghost Protocol) ──────────────────

    @app.route('/api/ruview/status', methods=['GET'])
    def ruview_status():
        """Check RuView WiFi sensing connection status"""
        if not context_provider or not context_provider.ruview:
            return jsonify({
                'status': 'unavailable',
                'message': 'RuView provider not configured',
            })
        rv = context_provider.ruview
        return jsonify({
            'status': 'connected' if rv.is_connected else 'disconnected',
            'host': rv.host,
            'http_port': rv.http_port,
            'ws_port': rv.ws_port,
        })

    @app.route('/api/ruview/connect', methods=['POST'])
    def ruview_connect():
        """Connect to RuView WiFi sensing service"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        rv = context_provider.ruview
        if not rv:
            return jsonify({'error': 'RuView provider not available', 'status': 'error'}), 503
        connected = rv.connect()
        return jsonify({
            'connected': connected,
            'status': 'success' if connected else 'failed',
        })

    @app.route('/api/ruview/biometrics', methods=['GET'])
    def ruview_biometrics():
        """Get current biometrics from RuView WiFi sensing"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        bio = context_provider.get_ruview_biometrics()
        if bio is None:
            return jsonify({'error': 'RuView data unavailable', 'status': 'error'}), 502
        if profile_engine:
            try:
                profile_engine.log_event('default', 'biometric', 'hr_reading', {
                    'hr': bio.get('heart_rate'),
                    'hrv': bio.get('hrv'),
                    'eda': bio.get('eda'),
                }, 'nanogpt')
            except Exception:
                pass
        return jsonify({**bio, 'status': 'success'})

    @app.route('/api/ruview/presence', methods=['GET'])
    def ruview_presence():
        """Get presence/occupancy from RuView WiFi sensing"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        presence = context_provider.get_ruview_presence()
        if presence is None:
            return jsonify({'error': 'RuView data unavailable', 'status': 'error'}), 502
        return jsonify({**presence, 'status': 'success'})

    @app.route('/api/ruview/analyze', methods=['POST'])
    def ruview_analyze():
        """
        Analyze emotion using text + RuView WiFi biometrics.
        Ghost protocol: biometric data sourced from WiFi sensing, no cameras/wearables.
        """
        if not multimodal_analyzer:
            return jsonify({'error': 'Multimodal analyzer not available', 'status': 'error'}), 503

        try:
            data = request.json or {}
            text = data.get('text', '')

            # Get biometrics from RuView
            ruview_bio = None
            if context_provider:
                ruview_bio = context_provider.get_ruview_biometrics()

            # Get pose features from RuView
            pose_features = None
            if context_provider and context_provider.ruview:
                pose_features = context_provider.ruview.get_pose_features()

            # Merge with any explicitly passed biometrics (explicit wins)
            biometrics = data.get('biometrics') or ruview_bio

            result = multimodal_analyzer.analyze(text, biometrics=biometrics, pose=pose_features)

            # Add RuView context
            result['pose_features'] = pose_features
            result['calibration_profile'] = (
                context_provider.ruview._calibration_buffer.profile_id
                if hasattr(context_provider.ruview, '_calibration_buffer') and context_provider.ruview._calibration_buffer
                else 'none'
            ) if context_provider and context_provider.ruview else 'none'

            if ruview_bio:
                result['ruview_source'] = True
                result['ruview_confidence'] = ruview_bio.get('confidence', 0)
                result['ruview_insight'] = context_provider.ruview.get_insight() if context_provider.ruview else None
                result['presence'] = context_provider.get_ruview_presence()
            else:
                result['ruview_source'] = False

            # Stream emotion update via WebSocket
            if HAS_WEBSOCKET:
                try:
                    ws_emit_emotion(result)
                except Exception:
                    pass

            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    # ─── Auto-Retrain Endpoints ─────────────────────────────────────────────

    @app.route('/api/calibration/retrain-status', methods=['GET'])
    def retrain_status():
        """Get current auto-retrain pipeline status"""
        if not auto_retrain_manager:
            return jsonify({
                'status': 'unavailable',
                'message': 'Auto-retrain manager not initialized',
            }), 503
        return jsonify(auto_retrain_manager.get_status())

    @app.route('/api/calibration/retrain', methods=['POST'])
    def trigger_retrain():
        """Manually trigger a calibration model retrain"""
        if not auto_retrain_manager:
            return jsonify({
                'status': 'error',
                'message': 'Auto-retrain manager not initialized',
            }), 503
        result = auto_retrain_manager.manual_retrain()
        return jsonify(result)

    @app.route('/api/calibration/retrain-log', methods=['GET'])
    def retrain_log():
        """Get retrain event history"""
        if not auto_retrain_manager:
            return jsonify({
                'status': 'unavailable',
                'message': 'Auto-retrain manager not initialized',
            }), 503
        limit = request.args.get('limit', 50, type=int)
        entries = auto_retrain_manager.retrain_log.get_log(limit=limit)
        return jsonify({'log': entries, 'count': len(entries)})

    # ── Emotion Transition Tracker endpoints (Therapist Dashboard) ─────

    @app.route('/api/emotion/transition/record', methods=['POST'])
    def transition_record():
        """Record an emotion reading for per-user transition tracking"""
        if not transition_tracker:
            return jsonify({'error': 'Transition tracker unavailable'}), 503
        try:
            data = request.json or {}
            user_id = data.get('user_id')
            emotion = data.get('emotion')
            if not user_id or not emotion:
                return jsonify({
                    'error': 'Missing required fields: user_id, emotion',
                }), 400
            result = transition_tracker.record(
                user_id=user_id,
                emotion=emotion,
                family=data.get('family'),
                confidence=float(data.get('confidence', 1.0)),
                timestamp=data.get('timestamp'),
            )
            if profile_engine:
                try:
                    confidence = float(data.get('confidence', 1.0))
                    family = data.get('family')
                    profile_engine.log_event(user_id, 'behavioral', 'transition_recorded', {
                        'emotion': emotion,
                        'family': family,
                    }, 'nanogpt', confidence)
                except Exception:
                    pass
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/emotion/transition/trajectory/<user_id>', methods=['GET'])
    def transition_trajectory(user_id):
        """Get emotion timeline for a user (therapist dashboard)"""
        if not transition_tracker:
            return jsonify({'error': 'Transition tracker unavailable'}), 503
        try:
            window = request.args.get('window_hours', 24, type=float)
            result = transition_tracker.get_trajectory(user_id, window_hours=window)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/emotion/transition/patterns/<user_id>', methods=['GET'])
    def transition_patterns(user_id):
        """Detect concerning patterns in a user's emotion history"""
        if not transition_tracker:
            return jsonify({'error': 'Transition tracker unavailable'}), 503
        try:
            alerts = transition_tracker.detect_patterns(user_id)
            return jsonify({
                'user_id': user_id,
                'alerts': alerts,
                'count': len(alerts),
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/emotion/transition/dashboard/<user_id>', methods=['GET'])
    def transition_dashboard(user_id):
        """Full therapist dashboard summary for a user"""
        if not transition_tracker:
            return jsonify({'error': 'Transition tracker unavailable'}), 503
        try:
            summary = transition_tracker.get_dashboard_summary(user_id)
            return jsonify(summary)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description='Quantara Emotion GPT API Server (32 Emotions)')
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
    print("  QUANTARA EMOTION GPT API SERVER (32 Emotions)")
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
    print(f"  - GET  /api/emotion/family")
    print(f"  - GET  /api/emotion/family/<name>")
    print(f"  - POST /api/emotion/generate")
    print(f"  - POST /api/emotion/analyze")
    print(f"  - POST /api/emotion/coach")
    print(f"  - POST /api/emotion/therapy")
    print(f"  - POST /api/neural/emotion-workflow")
    print(f"  - POST /api/neural/emotion-transition")
    print(f"  - GET  /api/ruview/status")
    print(f"  - POST /api/ruview/connect")
    print(f"  - GET  /api/ruview/biometrics")
    print(f"  - GET  /api/ruview/presence")
    print(f"  - POST /api/ruview/analyze")
    print(f"  - GET  /api/calibration/retrain-status")
    print(f"  - POST /api/calibration/retrain")
    print(f"  - GET  /api/calibration/retrain-log")
    print(f"  - POST /api/emotion/transition/record")
    print(f"  - GET  /api/emotion/transition/trajectory/<user_id>")
    print(f"  - GET  /api/emotion/transition/patterns/<user_id>")
    print(f"  - GET  /api/emotion/transition/dashboard/<user_id>")
    print(f"\n  Starting server on http://{args.host}:{args.port}")
    print("=" * 60 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


# Module-level app for gunicorn/Railway deployment
def get_app():
    """Create app for WSGI deployment (gunicorn)"""
    import os

    try:
        from download_model import download_model
        checkpoint = download_model()
    except Exception as e:
        print(f"[EmotionGPT] Model download failed: {e}")
        checkpoint = None

    if checkpoint is None:
        checkpoint = os.environ.get('CHECKPOINT_PATH', 'out-quantara-emotion-fast/ckpt.pt')

    device = os.environ.get('DEVICE', 'cpu')

    print(f"[EmotionGPT] Initializing for WSGI deployment")
    print(f"[EmotionGPT] Checkpoint: {checkpoint}")
    print(f"[EmotionGPT] Device: {device}")

    try:
        model = EmotionGPTModel(checkpoint_path=checkpoint, device=device)
        print(f"[EmotionGPT] Model loaded successfully")
    except Exception as e:
        print(f"[EmotionGPT] WARNING: Model loading failed: {e}")
        print(f"[EmotionGPT] Starting in multimodal-only mode (sentence-transformers)")
        model = None

    return create_app(model)

# Create app at module level for gunicorn
app = get_app()

if __name__ == '__main__':
    main()
