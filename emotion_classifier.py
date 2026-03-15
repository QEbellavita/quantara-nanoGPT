# emotion_classifier.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Multimodal Emotion Classifier (32-Emotion Taxonomy)
===============================================================================
ML-powered emotion analysis combining text embeddings with biometric signals.
Two-stage hierarchical classification: family (9) → sub-emotion (32).

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Biometric Integration Engine
- Real-time Dashboard Data
- Emotion-Aware Training Engine
- Therapist Dashboard Engine
===============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import time
from pathlib import Path

# Optional: sentence-transformers for better text embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ─── 32-Emotion Taxonomy ────────────────────────────────────────────────────

EMOTION_FAMILIES = {
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

FAMILY_NAMES = list(EMOTION_FAMILIES.keys())

# Build reverse lookup: emotion → family
_EMOTION_TO_FAMILY = {}
for _family, _emotions in EMOTION_FAMILIES.items():
    for _e in _emotions:
        _EMOTION_TO_FAMILY[_e] = _family

# Build family → emotion indices (populated after FusionHead.EMOTIONS is defined)
_FAMILY_EMOTION_INDICES = {}


def family_for_emotion(emotion: str) -> str:
    """Get family name for an emotion. O(1) lookup."""
    return _EMOTION_TO_FAMILY.get(emotion.lower(), 'Neutral')


class SentenceEncoderWrapper:
    """
    Wrapper for sentence-transformers providing semantic text embeddings.

    Much better for emotion classification than character-level nanoGPT.
    Uses all-MiniLM-L6-v2 (384-dim) by default.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.device = device

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to embedding tensor."""
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.unsqueeze(0)  # (1, dim)

    def encode_batch(self, texts: list) -> torch.Tensor:
        """Encode batch of texts."""
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings  # (batch, dim)


class BiometricEncoder(nn.Module):
    """
    Encode biometric signals into a dense representation.

    Input features (8 total):
    - heart_rate_norm, hrv_norm, eda_norm (3 raw normalized)
    - hr_arousal: high HR indicator (>100 bpm)
    - hrv_stress: low HRV indicator (<30 ms)
    - eda_activation: high EDA indicator (>5 µS)
    - hrv_calm: high HRV indicator (>65 ms) — distinguishes Calm family
    - eda_overwhelm: very high EDA indicator (>7 µS) — distinguishes overwhelmed

    Output: 16-dimensional encoded representation
    """

    # Normalization constants (based on physiological ranges)
    DEFAULTS = {
        'heart_rate': 70.0,
        'hrv': 50.0,
        'eda': 2.0
    }

    RANGES = {
        'heart_rate': (40.0, 180.0),
        'hrv': (10.0, 100.0),
        'eda': (0.5, 20.0)
    }

    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim

        # Input: 8 features (3 raw + 5 derived)
        self.encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )

    def _normalize(self, value: float, key: str) -> float:
        """Normalize value to [-1, 1] range."""
        min_val, max_val = self.RANGES[key]
        normalized = (value - min_val) / (max_val - min_val)
        return max(-1.0, min(1.0, 2.0 * normalized - 1.0))

    def _extract_features(self, biometrics: dict) -> torch.Tensor:
        """Extract and normalize features from biometrics dict."""
        if biometrics is None:
            biometrics = {}

        hr = biometrics.get('heart_rate', self.DEFAULTS['heart_rate'])
        hrv = biometrics.get('hrv', self.DEFAULTS['hrv'])
        eda = biometrics.get('eda', self.DEFAULTS['eda'])

        features = [
            self._normalize(hr, 'heart_rate'),
            self._normalize(hrv, 'hrv'),
            self._normalize(eda, 'eda'),
            1.0 if hr > 100 else 0.0,   # hr_arousal
            1.0 if hrv < 30 else 0.0,   # hrv_stress
            1.0 if eda > 5 else 0.0,    # eda_activation
            1.0 if hrv > 65 else 0.0,   # hrv_calm (Calm family discriminator)
            1.0 if eda > 7 else 0.0,    # eda_overwhelm (overwhelmed discriminator)
        ]

        return torch.tensor(features, dtype=torch.float32)

    def encode(self, biometrics: dict) -> torch.Tensor:
        """Encode single biometrics dict to tensor. Returns (1, output_dim)."""
        features = self._extract_features(biometrics).unsqueeze(0)
        with torch.no_grad():
            return self.encoder(features)

    def encode_batch(self, batch: list) -> torch.Tensor:
        """Encode batch of biometrics dicts. Returns (batch_size, output_dim)."""
        features = torch.stack([self._extract_features(b) for b in batch])
        with torch.no_grad():
            return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        return self.encoder(features)


class FusionHead(nn.Module):
    """
    Multimodal fusion classification head with two-stage hierarchy.

    Stage 1: Classify into 9 emotion families
    Stage 2: Classify sub-emotion within predicted family (32 total)

    Confidence fallback: if sub-emotion confidence < threshold,
    returns the family root emotion instead.
    """

    # All 32 emotions in canonical order (family-grouped)
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
        'surprise',
        'neutral',
    ]

    # Family root emotions (first emotion in each family, used for fallback)
    FAMILY_ROOTS = {
        'Joy': 'joy', 'Sadness': 'sadness', 'Anger': 'anger',
        'Fear': 'fear', 'Love': 'love', 'Calm': 'calm',
        'Self-Conscious': 'guilt', 'Surprise': 'surprise', 'Neutral': 'neutral',
    }

    def __init__(
        self,
        text_dim: int = 512,
        biometric_dim: int = 16,
        pose_dim: int = 16,
        hidden_dim: int = 128,
        num_emotions: int = 32,
        num_families: int = 9,
        dropout: float = 0.3
    ):
        super().__init__()
        self.text_dim = text_dim
        self.biometric_dim = biometric_dim
        self.pose_dim = pose_dim
        self.num_emotions = num_emotions
        self.num_families = num_families

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(text_dim + biometric_dim + pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Stage 1: Family classifier (9-way)
        self.family_classifier = nn.Linear(hidden_dim // 2, num_families)

        # Stage 2: Sub-emotion classifier (32-way)
        self.emotion_classifier = nn.Linear(hidden_dim // 2, num_emotions)

        # Zero biometric embedding for text-only mode
        self.register_buffer(
            'zero_biometric',
            torch.zeros(1, biometric_dim)
        )

        # Zero pose embedding for when pose data is unavailable
        if pose_dim > 0:
            self.register_buffer(
                'zero_pose',
                torch.zeros(1, pose_dim)
            )

        # Build family → emotion index mapping
        self._build_family_indices()

    def _build_family_indices(self):
        """Build mapping from family index to emotion indices."""
        global _FAMILY_EMOTION_INDICES
        for fi, family_name in enumerate(FAMILY_NAMES):
            emotion_names = EMOTION_FAMILIES[family_name]
            indices = [self.EMOTIONS.index(e) for e in emotion_names]
            _FAMILY_EMOTION_INDICES[fi] = indices

    def forward(
        self,
        text_embedding: torch.Tensor,
        biometric_embedding: torch.Tensor = None,
        pose_embedding: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass returning both emotion and family logits.

        Returns:
            (emotion_probs, family_probs) — both softmaxed
        """
        batch_size = text_embedding.shape[0]

        if biometric_embedding is None:
            biometric_embedding = self.zero_biometric.expand(batch_size, -1)

        parts = [text_embedding, biometric_embedding]

        if self.pose_dim > 0:
            if pose_embedding is None:
                pose_embedding = self.zero_pose.expand(batch_size, -1)
            parts.append(pose_embedding)

        fused = torch.cat(parts, dim=-1)
        shared_features = self.shared(fused)

        emotion_logits = self.emotion_classifier(shared_features)
        family_logits = self.family_classifier(shared_features)

        emotion_probs = torch.softmax(emotion_logits, dim=-1)
        family_probs = torch.softmax(family_logits, dim=-1)

        return emotion_probs, family_probs

    def classify_with_fallback(
        self,
        text_embedding: torch.Tensor,
        biometric_embedding: torch.Tensor = None,
        pose_embedding: torch.Tensor = None,
        threshold: float = 0.6
    ) -> dict:
        """
        Two-stage classification with confidence fallback.

        1. Classify into family (9-way)
        2. Classify sub-emotion within predicted family
        3. If sub-emotion confidence < threshold, return family root emotion

        Returns:
            {emotion, family, confidence, is_fallback}
        """
        emotion_probs, family_probs = self.forward(text_embedding, biometric_embedding, pose_embedding)

        # Squeeze batch dim for single-sample inference
        emotion_probs = emotion_probs.squeeze(0)
        family_probs = family_probs.squeeze(0)

        # Stage 1: Get predicted family
        family_idx = int(torch.argmax(family_probs))
        family_name = FAMILY_NAMES[family_idx]
        family_confidence = float(family_probs[family_idx])

        # Stage 2: Get best sub-emotion within that family
        family_emotion_indices = _FAMILY_EMOTION_INDICES.get(family_idx, [])

        if family_emotion_indices:
            family_emotion_probs = emotion_probs[family_emotion_indices]
            # Re-normalize within family
            family_emotion_probs = family_emotion_probs / (family_emotion_probs.sum() + 1e-8)

            best_within_family = int(torch.argmax(family_emotion_probs))
            sub_emotion_idx = family_emotion_indices[best_within_family]
            sub_emotion = self.EMOTIONS[sub_emotion_idx]
            sub_confidence = float(family_emotion_probs[best_within_family])
        else:
            sub_emotion = self.FAMILY_ROOTS.get(family_name, 'neutral')
            sub_confidence = 0.0

        # Fallback: if sub-emotion confidence is low, use family root
        is_fallback = sub_confidence < threshold
        if is_fallback:
            emotion = self.FAMILY_ROOTS.get(family_name, sub_emotion)
            confidence = family_confidence
        else:
            emotion = sub_emotion
            confidence = sub_confidence

        return {
            'emotion': emotion,
            'family': family_name,
            'confidence': float(confidence),
            'is_fallback': is_fallback,
            'family_confidence': float(family_confidence),
            'sub_emotion_confidence': float(sub_confidence),
        }

    def get_emotion_name(self, index: int) -> str:
        """Get emotion name from index."""
        return self.EMOTIONS[index]

    def get_emotion_index(self, name: str) -> int:
        """Get index from emotion name."""
        return self.EMOTIONS.index(name.lower())

    def get_family_index(self, family_name: str) -> int:
        """Get family index from name."""
        return FAMILY_NAMES.index(family_name)


class AttentionFusionHead(FusionHead):
    """
    Cross-modal attention fusion head.

    Instead of concatenating modalities equally, uses multi-head cross-attention
    where text is the query and biometric/pose signals are keys/values.
    This lets the model learn which modality matters most per emotion context.

    When non-text modalities are absent, they are masked out of the attention
    computation. When ALL non-text modalities are absent, falls back to a
    text-only projection path (skipping attention entirely).

    Connected to:
    - Neural Workflow AI Engine
    - ML Training & Prediction Systems
    - Biometric Integration Engine
    - Real-time Dashboard Data
    """

    def __init__(
        self,
        text_dim: int = 384,
        biometric_dim: int = 16,
        pose_dim: int = 16,
        hidden_dim: int = 128,
        num_emotions: int = 32,
        num_families: int = 9,
        dropout: float = 0.3,
        num_heads: int = 4
    ):
        # Initialize parent (keeps backward compat layers)
        super().__init__(
            text_dim=text_dim,
            biometric_dim=biometric_dim,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_emotions=num_emotions,
            num_families=num_families,
            dropout=dropout,
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Modality projections to shared hidden_dim space
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.bio_proj = nn.Linear(biometric_dim, hidden_dim)
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)

        # Multi-head cross-attention: text=query, bio+pose=key/value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention layers
        self.post_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Override parent classifiers with new dimensions
        self.family_classifier = nn.Linear(hidden_dim // 2, num_families)
        self.emotion_classifier = nn.Linear(hidden_dim // 2, num_emotions)

        # Store last attention weights for modality weight extraction
        self._last_attn_weights = None
        self._last_modality_mask = None  # tracks which modalities were present

    def forward(
        self,
        text_embedding: torch.Tensor,
        biometric_embedding: torch.Tensor = None,
        pose_embedding: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass with cross-modal attention.

        Returns:
            (emotion_probs, family_probs) — both softmaxed
        """
        batch_size = text_embedding.shape[0]

        # Project text to shared space
        text_proj = self.text_proj(text_embedding)  # (B, hidden_dim)

        # Check which non-text modalities are present
        has_bio = biometric_embedding is not None
        has_pose = pose_embedding is not None and self.pose_dim > 0

        if not has_bio and not has_pose:
            # Text-only path: skip attention, use text projection directly
            # Residual: text_proj + text_proj (identity residual)
            features = self.post_attn(text_proj)
            self._last_attn_weights = None
            self._last_modality_mask = (True, False, False)
        else:
            # Build key/value sequences from available modalities
            kv_parts = []
            modality_order = []  # track which modality each position corresponds to

            if has_bio:
                bio_proj = self.bio_proj(biometric_embedding)  # (B, hidden_dim)
                kv_parts.append(bio_proj.unsqueeze(1))  # (B, 1, hidden_dim)
                modality_order.append('bio')

            if has_pose:
                pose_proj = self.pose_proj(pose_embedding)  # (B, hidden_dim)
                kv_parts.append(pose_proj.unsqueeze(1))  # (B, 1, hidden_dim)
                modality_order.append('pose')

            # Stack key/value: (B, num_kv, hidden_dim)
            kv = torch.cat(kv_parts, dim=1)

            # Query is text: (B, 1, hidden_dim)
            query = text_proj.unsqueeze(1)

            # Cross-attention
            attended, attn_weights = self.cross_attention(
                query, kv, kv, need_weights=True
            )
            # attended: (B, 1, hidden_dim), attn_weights: (B, 1, num_kv)

            self._last_attn_weights = attn_weights.detach()
            self._last_modality_mask = (True, has_bio, has_pose)

            # Residual connection: attended + text_proj
            fused = attended.squeeze(1) + text_proj  # (B, hidden_dim)

            features = self.post_attn(fused)

        emotion_logits = self.emotion_classifier(features)
        family_logits = self.family_classifier(features)

        emotion_probs = torch.softmax(emotion_logits, dim=-1)
        family_probs = torch.softmax(family_logits, dim=-1)

        return emotion_probs, family_probs

    def classify_with_fallback(
        self,
        text_embedding: torch.Tensor,
        biometric_embedding: torch.Tensor = None,
        pose_embedding: torch.Tensor = None,
        threshold: float = 0.6
    ) -> dict:
        """
        Two-stage classification with confidence fallback and modality weights.

        Extends parent classify_with_fallback with modality_weights info.
        """
        # Run forward to populate _last_attn_weights
        result = super().classify_with_fallback(
            text_embedding, biometric_embedding, pose_embedding, threshold
        )

        # Add modality weights
        result['modality_weights'] = self._get_modality_weights()

        return result

    def _get_modality_weights(self) -> dict:
        """
        Extract normalized modality contributions from last attention pass.

        Text weight = 1 - sum(other weights).
        When text-only (no attention), text=1.0, others=0.0.
        """
        has_text, has_bio, has_pose = self._last_modality_mask or (True, False, False)

        if self._last_attn_weights is None:
            # Text-only path was used
            return {'text': 1.0, 'bio': 0.0, 'pose': 0.0}

        # attn_weights shape: (B, 1, num_kv) — take first batch item
        weights = self._last_attn_weights[0, 0]  # (num_kv,)

        bio_weight = 0.0
        pose_weight = 0.0
        idx = 0

        if has_bio:
            bio_weight = float(weights[idx])
            idx += 1
        if has_pose:
            pose_weight = float(weights[idx])
            idx += 1

        # Normalize: attention weights over non-text modalities represent
        # how much the model attends to them. Text contribution is the residual.
        # Scale so total = 1.0
        other_total = bio_weight + pose_weight
        # The residual connection means text always contributes.
        # We model text weight as complementary to attention weights.
        # Attention weights already sum to 1 over KV positions, so we scale:
        # text gets at least 50% (residual), rest split by attention
        text_weight = 0.5
        bio_weight = 0.5 * bio_weight / (other_total + 1e-8) * other_total
        pose_weight = 0.5 * pose_weight / (other_total + 1e-8) * other_total

        # Simplify: text=0.5, bio and pose share the other 0.5
        # proportional to attention weights
        if other_total > 1e-8:
            bio_weight = 0.5 * (bio_weight / other_total) if has_bio else 0.0
            pose_weight = 0.5 * (pose_weight / other_total) if has_pose else 0.0
        else:
            bio_weight = 0.0
            pose_weight = 0.0

        text_weight = 1.0 - bio_weight - pose_weight

        return {
            'text': round(text_weight, 6),
            'bio': round(bio_weight, 6),
            'pose': round(pose_weight, 6),
        }


class MultimodalEmotionAnalyzer:
    """
    Main interface for multimodal emotion analysis.

    Combines text embeddings with biometric signals for accurate
    emotion classification across 32 emotions in 9 families.

    Connected to:
    - Neural Workflow AI Engine
    - AI Conversational Coach
    - Biometric Integration Engine
    - Therapist Dashboard Engine
    """

    def __init__(
        self,
        gpt_checkpoint: str = None,
        classifier_checkpoint: str = None,
        device: str = 'auto',
        use_sentence_transformer: bool = True,
        sentence_model: str = 'all-MiniLM-L6-v2'
    ):
        self.device = self._detect_device(device)
        self.use_sentence_transformer = use_sentence_transformer and HAS_SENTENCE_TRANSFORMERS
        self.sentence_encoder = None

        # Try to use sentence-transformers for better text embeddings
        if self.use_sentence_transformer:
            try:
                print(f"[EmotionAnalyzer] Loading sentence-transformers ({sentence_model})...")
                self.sentence_encoder = SentenceEncoderWrapper(sentence_model, device=self.device)
                self.n_embd = self.sentence_encoder.embedding_dim
                print(f"[EmotionAnalyzer] Sentence encoder loaded: {self.n_embd}-dim embeddings")
            except Exception as e:
                print(f"[EmotionAnalyzer] Sentence encoder failed: {e}, falling back to nanoGPT")
                self.use_sentence_transformer = False

        # Fall back to nanoGPT if sentence-transformers not available
        if not self.use_sentence_transformer:
            if gpt_checkpoint and Path(gpt_checkpoint).exists():
                self._load_gpt(gpt_checkpoint)
            else:
                self._create_dummy_gpt()

        # Initialize biometric encoder (8-feature input)
        self.biometric_encoder = BiometricEncoder(output_dim=16)
        self.biometric_encoder.to(self.device)

        # Initialize pose encoder (8 pose features → 16-dim embedding)
        self.pose_encoder = None
        try:
            from pose_encoder import PoseEncoder
            self.pose_encoder = PoseEncoder()
            self.pose_encoder.to(self.device)
        except ImportError:
            pass

        # Load or create fusion head (32 emotions, 9 families)
        # Default to AttentionFusionHead (v2); fall back to FusionHead (v1) for old checkpoints
        pose_dim = 16 if self.pose_encoder else 0

        # Detect checkpoint version to choose correct head class
        fusion_version = 2  # default to AttentionFusionHead
        if classifier_checkpoint and Path(classifier_checkpoint).exists():
            state = torch.load(classifier_checkpoint, map_location=self.device, weights_only=False)
            meta = state.get('meta', {})
            fusion_version = meta.get('fusion_version', 1)  # old checkpoints are v1
        else:
            state = None

        if fusion_version >= 2:
            self.fusion_head = AttentionFusionHead(
                text_dim=self.n_embd,
                biometric_dim=16,
                pose_dim=pose_dim,
                num_emotions=32,
                num_families=9
            )
        else:
            self.fusion_head = FusionHead(
                text_dim=self.n_embd,
                biometric_dim=16,
                pose_dim=pose_dim,
                num_emotions=32,
                num_families=9
            )

        if state is not None:
            # Determine pose_dim from checkpoint meta or by inspecting shapes
            meta = state.get('meta', {})
            saved_pose_dim = meta.get('pose_dim', None)
            saved_input_dim = state['fusion_head']['shared.0.weight'].shape[1]
            saved_num_emotions = state['fusion_head']['emotion_classifier.weight'].shape[0]

            if saved_pose_dim is None:
                # Infer: input_dim = text_dim + biometric_dim + pose_dim
                # Try to detect old 400-dim (no pose) vs new 416-dim (with pose)
                saved_pose_dim = saved_input_dim - 16  # subtract biometric_dim
                saved_text_dim = saved_pose_dim  # Initially assume no pose
                # Check if the saved dim matches current text+bio+pose
                if saved_input_dim == self.n_embd + 16:
                    saved_pose_dim = 0
                    saved_text_dim = self.n_embd
                elif saved_input_dim == self.n_embd + 16 + pose_dim:
                    saved_pose_dim = pose_dim
                    saved_text_dim = self.n_embd
                else:
                    saved_text_dim = saved_input_dim - 16
                    saved_pose_dim = 0
            else:
                saved_text_dim = saved_input_dim - 16 - saved_pose_dim

            if saved_text_dim != self.n_embd:
                print(f"[EmotionAnalyzer] Dimension mismatch: saved={saved_text_dim}, current={self.n_embd}")
                print(f"[EmotionAnalyzer] Recreating fusion head with new dimensions (requires retraining)")
            elif saved_num_emotions != 32:
                print(f"[EmotionAnalyzer] Emotion count mismatch: saved={saved_num_emotions}, current=32")
                print(f"[EmotionAnalyzer] Recreating fusion head (requires retraining)")
            elif saved_pose_dim != pose_dim:
                print(f"[EmotionAnalyzer] Pose dim mismatch: saved={saved_pose_dim}, current={pose_dim}")
                print(f"[EmotionAnalyzer] Recreating fusion head (requires retraining)")
            else:
                self.fusion_head.load_state_dict(state['fusion_head'])
                self.biometric_encoder.load_state_dict(state['biometric_encoder'])
                # Load pose encoder state if present
                if self.pose_encoder and 'pose_encoder' in state:
                    self.pose_encoder.load_state_dict(state['pose_encoder'])

        self.fusion_head.to(self.device)
        self.fusion_head.eval()

        # Setup tokenizer (only needed for nanoGPT fallback)
        if not self.use_sentence_transformer:
            self._setup_tokenizer()

    def _detect_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        return device

    def _load_gpt(self, checkpoint_path: str):
        """Load trained GPT model."""
        from model import GPT, GPTConfig

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        gptconf = GPTConfig(**checkpoint['model_args'])
        self.gpt = GPT(gptconf)
        self.n_embd = gptconf.n_embd

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        self.gpt.load_state_dict(state_dict)
        self.gpt.eval()
        self.gpt.to(self.device)

    def _create_dummy_gpt(self):
        """Create a small GPT for testing without checkpoint."""
        from model import GPT, GPTConfig

        config = GPTConfig(
            block_size=256,
            vocab_size=256,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=True
        )
        self.gpt = GPT(config)
        self.gpt.eval()
        self.gpt.to(self.device)
        self.n_embd = 64

        # Adjust fusion head for smaller embedding
        self.fusion_head = AttentionFusionHead(
            text_dim=64,
            biometric_dim=16,
            pose_dim=0,
            num_emotions=32,
            num_families=9
        )

    def _setup_tokenizer(self):
        """Setup tokenizer (char-level or BPE) - must match training tokenizer."""
        meta_path = Path('data/quantara_emotion/meta.pkl')

        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi.get(c, 0) for c in s]
        else:
            # Use BPE tokenizer (must match training script)
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Extract embedding from text using sentence-transformers or nanoGPT."""
        if not text:
            text = " "  # Avoid empty tensor

        # Use sentence-transformers if available (much better for emotion)
        if self.use_sentence_transformer and self.sentence_encoder:
            embedding = self.sentence_encoder.encode(text)
            return embedding.to(self.device)

        # Fallback to nanoGPT character-level embeddings
        tokens = self.encode(text)
        # Truncate to block size
        max_len = min(len(tokens), self.gpt.config.block_size)
        tokens = tokens[:max_len]

        idx = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            embedding = self.gpt.get_embedding(idx)

        return embedding

    def analyze(
        self,
        text: str,
        biometrics: dict = None,
        pose: dict = None,
        return_embedding: bool = False
    ) -> dict:
        """
        Analyze emotional content of text with optional biometrics and pose.

        Uses two-stage classification with confidence fallback.

        Returns:
            Dict with emotion, family, confidence, is_fallback, scores
        """
        start_time = time.time()

        # Get text embedding
        text_embedding = self._get_text_embedding(text)

        # Get biometric embedding
        if biometrics:
            bio_embedding = self.biometric_encoder.encode(biometrics).to(self.device)
            biometric_contribution = 0.3
        else:
            bio_embedding = None
            biometric_contribution = 0.0

        # Get pose embedding
        pose_embedding = None
        if pose and self.pose_encoder:
            pose_embedding = self.pose_encoder.encode(pose).to(self.device)

        # Two-stage classify with fallback
        with torch.no_grad():
            classification = self.fusion_head.classify_with_fallback(
                text_embedding, bio_embedding, pose_embedding
            )
            emotion_probs, family_probs = self.fusion_head(
                text_embedding, bio_embedding, pose_embedding
            )

        emotion_probs_np = emotion_probs.squeeze(0).cpu().numpy()

        # Build scores dict
        scores = {
            emotion: float(emotion_probs_np[i])
            for i, emotion in enumerate(FusionHead.EMOTIONS)
        }

        latency_ms = (time.time() - start_time) * 1000

        result = {
            'dominant_emotion': classification['emotion'],
            'family': classification['family'],
            'confidence': classification['confidence'],
            'is_fallback': classification['is_fallback'],
            'scores': scores,
            'biometric_contribution': biometric_contribution,
            'latency_ms': round(latency_ms, 2),
            'status': 'success'
        }

        # Include modality weights if available (AttentionFusionHead)
        if 'modality_weights' in classification:
            result['modality_weights'] = classification['modality_weights']

        if return_embedding:
            result['embedding'] = text_embedding.squeeze(0).cpu().numpy().tolist()

        return result

    def analyze_batch(
        self,
        texts: list,
        biometrics_list: list = None
    ) -> list:
        """Analyze batch of texts."""
        if biometrics_list is None:
            biometrics_list = [None] * len(texts)

        return [
            self.analyze(text, bio)
            for text, bio in zip(texts, biometrics_list)
        ]
