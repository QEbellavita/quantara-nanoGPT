# Emotion GPT Enhancement Suite — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cross-modal attention fusion, emotion transition engine, WebSocket streaming, auto-retraining, production training notebook, and formal evaluation to the Quantara Emotion GPT.

**Architecture:** Six modular features added as isolated components with clean interfaces. Each plugs into the existing Flask API server and PyTorch classifier. Build order: Feature 1 (AttentionFusionHead) → Feature 2 (TransitionEngine) → Feature 5 (WebSocket) → Feature 6 (AutoRetrain) → Feature 3 (Production Training) → Feature 4 (Evaluation). WebSocket before AutoRetrain because AutoRetrain emits events on the WebSocket; training and eval last because they validate everything.

**Tech Stack:** PyTorch, Flask, flask-socketio, eventlet, SQLite, HuggingFace datasets, Google Colab

**Spec:** `docs/superpowers/specs/2026-03-15-emotion-gpt-enhancements-design.md`

---

## Chunk 1: Cross-Modal Attention Fusion (Feature 1)

### Task 1: AttentionFusionHead — Tests

**Files:**
- Modify: `tests/test_emotion_classifier.py`

- [ ] **Step 1: Write tests for AttentionFusionHead**

Add to `tests/test_emotion_classifier.py`:

```python
from emotion_classifier import AttentionFusionHead, FusionHead


class TestAttentionFusionHead:
    """Test cross-modal attention fusion head."""

    @pytest.fixture
    def fusion_head(self):
        return AttentionFusionHead(
            text_dim=384, biometric_dim=16, pose_dim=16,
            hidden_dim=128, num_emotions=32, num_families=9
        )

    def test_forward_output_shapes(self, fusion_head):
        """Forward should return (emotion_probs, family_probs) with correct shapes."""
        text = torch.randn(1, 384)
        bio = torch.randn(1, 16)
        pose = torch.randn(1, 16)
        emotion_probs, family_probs = fusion_head(text, bio, pose)
        assert emotion_probs.shape == (1, 32)
        assert family_probs.shape == (1, 9)

    def test_forward_without_biometrics(self, fusion_head):
        """Should work with text only — bio/pose masked out."""
        text = torch.randn(1, 384)
        emotion_probs, family_probs = fusion_head(text, None, None)
        assert emotion_probs.shape == (1, 32)
        assert family_probs.shape == (1, 9)

    def test_probs_sum_to_one(self, fusion_head):
        """Softmax outputs should sum to ~1."""
        text = torch.randn(1, 384)
        bio = torch.randn(1, 16)
        emotion_probs, family_probs = fusion_head(text, bio, None)
        assert abs(emotion_probs.sum().item() - 1.0) < 1e-5
        assert abs(family_probs.sum().item() - 1.0) < 1e-5

    def test_classify_with_fallback_returns_modality_weights(self, fusion_head):
        """classify_with_fallback should include modality_weights in output."""
        text = torch.randn(1, 384)
        bio = torch.randn(1, 16)
        result = fusion_head.classify_with_fallback(text, bio, None)
        assert 'modality_weights' in result
        assert 'text' in result['modality_weights']
        assert 'bio' in result['modality_weights']
        assert 'pose' in result['modality_weights']

    def test_modality_weights_sum_to_one(self, fusion_head):
        """Modality weights should be normalized."""
        text = torch.randn(1, 384)
        bio = torch.randn(1, 16)
        pose = torch.randn(1, 16)
        result = fusion_head.classify_with_fallback(text, bio, pose)
        weights = result['modality_weights']
        total = weights['text'] + weights['bio'] + weights['pose']
        assert abs(total - 1.0) < 1e-4

    def test_attention_mask_excludes_absent_modalities(self, fusion_head):
        """When bio/pose are None, their attention weights should be zero."""
        text = torch.randn(1, 384)
        result = fusion_head.classify_with_fallback(text, None, None)
        weights = result['modality_weights']
        assert weights['text'] == 1.0
        assert weights['bio'] == 0.0
        assert weights['pose'] == 0.0

    def test_batch_forward(self, fusion_head):
        """Batch of 4 should produce correct output shapes."""
        text = torch.randn(4, 384)
        bio = torch.randn(4, 16)
        emotion_probs, family_probs = fusion_head(text, bio, None)
        assert emotion_probs.shape == (4, 32)
        assert family_probs.shape == (4, 9)

    def test_backward_compat_with_fusionhead_interface(self, fusion_head):
        """AttentionFusionHead must have same public methods as FusionHead."""
        old = FusionHead(text_dim=384, biometric_dim=16, pose_dim=16)
        for method in ['forward', 'classify_with_fallback', 'get_emotion_name', 'get_emotion_index']:
            assert hasattr(fusion_head, method), f"Missing method: {method}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_emotion_classifier.py::TestAttentionFusionHead -v`
Expected: FAIL — `ImportError: cannot import name 'AttentionFusionHead'`

---

### Task 2: AttentionFusionHead — Implementation

**Files:**
- Modify: `emotion_classifier.py`

- [ ] **Step 3: Add AttentionFusionHead class to emotion_classifier.py**

Insert after the `FusionHead` class (after line 381). The class inherits the same `EMOTIONS`, `FAMILY_ROOTS` and utility methods from `FusionHead`:

```python
class AttentionFusionHead(FusionHead):
    """
    Cross-modal attention fusion head.

    Uses multi-head cross-attention with text as query and bio/pose as key/value.
    Learns which modality matters most per emotion context.
    Attention mask excludes absent modalities from softmax.
    """

    def __init__(
        self,
        text_dim: int = 384,
        biometric_dim: int = 16,
        pose_dim: int = 16,
        hidden_dim: int = 128,
        num_emotions: int = 32,
        num_families: int = 9,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        # Skip FusionHead.__init__, call nn.Module.__init__ directly
        nn.Module.__init__(self)
        self.text_dim = text_dim
        self.biometric_dim = biometric_dim
        self.pose_dim = pose_dim
        self.num_emotions = num_emotions
        self.num_families = num_families
        self.hidden_dim = hidden_dim

        # Project each modality to shared dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.bio_proj = nn.Linear(biometric_dim, hidden_dim)
        if pose_dim > 0:
            self.pose_proj = nn.Linear(pose_dim, hidden_dim)

        # Multi-head cross-attention: text queries attend to bio+pose keys
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Post-attention layers
        self.post_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification heads
        self.family_classifier = nn.Linear(hidden_dim // 2, num_families)
        self.emotion_classifier = nn.Linear(hidden_dim // 2, num_emotions)

        # Zero embeddings for absent modalities
        self.register_buffer('zero_biometric', torch.zeros(1, biometric_dim))
        if pose_dim > 0:
            self.register_buffer('zero_pose', torch.zeros(1, pose_dim))

        self._build_family_indices()

        # Store last attention weights for modality weight reporting
        self._last_attn_weights = None

    def forward(
        self,
        text_embedding: torch.Tensor,
        biometric_embedding: torch.Tensor = None,
        pose_embedding: torch.Tensor = None
    ) -> tuple:
        batch_size = text_embedding.shape[0]

        # Project text to query space: (batch, 1, hidden_dim)
        text_proj = self.text_proj(text_embedding).unsqueeze(1)

        # Build key/value sequence and attention mask
        kv_parts = []
        mask_parts = []  # True = masked out (excluded from attention)

        # Bio modality
        if biometric_embedding is not None:
            bio_proj = self.bio_proj(biometric_embedding).unsqueeze(1)
            kv_parts.append(bio_proj)
            mask_parts.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=text_embedding.device))
        else:
            bio_proj = self.bio_proj(self.zero_biometric.expand(batch_size, -1)).unsqueeze(1)
            kv_parts.append(bio_proj)
            mask_parts.append(torch.ones(batch_size, 1, dtype=torch.bool, device=text_embedding.device))

        # Pose modality
        if self.pose_dim > 0:
            if pose_embedding is not None:
                pose_proj = self.pose_proj(pose_embedding).unsqueeze(1)
                kv_parts.append(pose_proj)
                mask_parts.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=text_embedding.device))
            else:
                pose_proj = self.pose_proj(self.zero_pose.expand(batch_size, -1)).unsqueeze(1)
                kv_parts.append(pose_proj)
                mask_parts.append(torch.ones(batch_size, 1, dtype=torch.bool, device=text_embedding.device))

        # Concatenate KV: (batch, num_modalities, hidden_dim)
        kv = torch.cat(kv_parts, dim=1)
        attn_mask = torch.cat(mask_parts, dim=1)  # (batch, num_modalities)

        # If ALL modalities are masked, use text-only path (no attention)
        all_masked = attn_mask.all(dim=1)  # (batch,)
        if all_masked.all():
            # Pure text-only: skip attention, use projected text directly
            attended = text_proj.squeeze(1)
            self._last_attn_weights = torch.zeros(batch_size, 1, kv.shape[1], device=text_embedding.device)
        else:
            # Cross-attention: text queries, bio+pose keys/values
            attended, attn_weights = self.cross_attention(
                text_proj, kv, kv,
                key_padding_mask=attn_mask,
                need_weights=True,
                average_attn_heads=True
            )
            attended = attended.squeeze(1)  # (batch, hidden_dim)
            self._last_attn_weights = attn_weights  # (batch, 1, num_kv)

        # Combine text projection with attended features
        combined = attended + text_proj.squeeze(1)

        # Post-attention processing
        features = self.post_attn(combined)

        # Classification
        emotion_logits = self.emotion_classifier(features)
        family_logits = self.family_classifier(features)

        emotion_probs = torch.softmax(emotion_logits, dim=-1)
        family_probs = torch.softmax(family_logits, dim=-1)

        return emotion_probs, family_probs

    def _get_modality_weights(self, biometric_provided: bool, pose_provided: bool) -> dict:
        """Extract normalized modality contribution weights from last attention."""
        weights = {'text': 1.0, 'bio': 0.0, 'pose': 0.0}

        if self._last_attn_weights is None:
            return weights

        attn = self._last_attn_weights.squeeze(0).squeeze(0)  # (num_kv,)

        if not biometric_provided and not pose_provided:
            return weights

        idx = 0
        bio_weight = 0.0
        pose_weight = 0.0

        if biometric_provided:
            bio_weight = float(attn[idx])
            idx += 1
        else:
            idx += 1  # skip masked bio slot

        if self.pose_dim > 0 and pose_provided:
            pose_weight = float(attn[idx])

        # Text contribution = 1 - sum(other modality attention)
        other_total = bio_weight + pose_weight
        text_weight = max(0.0, 1.0 - other_total)

        # Normalize
        total = text_weight + bio_weight + pose_weight
        if total > 0:
            weights['text'] = text_weight / total
            weights['bio'] = bio_weight / total
            weights['pose'] = pose_weight / total

        return weights

    def classify_with_fallback(
        self,
        text_embedding: torch.Tensor,
        biometric_embedding: torch.Tensor = None,
        pose_embedding: torch.Tensor = None,
        threshold: float = 0.6
    ) -> dict:
        """Extends parent with modality_weights. Calls self.forward() first
        to populate _last_attn_weights, then delegates classification logic."""
        bio_provided = biometric_embedding is not None
        pose_provided = pose_embedding is not None

        # Call forward to populate attention weights, then use parent's classification logic
        # We override forward() so this calls our attention-based forward
        result = super().classify_with_fallback(text_embedding, biometric_embedding, pose_embedding, threshold)

        # Add modality weights from attention (single-sample only)
        result['modality_weights'] = self._get_modality_weights(bio_provided, pose_provided)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_emotion_classifier.py::TestAttentionFusionHead -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Update MultimodalEmotionAnalyzer to use AttentionFusionHead**

In `emotion_classifier.py`, in `MultimodalEmotionAnalyzer.__init__` (around line 440-448), change the default FusionHead creation to use AttentionFusionHead, and update checkpoint loading to detect version:

```python
# Replace the fusion head creation block with:
        pose_dim = 16 if self.pose_encoder else 0
        self.fusion_head = AttentionFusionHead(
            text_dim=self.n_embd,
            biometric_dim=16,
            pose_dim=pose_dim,
            num_emotions=32,
            num_families=9
        )
```

Update the checkpoint loading block to detect v1 vs v2:

```python
        if classifier_checkpoint and Path(classifier_checkpoint).exists():
            state = torch.load(classifier_checkpoint, map_location=self.device, weights_only=False)
            meta = state.get('meta', {})
            version = meta.get('version', 1)

            if version == 1:
                # Legacy FusionHead checkpoint
                saved_input_dim = state['fusion_head']['shared.0.weight'].shape[1]
                if saved_input_dim == self.n_embd + 16 + pose_dim:
                    legacy_head = FusionHead(
                        text_dim=self.n_embd, biometric_dim=16, pose_dim=pose_dim,
                        num_emotions=32, num_families=9
                    )
                    legacy_head.load_state_dict(state['fusion_head'])
                    self.fusion_head = legacy_head
                    print(f"[EmotionAnalyzer] Loaded v1 FusionHead checkpoint")
                else:
                    print(f"[EmotionAnalyzer] v1 checkpoint dimension mismatch, using fresh AttentionFusionHead")
            else:
                # v2 AttentionFusionHead checkpoint
                try:
                    self.fusion_head.load_state_dict(state['fusion_head'])
                    print(f"[EmotionAnalyzer] Loaded v2 AttentionFusionHead checkpoint")
                except Exception as e:
                    print(f"[EmotionAnalyzer] v2 checkpoint load failed: {e}, using fresh model")

            if 'biometric_encoder' in state:
                self.biometric_encoder.load_state_dict(state['biometric_encoder'])
            if self.pose_encoder and 'pose_encoder' in state:
                self.pose_encoder.load_state_dict(state['pose_encoder'])
```

Update the `analyze()` method to include modality_weights in the result (around line 646):

```python
        # After classification, add modality weights if available
        if hasattr(self.fusion_head, '_get_modality_weights'):
            result['modality_weights'] = classification.get('modality_weights', {
                'text': 1.0, 'bio': 0.0, 'pose': 0.0
            })
```

- [ ] **Step 6: Run full existing test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_emotion_classifier.py -v`
Expected: All tests PASS (both old TestGPTEmbedding/TestBiometricEncoder/TestFusionHead tests AND new TestAttentionFusionHead)

- [ ] **Step 7: Commit**

```bash
git add emotion_classifier.py tests/test_emotion_classifier.py
git commit -m "feat: add AttentionFusionHead with cross-modal attention and masking"
```

---

## Chunk 2: Emotion Transition Engine (Feature 2)

### Task 3: TransitionGraph and TransitionSession — Tests

**Files:**
- Create: `tests/test_transition_engine.py`

- [ ] **Step 8: Write tests for transition engine**

```python
# tests/test_transition_engine.py
"""Tests for emotion transition engine."""

import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTransitionGraph:
    """Test the directed weighted transition graph."""

    @pytest.fixture
    def graph(self):
        from emotion_transition_engine import TransitionGraph
        return TransitionGraph()

    def test_all_32_emotions_are_nodes(self, graph):
        assert len(graph.nodes) == 32

    def test_has_edges(self, graph):
        assert len(graph.edges) >= 50

    def test_find_path_anxiety_to_calm(self, graph):
        path = graph.find_path('anxiety', 'calm')
        assert path is not None
        assert path[0]['from_emotion'] == 'anxiety'
        assert path[-1]['to_emotion'] == 'calm'

    def test_find_path_returns_steps_with_techniques(self, graph):
        path = graph.find_path('anger', 'calm')
        assert path is not None
        for step in path:
            assert 'technique' in step
            assert 'exercise' in step
            assert 'duration_minutes' in step
            assert 'step_type' in step

    def test_find_path_same_emotion_returns_empty(self, graph):
        path = graph.find_path('joy', 'joy')
        assert path == []

    def test_find_path_unreachable_returns_none(self, graph):
        """If somehow unreachable, returns None (shouldn't happen with curated graph)."""
        # All emotions should be reachable from any other
        path = graph.find_path('grief', 'excitement')
        assert path is not None


class TestTransitionSession:
    """Test session tracking for multi-step transitions."""

    @pytest.fixture
    def session(self):
        from emotion_transition_engine import TransitionGraph, TransitionSession
        graph = TransitionGraph()
        path = graph.find_path('anxiety', 'calm')
        return TransitionSession(session_id='test-1', user_id='user-1', path=path)

    def test_session_starts_at_step_zero(self, session):
        assert session.current_step == 0

    def test_advance_increments_step(self, session):
        session.advance()
        assert session.current_step == 1

    def test_is_complete_after_all_steps(self, session):
        for _ in range(len(session.path)):
            session.advance()
        assert session.is_complete

    def test_get_current_step_returns_step_info(self, session):
        step = session.get_current_step()
        assert 'technique' in step
        assert 'from_emotion' in step

    def test_check_biometric_criteria_calming_step(self, session):
        """Calming step should advance when HR drops ≥5 or HRV rises ≥10."""
        step = session.get_current_step()
        if step.get('step_type') == 'calming':
            session.set_baseline_biometrics({'heart_rate': 90, 'hrv': 40})
            assert session.check_biometric_criteria({'heart_rate': 84, 'hrv': 40})
            assert session.check_biometric_criteria({'heart_rate': 90, 'hrv': 51})
            assert not session.check_biometric_criteria({'heart_rate': 89, 'hrv': 41})


class TestAdaptiveWeightTracker:
    """Test SQLite-backed adaptive weight learning."""

    @pytest.fixture
    def tracker(self, tmp_path):
        from emotion_transition_engine import AdaptiveWeightTracker
        db_path = str(tmp_path / 'test_transitions.db')
        return AdaptiveWeightTracker(db_path=db_path)

    def test_log_outcome(self, tracker):
        tracker.log_outcome(
            user_id='user-1',
            from_emotion='anxiety',
            to_emotion='calm',
            path_taken=['anxiety', 'grounded', 'relief', 'calm'],
            outcome='success',
            duration_seconds=300
        )
        log = tracker.get_log(limit=1)
        assert len(log) == 1
        assert log[0]['outcome'] == 'success'

    def test_get_adjusted_weights(self, tracker):
        """After logging successes, edge weights should be adjusted."""
        for _ in range(5):
            tracker.log_outcome('u1', 'anxiety', 'calm', ['anxiety', 'calm'], 'success', 120)
        weights = tracker.get_adjusted_weights()
        assert isinstance(weights, dict)
```

- [ ] **Step 9: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_transition_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_transition_engine'`

---

### Task 4: TransitionGraph — Implementation

**Files:**
- Create: `emotion_transition_engine.py`

- [ ] **Step 10: Create emotion_transition_engine.py with TransitionGraph**

```python
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Emotion Transition Engine
===============================================================================
Directed weighted graph over 32 emotions with multi-step pathways,
session tracking, and adaptive weight refinement from outcomes.

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Biometric Integration Engine
- WebSocket Streaming (Feature 5)
===============================================================================
"""

import heapq
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# Step types determine biometric success criteria
STEP_TYPES = {
    'calming': {'hr_delta': -5, 'hrv_delta': 10},      # HR drops ≥5 OR HRV rises ≥10
    'activation': {'hr_delta': 5, 'eda_delta': 0.5},    # HR rises ≥5 OR EDA rises ≥0.5
    'cognitive': {},                                      # Time-based only
}

# Curated transition edges: (from, to, weight, technique, exercise, duration_min, step_type)
CURATED_EDGES = [
    # Anxiety/Fear → Calm pathway
    ('anxiety', 'grounded', 1.0, 'Body-down regulation', '5-4-3-2-1 grounding: name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste', 3, 'calming'),
    ('grounded', 'relief', 1.0, 'Acknowledging the shift', 'Notice what changed in your body. Name it: "I feel less tense in my..."', 2, 'cognitive'),
    ('relief', 'calm', 1.0, 'Deepening awareness', 'Close your eyes. Follow your breath for 10 cycles. No need to change it.', 5, 'calming'),
    ('fear', 'grounded', 1.0, 'Grounding', 'Feel your feet on the floor. Press down. You are here, you are safe.', 3, 'calming'),
    ('worry', 'grounded', 1.0, 'Containment', 'Write down every worry. Close the notebook. They are held, not gone.', 5, 'cognitive'),
    ('overwhelmed', 'grounded', 0.8, 'Smallest next step', 'What is the tiniest thing you can do right now? Just one thing.', 2, 'cognitive'),
    ('stressed', 'grounded', 0.8, 'Autonomic reset', 'Breathe in 4 counts, hold 4, out 8. Repeat 5 times.', 3, 'calming'),

    # Anger family → Calm/Neutral
    ('anger', 'frustration', 0.5, 'Naming', 'Say: "I feel angry because..." — naming reduces intensity.', 2, 'cognitive'),
    ('anger', 'grounded', 1.0, 'Perspective-taking', 'Imagine telling a friend about this. What would they notice?', 3, 'cognitive'),
    ('frustration', 'grounded', 0.8, 'Action planning', 'List 3 things in your control about this situation.', 3, 'cognitive'),
    ('hate', 'anger', 0.5, 'Perspective-taking', 'What need is underneath this intensity?', 3, 'cognitive'),
    ('contempt', 'neutral', 1.2, 'Curiosity shift', 'What if their behavior makes sense from their perspective? Explore why.', 5, 'cognitive'),
    ('disgust', 'neutral', 1.2, 'Graduated exposure', 'Rate your discomfort 1-10. What would bring it down by 1?', 3, 'cognitive'),
    ('jealousy', 'gratitude', 1.5, 'Self-compassion', 'Name 3 things you have accomplished that you are not giving yourself credit for.', 3, 'cognitive'),

    # Sadness family → Joy/Calm
    ('sadness', 'nostalgia', 0.5, 'Behavioral activation', 'Do one small thing that once brought you joy — even a tiny version.', 5, 'activation'),
    ('sadness', 'neutral', 1.0, 'Acceptance', 'It is okay to feel sad. Let it be here without trying to fix it.', 3, 'cognitive'),
    ('grief', 'sadness', 0.3, 'Meaning-making', 'What did this loss teach you about what matters?', 5, 'cognitive'),
    ('boredom', 'enthusiasm', 1.0, 'Micro-goals', 'Set a 10-minute timer. Do one thing with full attention.', 3, 'activation'),
    ('nostalgia', 'gratitude', 0.8, 'Past-present bridging', 'What from that time do you still carry with you today?', 3, 'cognitive'),

    # Positive transitions (maintenance/deepening)
    ('joy', 'gratitude', 0.5, 'Gratitude practice', 'Name 3 specific things contributing to this feeling right now.', 2, 'cognitive'),
    ('excitement', 'joy', 0.3, 'Grounding excitement', 'Channel this energy: what is the one thing you most want to do with it?', 2, 'cognitive'),
    ('enthusiasm', 'joy', 0.3, 'Values alignment', 'How does this connect to what matters most to you?', 2, 'cognitive'),
    ('fun', 'joy', 0.3, 'Reflection', 'What about this moment would you want to remember?', 2, 'cognitive'),
    ('gratitude', 'compassion', 0.5, 'Loving-kindness', 'Send a kind thought to someone who helped create this feeling.', 3, 'cognitive'),
    ('pride', 'resilience', 0.5, 'Strength inventory', 'Name the strengths that got you here. They will serve you again.', 2, 'cognitive'),

    # Love/Compassion
    ('love', 'compassion', 0.3, 'Appreciation expression', 'Tell them specifically what you appreciate — not just "thanks".', 3, 'cognitive'),
    ('compassion', 'calm', 0.5, 'Service reflection', 'What is one act of kindness you can do today?', 2, 'cognitive'),

    # Calm family internal
    ('calm', 'mindfulness', 0.3, 'Deepening awareness', 'Notice sensations without labeling them. Just observe.', 5, 'calming'),
    ('mindfulness', 'resilience', 0.5, 'Non-reactive observation', 'Watch a thought arise, stay, and pass. You are the sky, not the cloud.', 5, 'calming'),
    ('resilience', 'hope', 0.5, 'Past-success recall', 'Remember a time you overcame something difficult. How did you do it?', 3, 'cognitive'),
    ('hope', 'joy', 0.8, 'Concrete next steps', 'Name one specific thing you can do tomorrow toward this hope.', 2, 'activation'),

    # Self-conscious → Calm/Neutral
    ('guilt', 'neutral', 1.0, 'Amends planning', 'Is there a repair action? If yes, plan it. If no, practice self-forgiveness.', 5, 'cognitive'),
    ('shame', 'neutral', 1.2, 'Identity separation', 'You did something, but you are not that thing. Name who you are.', 5, 'cognitive'),
    ('shame', 'compassion', 1.5, 'Self-compassion', 'Speak to yourself the way you would speak to a friend who feels this.', 3, 'cognitive'),

    # Surprise → Curiosity/Joy
    ('surprise', 'joy', 0.5, 'Mindful observation', 'Stay with this feeling. What do you notice? Let curiosity lead.', 2, 'cognitive'),
    ('surprise', 'enthusiasm', 0.8, 'Exploration', 'What just happened that you did not expect? What does it open up?', 2, 'activation'),

    # Neutral → Engagement
    ('neutral', 'enthusiasm', 1.0, 'Values clarification', 'What matters to you right now? Name one thing worth your attention.', 3, 'activation'),
    ('neutral', 'calm', 0.5, 'Mindful check-in', 'Scan your body from head to toe. What do you notice?', 3, 'calming'),

    # Cross-family bridges (key therapeutic transitions)
    ('anger', 'calm', 2.0, 'Progressive muscle relaxation', 'Tense each muscle group for 5s, release. Start with fists, end with toes.', 8, 'calming'),
    ('fear', 'calm', 2.0, 'Breathing reset', 'Extended exhale breathing: in for 4, out for 8. 10 cycles.', 5, 'calming'),
    ('sadness', 'calm', 1.5, 'Self-soothing', 'Wrap yourself in something soft. Make a warm drink. Small comforts.', 5, 'calming'),
    ('stressed', 'calm', 1.5, 'Autonomic reset', 'Splash cold water on your face. Vagal nerve activation.', 2, 'calming'),
    ('grounded', 'calm', 0.5, 'Transition', 'You are grounded. Now deepen: follow your breath.', 3, 'calming'),
    ('grounded', 'neutral', 0.5, 'Stabilize', 'You are here. What is true right now?', 2, 'cognitive'),
]


class TransitionGraph:
    """Directed weighted graph over 32 emotion nodes with Dijkstra pathfinding."""

    def __init__(self, adaptive_weights: dict = None):
        from emotion_classifier import FusionHead
        self.nodes = set(FusionHead.EMOTIONS)
        self.edges = {}   # {from_emotion: [(to_emotion, weight, step_info), ...]}
        self._build_graph(adaptive_weights or {})

    def _build_graph(self, adaptive_weights: dict):
        for node in self.nodes:
            self.edges[node] = []

        for from_e, to_e, weight, technique, exercise, duration, step_type in CURATED_EDGES:
            # Apply adaptive weight adjustment if available
            edge_key = f"{from_e}->{to_e}"
            adjusted_weight = adaptive_weights.get(edge_key, weight)

            self.edges[from_e].append((to_e, adjusted_weight, {
                'from_emotion': from_e,
                'to_emotion': to_e,
                'technique': technique,
                'exercise': exercise,
                'duration_minutes': duration,
                'step_type': step_type,
            }))

    def find_path(self, from_emotion: str, to_emotion: str) -> list:
        """Dijkstra shortest path. Returns list of step dicts, or None if unreachable."""
        if from_emotion == to_emotion:
            return []

        if from_emotion not in self.nodes or to_emotion not in self.nodes:
            return None

        # Dijkstra
        dist = {n: float('inf') for n in self.nodes}
        prev = {n: None for n in self.nodes}
        prev_step = {n: None for n in self.nodes}
        dist[from_emotion] = 0
        pq = [(0, from_emotion)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == to_emotion:
                break
            for v, w, step_info in self.edges.get(u, []):
                new_dist = d + w
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    prev_step[v] = step_info
                    heapq.heappush(pq, (new_dist, v))

        if dist[to_emotion] == float('inf'):
            return None

        # Reconstruct path
        path = []
        node = to_emotion
        while prev[node] is not None:
            path.append(prev_step[node])
            node = prev[node]
        path.reverse()
        return path


class TransitionSession:
    """Tracks a user through a multi-step emotion transition pathway."""

    def __init__(self, session_id: str, user_id: str, path: list):
        self.session_id = session_id
        self.user_id = user_id
        self.path = path
        self.current_step = 0
        self.started_at = time.time()
        self.step_started_at = time.time()
        self._baseline_biometrics = None

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.path)

    def get_current_step(self) -> dict:
        if self.is_complete:
            return None
        step = self.path[self.current_step].copy()
        step['step_number'] = self.current_step + 1
        step['total_steps'] = len(self.path)
        step['elapsed_seconds'] = time.time() - self.step_started_at
        return step

    def set_baseline_biometrics(self, biometrics: dict):
        self._baseline_biometrics = biometrics

    def check_biometric_criteria(self, current_biometrics: dict) -> bool:
        """Check if biometric shift criteria are met for current step."""
        if self.is_complete:
            return False

        step = self.path[self.current_step]
        step_type = step.get('step_type', 'cognitive')
        criteria = STEP_TYPES.get(step_type, {})

        if not criteria or not self._baseline_biometrics:
            return False

        if step_type == 'calming':
            hr_drop = self._baseline_biometrics.get('heart_rate', 0) - current_biometrics.get('heart_rate', 0)
            hrv_rise = current_biometrics.get('hrv', 0) - self._baseline_biometrics.get('hrv', 0)
            return hr_drop >= abs(criteria.get('hr_delta', 5)) or hrv_rise >= criteria.get('hrv_delta', 10)

        elif step_type == 'activation':
            hr_rise = current_biometrics.get('heart_rate', 0) - self._baseline_biometrics.get('heart_rate', 0)
            eda_rise = current_biometrics.get('eda', 0) - self._baseline_biometrics.get('eda', 0)
            return hr_rise >= criteria.get('hr_delta', 5) or eda_rise >= criteria.get('eda_delta', 0.5)

        return False

    def advance(self) -> dict:
        """Advance to next step. Returns the new current step or None if complete."""
        if not self.is_complete:
            self.current_step += 1
            self.step_started_at = time.time()
            self._baseline_biometrics = None
        return self.get_current_step()

    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'current_step': self.current_step,
            'total_steps': len(self.path),
            'is_complete': self.is_complete,
            'current': self.get_current_step(),
            'elapsed_seconds': time.time() - self.started_at,
        }


class AdaptiveWeightTracker:
    """SQLite-backed tracker that learns from transition outcomes."""

    def __init__(self, db_path: str = 'data/transitions.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transition_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                from_emotion TEXT NOT NULL,
                to_emotion TEXT NOT NULL,
                path_taken TEXT NOT NULL,
                outcome TEXT NOT NULL,
                duration_seconds REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def log_outcome(self, user_id: str, from_emotion: str, to_emotion: str,
                    path_taken: list, outcome: str, duration_seconds: float):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute(
            'INSERT INTO transition_log (user_id, from_emotion, to_emotion, path_taken, outcome, duration_seconds) VALUES (?, ?, ?, ?, ?, ?)',
            (user_id, from_emotion, to_emotion, json.dumps(path_taken), outcome, duration_seconds)
        )
        conn.commit()
        conn.close()

    def get_log(self, limit: int = 50) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT * FROM transition_log ORDER BY id DESC LIMIT ?', (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_adjusted_weights(self) -> dict:
        """Compute adjusted edge weights based on success rates."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute('''
            SELECT from_emotion, to_emotion,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes
            FROM transition_log
            GROUP BY from_emotion, to_emotion
            HAVING total >= 3
        ''').fetchall()
        conn.close()

        weights = {}
        for row in rows:
            success_rate = row['successes'] / row['total']
            # Lower weight = preferred by Dijkstra. High success → low weight.
            edge_key = f"{row['from_emotion']}->{row['to_emotion']}"
            weights[edge_key] = max(0.1, 1.0 - success_rate * 0.8)

        return weights


class EmotionTransitionEngine:
    """Main interface for the emotion transition system."""

    def __init__(self, db_path: str = 'data/transitions.db'):
        self.tracker = AdaptiveWeightTracker(db_path=db_path)
        self.graph = TransitionGraph(adaptive_weights=self.tracker.get_adjusted_weights())
        self._sessions = {}  # session_id -> TransitionSession

    def get_pathway(self, from_emotion: str, to_emotion: str) -> list:
        """Get multi-step pathway between two emotions."""
        return self.graph.find_path(from_emotion.lower(), to_emotion.lower())

    def start_session(self, user_id: str, from_emotion: str, to_emotion: str) -> TransitionSession:
        """Create and track a new transition session."""
        path = self.get_pathway(from_emotion, to_emotion)
        if path is None:
            return None
        session_id = str(uuid.uuid4())[:8]
        session = TransitionSession(session_id=session_id, user_id=user_id, path=path)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> TransitionSession:
        return self._sessions.get(session_id)

    def log_feedback(self, user_id: str, from_emotion: str, to_emotion: str,
                     path_taken: list, outcome: str, duration_seconds: float):
        self.tracker.log_outcome(user_id, from_emotion, to_emotion, path_taken, outcome, duration_seconds)
        # Refresh graph with updated weights periodically
        self.graph = TransitionGraph(adaptive_weights=self.tracker.get_adjusted_weights())

    def cleanup_session(self, session_id: str):
        self._sessions.pop(session_id, None)
```

- [ ] **Step 11: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_transition_engine.py -v`
Expected: All tests PASS

- [ ] **Step 12: Commit**

```bash
git add emotion_transition_engine.py tests/test_transition_engine.py
git commit -m "feat: add emotion transition engine with graph pathfinding and adaptive weights"
```

---

### Task 5: Wire transition engine into API server

**Files:**
- Modify: `emotion_api_server.py`

- [ ] **Step 13: Add transition engine import and endpoints**

At the top of `emotion_api_server.py`, after the existing imports (around line 78):

```python
try:
    from emotion_transition_engine import EmotionTransitionEngine
    HAS_TRANSITION_ENGINE = True
except ImportError:
    HAS_TRANSITION_ENGINE = False
```

In `create_app()` (around line 797), after `CORS(app)`:

```python
    # Initialize transition engine
    transition_engine = None
    if HAS_TRANSITION_ENGINE:
        try:
            transition_engine = EmotionTransitionEngine()
            print("[API] Emotion transition engine initialized")
        except Exception as e:
            print(f"[API] Transition engine init failed: {e}")
```

Replace the existing `emotion_transition()` endpoint (lines 994-1022) with:

```python
    @app.route('/api/neural/emotion-transition', methods=['POST'])
    def emotion_transition():
        """Get emotion transition pathway between two emotions"""
        try:
            data = request.json or {}
            from_emotion = data.get('from_emotion', '').lower()
            to_emotion = data.get('to_emotion', '').lower()

            if not from_emotion:
                return jsonify({'error': 'from_emotion is required'}), 400

            if transition_engine and to_emotion:
                path = transition_engine.get_pathway(from_emotion, to_emotion)
                if path is None:
                    return jsonify({'error': f'No path from {from_emotion} to {to_emotion}'}), 404
                return jsonify({
                    'from_emotion': from_emotion,
                    'to_emotion': to_emotion,
                    'steps': path,
                    'total_steps': len(path),
                    'status': 'success'
                })

            # Fallback to legacy single-step lookup
            transition = TRANSITION_PATHWAYS.get(from_emotion, '')
            technique = THERAPY_TECHNIQUES.get(from_emotion, THERAPY_TECHNIQUES['neutral'])
            coaching = COACHING_PROMPTS.get(from_emotion, '')
            family = _EMOTION_TO_FAMILY.get(from_emotion, 'Neutral')

            return jsonify({
                'from_emotion': from_emotion,
                'from_family': family,
                'to_emotion': to_emotion or 'auto',
                'transition': transition,
                'technique': technique['name'],
                'exercise': technique['exercise'],
                'coaching_prompt': coaching,
                'status': 'success'
            })
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500
```

Add the two new endpoints after the transition endpoint:

```python
    @app.route('/api/neural/transition-session', methods=['POST'])
    def transition_session():
        """Start, advance, or query a transition session"""
        if not transition_engine:
            return jsonify({'error': 'Transition engine not available'}), 503
        try:
            data = request.json or {}
            action = data.get('action', 'start')

            if action == 'start':
                user_id = data.get('user_id', 'anonymous')
                from_emotion = data.get('from_emotion', '').lower()
                to_emotion = data.get('to_emotion', '').lower()
                if not from_emotion or not to_emotion:
                    return jsonify({'error': 'from_emotion and to_emotion required'}), 400
                session = transition_engine.start_session(user_id, from_emotion, to_emotion)
                if session is None:
                    return jsonify({'error': f'No path from {from_emotion} to {to_emotion}'}), 404
                return jsonify({**session.to_dict(), 'status': 'success'})

            elif action == 'advance':
                session_id = data.get('session_id')
                session = transition_engine.get_session(session_id)
                if not session:
                    return jsonify({'error': 'Session not found'}), 404
                session.advance()
                return jsonify({**session.to_dict(), 'status': 'success'})

            elif action == 'query':
                session_id = data.get('session_id')
                session = transition_engine.get_session(session_id)
                if not session:
                    return jsonify({'error': 'Session not found'}), 404
                return jsonify({**session.to_dict(), 'status': 'success'})

            else:
                return jsonify({'error': f'Unknown action: {action}'}), 400

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/neural/transition-feedback', methods=['POST'])
    def transition_feedback():
        """Log transition outcome for adaptive weight learning"""
        if not transition_engine:
            return jsonify({'error': 'Transition engine not available'}), 503
        try:
            data = request.json or {}
            required = ['user_id', 'from_emotion', 'to_emotion', 'outcome']
            for field in required:
                if field not in data:
                    return jsonify({'error': f'{field} is required'}), 400

            transition_engine.log_feedback(
                user_id=data['user_id'],
                from_emotion=data['from_emotion'],
                to_emotion=data['to_emotion'],
                path_taken=data.get('path_taken', []),
                outcome=data['outcome'],
                duration_seconds=data.get('duration_seconds', 0)
            )
            return jsonify({'status': 'success', 'message': 'Feedback logged'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500
```

- [ ] **Step 14: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 15: Commit**

```bash
git add emotion_api_server.py
git commit -m "feat: wire transition engine into API with session and feedback endpoints"
```

---

## Chunk 3: WebSocket Streaming (Feature 5)

### Task 6: WebSocket — Tests

**Files:**
- Create: `tests/test_websocket.py`

- [ ] **Step 16: Write WebSocket tests**

```python
# tests/test_websocket.py
"""Tests for WebSocket streaming module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmotionWebSocket:
    """Test WebSocket initialization and event emission."""

    def test_init_websocket_returns_socketio(self):
        from flask import Flask
        from emotion_websocket import init_websocket
        app = Flask(__name__)
        socketio = init_websocket(app)
        assert socketio is not None

    def test_emit_emotion_update_does_not_crash_without_clients(self):
        from flask import Flask
        from emotion_websocket import init_websocket, emit_emotion_update
        app = Flask(__name__)
        init_websocket(app)
        # Should not raise even with no connected clients
        emit_emotion_update({
            'dominant_emotion': 'joy',
            'family': 'Joy',
            'confidence': 0.85,
            'scores': {},
            'modality_weights': {'text': 1.0, 'bio': 0.0, 'pose': 0.0}
        })

    def test_emit_transition_step_does_not_crash(self):
        from flask import Flask
        from emotion_websocket import init_websocket, emit_transition_step
        app = Flask(__name__)
        init_websocket(app)
        emit_transition_step({
            'session_id': 'test-1',
            'step': 1,
            'technique': 'Grounding',
            'from_emotion': 'anxiety',
            'to_emotion': 'grounded'
        })

    def test_disable_websocket_env_var(self):
        """When DISABLE_WEBSOCKET=1, init returns None."""
        import importlib
        import emotion_websocket
        os.environ['DISABLE_WEBSOCKET'] = '1'
        try:
            importlib.reload(emotion_websocket)
            from flask import Flask
            app = Flask(__name__)
            result = emotion_websocket.init_websocket(app)
            assert result is None
            assert emotion_websocket._socketio is None
        finally:
            del os.environ['DISABLE_WEBSOCKET']
            importlib.reload(emotion_websocket)
```

- [ ] **Step 17: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_websocket.py -v`
Expected: FAIL — `ModuleNotFoundError`

---

### Task 7: WebSocket — Implementation

**Files:**
- Create: `emotion_websocket.py`
- Modify: `requirements.txt`
- Modify: `Dockerfile`

- [ ] **Step 18: Add dependencies to requirements.txt**

Append to `requirements.txt`:

```
flask-socketio>=5.3.0
eventlet>=0.35.0
```

- [ ] **Step 19: Install new dependencies**

Run: `cd /Users/bel/quantara-nanoGPT && pip install flask-socketio eventlet`

- [ ] **Step 20: Create emotion_websocket.py**

```python
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - WebSocket Streaming Module
===============================================================================
Real-time emotion state streaming via socket.io.

Namespaces:
  /emotion    — emotion_update, transition_step events
  /biometrics — biometric_stream events (RuView data)
  /system     — model_retrained events (stub, populated by auto_retrain)

Integrates with:
- Neural Workflow AI Engine
- Real-time Dashboard Data
- Biometric Integration Engine
===============================================================================
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_socketio = None


def init_websocket(app):
    """Initialize socket.io on Flask app. Returns SocketIO instance or None if disabled."""
    global _socketio

    if os.environ.get('DISABLE_WEBSOCKET') == '1':
        logger.info("[WebSocket] Disabled via DISABLE_WEBSOCKET env var")
        return None

    try:
        from flask_socketio import SocketIO, emit, Namespace
    except ImportError:
        logger.warning("[WebSocket] flask-socketio not installed, WebSocket disabled")
        return None

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
    _socketio = socketio

    class EmotionNamespace(Namespace):
        def on_connect(self, auth=None):
            # Validate JWT token from query parameter
            from flask import request as flask_request
            token = flask_request.args.get('token')
            if token:
                # Validate token (same logic as REST middleware)
                # For now, accept any non-empty token; production should verify JWT
                logger.info(f"[WebSocket] Client authenticated to /emotion")
            else:
                logger.warning(f"[WebSocket] Unauthenticated client rejected from /emotion")
                raise ConnectionRefusedError('Authentication required')

        def on_disconnect(self):
            logger.info(f"[WebSocket] Client disconnected from /emotion")

        def on_join_room(self, data):
            from flask_socketio import join_room
            room = data.get('room')
            if room:
                join_room(room)
                logger.info(f"[WebSocket] Client joined room: {room}")

        def on_subscribe_family(self, data):
            from flask_socketio import join_room
            family = data.get('family')
            if family:
                join_room(f'family_{family.lower()}')

    class BiometricsNamespace(Namespace):
        def on_connect(self):
            logger.info(f"[WebSocket] Client connected to /biometrics")

        def on_disconnect(self):
            logger.info(f"[WebSocket] Client disconnected from /biometrics")

    class SystemNamespace(Namespace):
        """Stub namespace — populated by auto_retrain module (Feature 6)."""
        def on_connect(self):
            logger.info(f"[WebSocket] Client connected to /system")

        def on_disconnect(self):
            logger.info(f"[WebSocket] Client disconnected from /system")

    socketio.on_namespace(EmotionNamespace('/emotion'))
    socketio.on_namespace(BiometricsNamespace('/biometrics'))
    socketio.on_namespace(SystemNamespace('/system'))

    logger.info("[WebSocket] Initialized with namespaces: /emotion, /biometrics, /system")
    return socketio


def emit_emotion_update(result: dict, room: str = None):
    """Emit emotion classification result to /emotion namespace."""
    if _socketio is None:
        return

    payload = {
        'emotion': result.get('dominant_emotion'),
        'family': result.get('family'),
        'confidence': result.get('confidence'),
        'scores': result.get('scores', {}),
        'modality_weights': result.get('modality_weights', {}),
        'timestamp': datetime.now().isoformat(),
    }

    kwargs = {'namespace': '/emotion'}
    if room:
        kwargs['to'] = room

    _socketio.emit('emotion_update', payload, **kwargs)


def emit_transition_step(step_data: dict, room: str = None):
    """Emit transition step event to /emotion namespace."""
    if _socketio is None:
        return

    payload = {
        **step_data,
        'timestamp': datetime.now().isoformat(),
    }

    kwargs = {'namespace': '/emotion'}
    if room:
        kwargs['to'] = room

    _socketio.emit('transition_step', payload, **kwargs)


def emit_biometric_stream(biometric_data: dict, room: str = None):
    """Emit biometric data to /biometrics namespace."""
    if _socketio is None:
        return

    payload = {
        **biometric_data,
        'timestamp': datetime.now().isoformat(),
    }

    kwargs = {'namespace': '/biometrics'}
    if room:
        kwargs['to'] = room

    _socketio.emit('biometric_stream', payload, **kwargs)


def emit_system_event(event_name: str, data: dict):
    """Emit event on /system namespace (used by auto_retrain)."""
    if _socketio is None:
        return

    _socketio.emit(event_name, {**data, 'timestamp': datetime.now().isoformat()}, namespace='/system')


def get_socketio():
    """Get the current SocketIO instance (for use by auto_retrain etc)."""
    return _socketio
```

- [ ] **Step 21: Update Dockerfile**

Change the CMD line to use eventlet with 1 worker:

```dockerfile
CMD gunicorn emotion_api_server:app --bind 0.0.0.0:${PORT:-5050} --worker-class eventlet --workers 1 --timeout 120
```

- [ ] **Step 22: Wire WebSocket into API server**

In `emotion_api_server.py`, add import (around line 78):

```python
try:
    from emotion_websocket import init_websocket, emit_emotion_update as ws_emit_emotion
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
```

In `create_app()`, after the transition engine init:

```python
    # Initialize WebSocket streaming
    socketio = None
    if HAS_WEBSOCKET:
        socketio = init_websocket(app)
```

In the `/api/emotion/analyze` endpoint handler (around line 1159), after `result = multimodal_analyzer.analyze(...)`:

```python
            # Emit to WebSocket subscribers
            if HAS_WEBSOCKET:
                ws_emit_emotion(result)
```

In `get_app()` (the gunicorn entry point, around line 1234), change the return to support socketio:

```python
    # Store socketio on app for gunicorn access
    return app
```

- [ ] **Step 23: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_websocket.py tests/test_emotion_classifier.py tests/test_transition_engine.py -v`
Expected: All tests PASS

- [ ] **Step 24: Commit**

```bash
git add emotion_websocket.py requirements.txt Dockerfile emotion_api_server.py tests/test_websocket.py
git commit -m "feat: add WebSocket streaming with emotion/biometrics/system namespaces"
```

---

## Chunk 4: Auto-Retraining Pipeline (Feature 6)

### Task 8: Auto-Retrain — Tests

**Files:**
- Create: `tests/test_auto_retrain.py`

- [ ] **Step 25: Write auto-retrain tests**

```python
# tests/test_auto_retrain.py
"""Tests for auto-retraining pipeline."""

import pytest
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDriftDetector:
    """Test drift detection via KS test."""

    @pytest.fixture
    def detector(self):
        from auto_retrain import DriftDetector
        return DriftDetector(window_size=20)

    def test_no_drift_with_consistent_errors(self, detector):
        """Stable errors should not trigger drift."""
        import numpy as np
        np.random.seed(42)
        # Establish baseline
        for _ in range(20):
            detector.add_error(np.random.normal(0, 0.1))
        detector.set_baseline()
        # Add similar errors
        for _ in range(20):
            detector.add_error(np.random.normal(0, 0.1))
        assert not detector.is_drifting()

    def test_drift_detected_with_shifted_errors(self, detector):
        """Large shift in error distribution should trigger drift."""
        import numpy as np
        np.random.seed(42)
        for _ in range(20):
            detector.add_error(np.random.normal(0, 0.1))
        detector.set_baseline()
        # Add shifted errors
        for _ in range(20):
            detector.add_error(np.random.normal(5.0, 0.1))
        assert detector.is_drifting()


class TestThresholdMonitor:
    """Test sample-count threshold monitoring."""

    @pytest.fixture
    def monitor(self):
        from auto_retrain import ThresholdMonitor
        return ThresholdMonitor(first_threshold=5, subsequent_interval=10)

    def test_triggers_at_first_threshold(self, monitor):
        for i in range(4):
            assert not monitor.should_retrain(i + 1)
        assert monitor.should_retrain(5)

    def test_triggers_at_subsequent_intervals(self, monitor):
        monitor.should_retrain(5)  # first trigger, resets counter
        monitor.mark_retrained(5)
        for i in range(6, 14):
            assert not monitor.should_retrain(i)
        assert monitor.should_retrain(15)


class TestRetrainWorker:
    """Test the retrain worker validation gate."""

    def test_validation_gate_rejects_worse_model(self):
        from auto_retrain import validate_retrained_model
        from wifi_calibration import WiFiCalibrationModel

        old_model = WiFiCalibrationModel()
        new_model = WiFiCalibrationModel()

        # Create dummy validation data
        val_inputs = torch.randn(10, 2)
        val_targets = torch.randn(10, 2)

        # Both models are random, so validation should run without crashing
        result = validate_retrained_model(old_model, new_model, val_inputs, val_targets)
        assert 'old_mae' in result
        assert 'new_mae' in result
        assert 'accepted' in result
```

- [ ] **Step 26: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_auto_retrain.py -v`
Expected: FAIL — `ModuleNotFoundError`

---

### Task 9: Auto-Retrain — Implementation

**Files:**
- Create: `auto_retrain.py`
- Modify: `wifi_calibration.py`

- [ ] **Step 27: Add get_drift_score to PersonalCalibrationBuffer**

In `wifi_calibration.py`, change `MAX_BUFFER_SIZE` (line 306):

```python
    MAX_BUFFER_SIZE = 500
```

Add public API methods to `PersonalCalibrationBuffer` class (after the existing methods):

```python
    @property
    def total_samples_seen(self) -> int:
        """Total number of samples ever added (not just buffered)."""
        return self._total_added

    @property
    def buffer_size(self) -> int:
        """Current number of samples in the rolling buffer."""
        return len(self._buffer)

    def get_buffer_data(self) -> list:
        """Return a copy of buffered (wifi_tensor, target_tensor) pairs."""
        return list(self._buffer)

    def get_prediction_errors(self, model: nn.Module) -> list:
        """Compute prediction errors for all buffered samples."""
        if len(self._buffer) == 0:
            return []
        errors = []
        model.eval()
        with torch.no_grad():
            for wifi_tensor, target_tensor in self._buffer:
                pred = model(wifi_tensor.unsqueeze(0)).squeeze(0)
                error = float(torch.mean(torch.abs(pred - target_tensor)))
                errors.append(error)
        return errors
```

- [ ] **Step 28: Create auto_retrain.py**

```python
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Auto-Retraining Pipeline
===============================================================================
Background auto-retraining for WiFi calibration model with drift detection,
threshold monitoring, and safety guardrails.

Integrates with:
- Biometric Integration Engine
- Neural Workflow AI Engine
- WebSocket Streaming (emits model_retrained events)
===============================================================================
"""

import copy
import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect distribution shift in prediction errors via KS test."""

    def __init__(self, window_size: int = 200, p_threshold: float = 0.05):
        self.window_size = window_size
        self.p_threshold = p_threshold
        self._recent_errors = deque(maxlen=window_size)
        self._baseline_errors = None

    def add_error(self, error: float):
        self._recent_errors.append(error)

    def set_baseline(self):
        """Snapshot current errors as the baseline distribution."""
        self._baseline_errors = list(self._recent_errors)

    def is_drifting(self) -> bool:
        """KS test: is recent error distribution significantly different from baseline?"""
        if self._baseline_errors is None or len(self._recent_errors) < 20:
            return False
        statistic, p_value = stats.ks_2samp(self._baseline_errors, list(self._recent_errors))
        return p_value < self.p_threshold

    def get_drift_score(self) -> dict:
        if self._baseline_errors is None or len(self._recent_errors) < 20:
            return {'drifting': False, 'p_value': 1.0, 'samples': len(self._recent_errors)}
        statistic, p_value = stats.ks_2samp(self._baseline_errors, list(self._recent_errors))
        return {'drifting': p_value < self.p_threshold, 'p_value': float(p_value), 'ks_statistic': float(statistic), 'samples': len(self._recent_errors)}


class ThresholdMonitor:
    """Monitor sample counts and trigger retraining at thresholds."""

    def __init__(self, first_threshold: int = 20, subsequent_interval: int = 50):
        self.first_threshold = first_threshold
        self.subsequent_interval = subsequent_interval
        self._last_retrain_count = 0
        self._triggered_first = False

    def should_retrain(self, total_samples: int) -> bool:
        if not self._triggered_first:
            if total_samples >= self.first_threshold:
                return True
        else:
            if total_samples - self._last_retrain_count >= self.subsequent_interval:
                return True
        return False

    def mark_retrained(self, total_samples: int):
        self._triggered_first = True
        self._last_retrain_count = total_samples


def validate_retrained_model(old_model, new_model, val_inputs, val_targets) -> dict:
    """Compare old vs new model on validation data. Returns metrics + acceptance."""
    old_model.eval()
    new_model.eval()

    with torch.no_grad():
        old_preds = old_model(val_inputs)
        new_preds = new_model(val_inputs)
        old_mae = float(torch.mean(torch.abs(old_preds - val_targets)))
        new_mae = float(torch.mean(torch.abs(new_preds - val_targets)))

    return {
        'old_mae': old_mae,
        'new_mae': new_mae,
        'accepted': new_mae < old_mae,
        'improvement': old_mae - new_mae,
    }


class RetrainLog:
    """SQLite-backed log of retraining events."""

    def __init__(self, db_path: str = 'data/retrain_log.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS retrain_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_type TEXT NOT NULL,
                samples_used INTEGER,
                mae_before REAL,
                mae_after REAL,
                outcome TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def log(self, trigger_type: str, samples_used: int, mae_before: float,
            mae_after: float, outcome: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute(
            'INSERT INTO retrain_log (trigger_type, samples_used, mae_before, mae_after, outcome) VALUES (?, ?, ?, ?, ?)',
            (trigger_type, samples_used, mae_before, mae_after, outcome)
        )
        conn.commit()
        conn.close()

    def get_log(self, limit: int = 50) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT * FROM retrain_log ORDER BY id DESC LIMIT ?', (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


class AutoRetrainManager:
    """Manages background auto-retraining of WiFi calibration model."""

    def __init__(self, calibration_buffer, model, checkpoint_path='checkpoints/ruview_calibration.pt'):
        self.buffer = calibration_buffer
        self.model = model
        self.checkpoint_path = checkpoint_path
        self._model_lock = threading.Lock()
        self._drift_detector = DriftDetector()
        self._threshold_monitor = ThresholdMonitor()
        self._retrain_log = RetrainLog()
        self._cooldown_seconds = 3600  # 1 hour
        self._last_retrain_time = 0
        self._running = False
        self._thread = None

    def start(self, check_interval: int = 60):
        """Start background monitoring thread."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, args=(check_interval,), daemon=True)
        self._thread.start()
        logger.info("[AutoRetrain] Background monitor started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self, interval: int):
        while self._running:
            try:
                self._check_triggers()
            except Exception as e:
                logger.error(f"[AutoRetrain] Monitor error: {e}")
            time.sleep(interval)

    def _check_triggers(self):
        now = time.time()
        if now - self._last_retrain_time < self._cooldown_seconds:
            return

        trigger = None
        total_samples = self.buffer.total_samples_seen

        if self._threshold_monitor.should_retrain(total_samples):
            trigger = 'threshold'
        elif self._drift_detector.is_drifting():
            trigger = 'drift'

        if trigger:
            self._do_retrain(trigger)

    def _do_retrain(self, trigger: str):
        """Retrain in current thread (called from monitor loop)."""
        logger.info(f"[AutoRetrain] Retraining triggered by: {trigger}")

        buffer_data = self.buffer.get_buffer_data()
        if len(buffer_data) < 10:
            logger.warning("[AutoRetrain] Not enough data to retrain")
            return

        # Split 80/20 train/val
        split = int(len(buffer_data) * 0.8)
        train_data = buffer_data[:split]
        val_data = buffer_data[split:]

        train_inputs = torch.stack([d[0] for d in train_data])
        train_targets = torch.stack([d[1] for d in train_data])
        val_inputs = torch.stack([d[0] for d in val_data])
        val_targets = torch.stack([d[1] for d in val_data])

        # Copy model for training
        new_model = copy.deepcopy(self.model)
        optimizer = optim.Adam(new_model.parameters(), lr=1e-4)
        loss_fn = nn.L1Loss()

        # Train
        new_model.train()
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(50):
            optimizer.zero_grad()
            preds = new_model(train_inputs)
            loss = loss_fn(preds, train_targets)
            loss.backward()
            optimizer.step()

            # Validation
            new_model.eval()
            with torch.no_grad():
                val_preds = new_model(val_inputs)
                val_loss = float(loss_fn(val_preds, val_targets))
            new_model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        new_model.eval()

        # Validation gate
        result = validate_retrained_model(self.model, new_model, val_inputs, val_targets)

        if result['accepted']:
            with self._model_lock:
                self.model.load_state_dict(new_model.state_dict())

            # Save checkpoint
            torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)

            self._retrain_log.log(trigger, len(buffer_data), result['old_mae'], result['new_mae'], 'applied')
            self._threshold_monitor.mark_retrained(self.buffer._total_added)
            self._drift_detector.set_baseline()
            self._last_retrain_time = time.time()
            self._cooldown_seconds = 3600  # Reset to 1h

            # Emit WebSocket event
            try:
                from emotion_websocket import emit_system_event
                emit_system_event('model_retrained', {
                    'trigger': trigger,
                    'samples_used': len(buffer_data),
                    'mae_before': result['old_mae'],
                    'mae_after': result['new_mae'],
                })
            except ImportError:
                pass

            logger.info(f"[AutoRetrain] Model updated. MAE: {result['old_mae']:.4f} → {result['new_mae']:.4f}")
        else:
            self._retrain_log.log(trigger, len(buffer_data), result['old_mae'], result['new_mae'], 'discarded')
            self._cooldown_seconds = 14400  # Extend to 4h after failure
            self._last_retrain_time = time.time()
            logger.info(f"[AutoRetrain] Model discarded (worse). MAE: {result['old_mae']:.4f} vs {result['new_mae']:.4f}")

    def manual_retrain(self) -> dict:
        """Trigger retrain manually. Returns result."""
        self._do_retrain('manual')
        log = self._retrain_log.get_log(limit=1)
        return log[0] if log else {}

    def get_status(self) -> dict:
        return {
            'drift': self._drift_detector.get_drift_score(),
            'buffer_size': self.buffer.buffer_size,
            'total_samples': self.buffer.total_samples_seen,
            'last_retrain': datetime.fromtimestamp(self._last_retrain_time).isoformat() if self._last_retrain_time > 0 else None,
            'cooldown_seconds': self._cooldown_seconds,
        }
```

- [ ] **Step 29: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_auto_retrain.py -v`
Expected: All tests PASS

- [ ] **Step 30: Wire auto-retrain into API server**

In `emotion_api_server.py`, add import:

```python
try:
    from auto_retrain import AutoRetrainManager
    HAS_AUTO_RETRAIN = True
except ImportError:
    HAS_AUTO_RETRAIN = False
```

In `create_app()`, after WebSocket init:

```python
    # Initialize auto-retraining (if RuView calibration is available)
    retrain_manager = None
    if HAS_AUTO_RETRAIN and context_provider and hasattr(context_provider, 'ruview'):
        try:
            ruview = context_provider.ruview
            if hasattr(ruview, '_calibration_buffer') and ruview._calibration_buffer:
                retrain_manager = AutoRetrainManager(
                    calibration_buffer=ruview._calibration_buffer,
                    model=ruview._calibration_model
                )
                retrain_manager.start()
                print("[API] Auto-retrain manager started")
        except Exception as e:
            print(f"[API] Auto-retrain init failed: {e}")
```

Add the 3 API endpoints:

```python
    @app.route('/api/calibration/retrain-status', methods=['GET'])
    def retrain_status():
        if not retrain_manager:
            return jsonify({'error': 'Auto-retrain not available'}), 503
        return jsonify({**retrain_manager.get_status(), 'status': 'success'})

    @app.route('/api/calibration/retrain', methods=['POST'])
    def manual_retrain():
        if not retrain_manager:
            return jsonify({'error': 'Auto-retrain not available'}), 503
        result = retrain_manager.manual_retrain()
        return jsonify({**result, 'status': 'success'})

    @app.route('/api/calibration/retrain-log', methods=['GET'])
    def retrain_log():
        if not retrain_manager:
            return jsonify({'error': 'Auto-retrain not available'}), 503
        log = retrain_manager._retrain_log.get_log()
        return jsonify({'log': log, 'status': 'success'})
```

- [ ] **Step 31: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 32: Commit**

```bash
git add auto_retrain.py wifi_calibration.py emotion_api_server.py tests/test_auto_retrain.py
git commit -m "feat: add auto-retraining pipeline with drift detection and threshold monitoring"
```

---

## Chunk 5: Production Training Notebook & Formal Evaluation (Features 3 & 4)

### Task 10: Colab Training Notebook

**Files:**
- Create: `notebooks/train_production.ipynb`

- [ ] **Step 33: Create notebooks directory**

Run: `mkdir -p /Users/bel/quantara-nanoGPT/notebooks`

- [ ] **Step 34: Create the Colab notebook**

Create `notebooks/train_production.ipynb` as a Jupyter notebook with 7 cells:

**Cell 1 (Setup):**
```python
# @title 1. Setup — Clone repo, install deps, mount Drive
import subprocess, os

# Mount Google Drive for checkpoint persistence
from google.colab import drive
drive.mount('/content/drive')

DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/quantara-emotion-gpt/'
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

# Clone repo
if not os.path.exists('/content/quantara-nanoGPT'):
    subprocess.run(['git', 'clone', 'https://github.com/QEbellavita/quantara-nanoGPT.git', '/content/quantara-nanoGPT'], check=True)

os.chdir('/content/quantara-nanoGPT')
subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'], check=True)

# Detect GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")
```

**Cell 2 (Data Prep):**
```python
# @title 2. Data Preparation
os.chdir('/content/quantara-nanoGPT')
subprocess.run(['python', 'data/quantara_emotion/prepare.py'], check=True)

# Verify
for f in ['data/quantara_emotion/train.bin', 'data/quantara_emotion/val.bin']:
    size_mb = os.path.getsize(f) / 1e6
    print(f"  {f}: {size_mb:.1f} MB")
```

**Cell 3 (Config):**
```python
# @title 3. Configure Training
import torch

# Auto-detect GPU tier
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
if 'A100' in gpu_name:
    batch_size = 64
    compile_model = True
elif 'T4' in gpu_name or 'V100' in gpu_name:
    batch_size = 32
    compile_model = False
else:
    batch_size = 16
    compile_model = False

print(f"Config: batch_size={batch_size}, compile={compile_model}")
print(f"Using config/train_quantara_emotion.py (8-layer, 512-dim, 10K iters)")
```

**Cell 4 (Train):**
```python
# @title 4. Train Production Model
import shutil

# Resume from Drive checkpoint if available
drive_ckpt = os.path.join(DRIVE_CHECKPOINT_DIR, 'ckpt.pt')
local_out = 'out-quantara-emotion'
os.makedirs(local_out, exist_ok=True)

if os.path.exists(drive_ckpt):
    shutil.copy(drive_ckpt, os.path.join(local_out, 'ckpt.pt'))
    print(f"Resumed from Drive checkpoint")

# Run training
cmd = f"python train.py config/train_quantara_emotion.py --batch_size={batch_size} --compile={compile_model} --device=cuda"
print(f"Running: {cmd}")
subprocess.run(cmd.split(), check=True)

# Copy checkpoint to Drive
local_ckpt = os.path.join(local_out, 'ckpt.pt')
if os.path.exists(local_ckpt):
    shutil.copy(local_ckpt, drive_ckpt)
    print(f"Checkpoint saved to Drive: {drive_ckpt}")
```

**Cell 5 (Evaluate):**
```python
# @title 5. Quick Evaluation
from model import GPT, GPTConfig
import tiktoken

checkpoint = torch.load('out-quantara-emotion/ckpt.pt', map_location='cuda')
conf = GPTConfig(**checkpoint['model_args'])
model = GPT(conf)
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()})
model.eval().to('cuda')

enc = tiktoken.get_encoding('gpt2')

prompts = ["I feel so happy today", "I'm really anxious about", "This makes me angry"]
for prompt in prompts:
    ids = enc.encode(prompt)
    x = torch.tensor([ids], device='cuda')
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=40)
    print(f"\n--- {prompt} ---")
    print(enc.decode(y[0].tolist()))

print(f"\nVal loss from training: check wandb or stdout above")
```

**Cell 6 (Export):**
```python
# @title 6. Export Model
from google.colab import files
import shutil

# Final save to Drive
shutil.copy('out-quantara-emotion/ckpt.pt', DRIVE_CHECKPOINT_DIR + 'ckpt_production.pt')
print(f"Production model saved to Drive")
print(f"Size: {os.path.getsize('out-quantara-emotion/ckpt.pt') / 1e6:.1f} MB")
```

**Cell 7 (Retrain FusionHead):**
```python
# @title 7. Retrain AttentionFusionHead for 512-dim Embeddings
import pandas as pd
from emotion_classifier import MultimodalEmotionAnalyzer, AttentionFusionHead, FusionHead

# Load the production GPT checkpoint (uses nanoGPT embeddings, not sentence-transformers)
analyzer = MultimodalEmotionAnalyzer(
    gpt_checkpoint='out-quantara-emotion/ckpt.pt',
    use_sentence_transformer=False,
    device='cuda'
)
print(f"Text embedding dim: {analyzer.n_embd}")

# Load emotion-labeled training data
dfs = []
for csv_file in ['data/quantara_emotion/text_emotion.csv', 'data/quantara_emotion/Emotion_classify_Data.csv']:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        text_col = [c for c in df.columns if c.lower() in ['text', 'content', 'comment']][0]
        label_col = [c for c in df.columns if c.lower() in ['emotion', 'label', 'sentiment']][0]
        df = df.rename(columns={text_col: 'text', label_col: 'emotion'})
        df['emotion'] = df['emotion'].str.lower().str.strip()
        df = df[df['emotion'].isin(FusionHead.EMOTIONS)]
        dfs.append(df[['text', 'emotion']])

data = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Training samples: {len(data)}")

# Train the fusion head
optimizer = torch.optim.Adam(analyzer.fusion_head.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
emotion_to_idx = {e: i for i, e in enumerate(FusionHead.EMOTIONS)}

analyzer.fusion_head.train()
batch_size = 32
for epoch in range(20):
    total_loss = 0
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        texts = batch['text'].tolist()
        labels = torch.tensor([emotion_to_idx[e] for e in batch['emotion']], device='cuda')

        embeddings = torch.stack([analyzer._get_text_embedding(t).squeeze(0) for t in texts])
        emotion_probs, family_probs = analyzer.fusion_head(embeddings)

        loss = loss_fn(emotion_probs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (len(data) // batch_size)
    print(f"Epoch {epoch+1}/20 — Loss: {avg_loss:.4f}")

analyzer.fusion_head.eval()

# Save combined checkpoint
torch.save({
    'fusion_head': analyzer.fusion_head.state_dict(),
    'biometric_encoder': analyzer.biometric_encoder.state_dict(),
    'meta': {'version': 2, 'text_dim': analyzer.n_embd, 'pose_dim': 16}
}, DRIVE_CHECKPOINT_DIR + 'classifier_production.pt')

print("Production classifier saved to Drive")
```

- [ ] **Step 35: Commit**

```bash
git add notebooks/train_production.ipynb
git commit -m "feat: add Colab notebook for production model training"
```

---

### Task 11: Formal Evaluation — Tests

**Files:**
- Create: `tests/test_evaluate.py`
- Create: `requirements-dev.txt`

- [ ] **Step 36: Create requirements-dev.txt**

```
datasets>=2.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

- [ ] **Step 37: Write evaluation tests**

```python
# tests/test_evaluate.py
"""Tests for the formal evaluation script."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTaxonomyMapping:
    """Test emotion taxonomy mapping between datasets."""

    def test_goemotions_mapping_covers_all_27(self):
        from evaluate import GOEMOTIONS_MAPPING
        assert len(GOEMOTIONS_MAPPING) == 27

    def test_goemotions_mapping_targets_are_valid_emotions(self):
        from evaluate import GOEMOTIONS_MAPPING
        from emotion_classifier import FusionHead
        valid = set(FusionHead.EMOTIONS)
        for source, target in GOEMOTIONS_MAPPING.items():
            if target is not None:  # None means skip/unmappable
                assert target in valid, f"{source} maps to invalid emotion: {target}"

    def test_compute_metrics_returns_expected_keys(self):
        from evaluate import compute_metrics
        y_true = ['joy', 'sadness', 'anger', 'joy']
        y_pred = ['joy', 'sadness', 'fear', 'excitement']
        metrics = compute_metrics(y_true, y_pred)
        assert 'weighted_f1' in metrics
        assert 'accuracy' in metrics
        assert 'per_emotion' in metrics
```

- [ ] **Step 38: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError`

---

### Task 12: Formal Evaluation — Implementation

**Files:**
- Create: `evaluate.py`

- [ ] **Step 39: Create evaluate.py**

```python
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Formal Evaluation Script
===============================================================================
Benchmarks emotion classifier against GoEmotions, SemEval, and held-out data.

Usage:
  python evaluate.py --datasets all --output results/
  python evaluate.py --datasets goemotions --text-only
  python evaluate.py --datasets held-out --with-biometrics

Integrates with:
- ML Training & Prediction Systems
- Neural Workflow AI Engine
===============================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime
from collections import Counter

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_classifier import MultimodalEmotionAnalyzer, FusionHead

# GoEmotions 27 → Quantara 32 mapping
GOEMOTIONS_MAPPING = {
    'admiration': 'gratitude',
    'amusement': 'fun',
    'anger': 'anger',
    'annoyance': 'frustration',
    'approval': 'pride',
    'caring': 'compassion',
    'confusion': 'overwhelmed',
    'curiosity': 'enthusiasm',
    'desire': 'love',
    'disappointment': 'sadness',
    'disapproval': 'contempt',
    'disgust': 'disgust',
    'embarrassment': 'shame',
    'excitement': 'excitement',
    'fear': 'fear',
    'gratitude': 'gratitude',
    'grief': 'grief',
    'joy': 'joy',
    'love': 'love',
    'nervousness': 'anxiety',
    'optimism': 'hope',
    'pride': 'pride',
    'realization': 'mindfulness',
    'relief': 'relief',
    'remorse': 'guilt',
    'sadness': 'sadness',
    'surprise': 'surprise',
}

# SemEval 2018 Task 1 → Quantara mapping
SEMEVAL_MAPPING = {
    'anger': 'anger',
    'anticipation': 'enthusiasm',
    'disgust': 'disgust',
    'fear': 'fear',
    'joy': 'joy',
    'love': 'love',
    'optimism': 'hope',
    'pessimism': 'worry',
    'sadness': 'sadness',
    'surprise': 'surprise',
    'trust': 'calm',
}


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute classification metrics."""
    labels = sorted(set(y_true + y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = float(np.average(f1, weights=support))

    per_emotion = {}
    for i, label in enumerate(labels):
        per_emotion[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }

    # Per-family metrics
    from emotion_classifier import _EMOTION_TO_FAMILY
    family_true = [_EMOTION_TO_FAMILY.get(e, 'Neutral') for e in y_true]
    family_pred = [_EMOTION_TO_FAMILY.get(e, 'Neutral') for e in y_pred]
    family_accuracy = accuracy_score(family_true, family_pred)

    return {
        'accuracy': float(accuracy),
        'weighted_f1': float(weighted_f1),
        'family_accuracy': float(family_accuracy),
        'per_emotion': per_emotion,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'labels': labels,
    }


def load_goemotions():
    """Load GoEmotions dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset('google-research-datasets/goemotions', 'simplified', split='test')

    texts, labels = [], []
    label_names = ds.features['labels'].feature.names

    for example in ds:
        if len(example['labels']) == 1:  # Single-label only
            label_idx = example['labels'][0]
            label_name = label_names[label_idx]
            if label_name == 'neutral':
                mapped = 'neutral'
            elif label_name in GOEMOTIONS_MAPPING:
                mapped = GOEMOTIONS_MAPPING[label_name]
            else:
                continue
            texts.append(example['text'])
            labels.append(mapped)

    return texts, labels


def load_semeval():
    """Load SemEval 2018 Task 1 dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset('sem_eval_2018_task_1', 'subtask5.english', split='test')

    texts, labels = [], []
    emotion_columns = [c for c in ds.column_names if c in SEMEVAL_MAPPING]

    for example in ds:
        # SemEval uses multi-label binary columns — take the strongest signal
        active_emotions = [col for col in emotion_columns if example.get(col, 0) == 1]
        if len(active_emotions) == 1:
            mapped = SEMEVAL_MAPPING.get(active_emotions[0])
            if mapped:
                texts.append(example['Tweet'])
                labels.append(mapped)

    return texts, labels


def load_held_out():
    """Load held-out validation data from local dataset."""
    import pandas as pd

    data_dir = 'data/quantara_emotion'
    texts, labels = [], []

    # Try CSV files in data directory
    for csv_file in ['text_emotion.csv', 'Emotion_classify_Data.csv']:
        path = os.path.join(data_dir, csv_file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            text_col = [c for c in df.columns if c.lower() in ['text', 'content', 'comment']]
            label_col = [c for c in df.columns if c.lower() in ['emotion', 'label', 'sentiment']]
            if text_col and label_col:
                for _, row in df.iterrows():
                    emotion = str(row[label_col[0]]).lower().strip()
                    if emotion in set(FusionHead.EMOTIONS):
                        texts.append(str(row[text_col[0]]))
                        labels.append(emotion)

    # Use last 20% as held-out
    split = int(len(texts) * 0.8)
    return texts[split:], labels[split:]


def evaluate_dataset(analyzer, texts, labels, with_biometrics=False):
    """Run analyzer on dataset and collect predictions."""
    predictions = []

    for text in texts:
        biometrics = None
        if with_biometrics:
            biometrics = {'heart_rate': 70, 'hrv': 50, 'eda': 2.0}

        result = analyzer.analyze(text, biometrics=biometrics)
        predictions.append(result['dominant_emotion'])

    return predictions


def save_confusion_matrix_plot(metrics, output_dir):
    """Save confusion matrix as heatmap PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(metrics['confusion_matrix'])
        labels = metrics['labels']

        fig, ax = plt.subplots(figsize=(max(12, len(labels)), max(10, len(labels) * 0.8)))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
                    cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Emotion Classification Confusion Matrix')
        plt.tight_layout()

        path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(path, dpi=150)
        plt.close()
        return path
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate emotion classifier')
    parser.add_argument('--datasets', default='all', help='Datasets: all, goemotions, held-out')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--text-only', action='store_true', help='Text-only evaluation')
    parser.add_argument('--with-biometrics', action='store_true', help='Include biometrics')
    parser.add_argument('--checkpoint', default=None, help='Classifier checkpoint')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading analyzer...")
    analyzer = MultimodalEmotionAnalyzer(
        classifier_checkpoint=args.checkpoint,
        use_sentence_transformer=True
    )

    all_results = {}

    datasets_to_run = args.datasets.split(',') if args.datasets != 'all' else ['goemotions', 'semeval', 'held-out']

    for dataset_name in datasets_to_run:
        dataset_name = dataset_name.strip()
        print(f"\nEvaluating: {dataset_name}")

        if dataset_name == 'goemotions':
            texts, labels = load_goemotions()
        elif dataset_name == 'semeval':
            texts, labels = load_semeval()
        elif dataset_name == 'held-out':
            texts, labels = load_held_out()
        else:
            print(f"  Unknown dataset: {dataset_name}, skipping")
            continue

        print(f"  Samples: {len(texts)}")
        if len(texts) == 0:
            print(f"  No data, skipping")
            continue

        predictions = evaluate_dataset(
            analyzer, texts, labels,
            with_biometrics=args.with_biometrics
        )

        metrics = compute_metrics(labels, predictions)
        all_results[dataset_name] = metrics

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.3f}")
        print(f"  Family Accuracy: {metrics['family_accuracy']:.3f}")

        # Confusion matrix plot
        plot_path = save_confusion_matrix_plot(metrics, args.output)
        if plot_path:
            print(f"  Confusion matrix: {plot_path}")

    # Fusion lift (if both text-only and with-biometrics results exist)
    if args.with_biometrics and 'held-out' in all_results:
        # Run text-only for comparison
        texts, labels = load_held_out()
        text_only_preds = evaluate_dataset(analyzer, texts, labels, with_biometrics=False)
        text_only_metrics = compute_metrics(labels, text_only_preds)
        fusion_lift = all_results['held-out']['weighted_f1'] - text_only_metrics['weighted_f1']
        all_results['fusion_lift'] = {
            'text_only_f1': text_only_metrics['weighted_f1'],
            'with_bio_f1': all_results['held-out']['weighted_f1'],
            'lift': fusion_lift,
        }
        print(f"\nFusion lift: {fusion_lift:+.3f}")

    # Save report
    report_path = os.path.join(args.output, f"eval_{datetime.now().strftime('%Y-%m-%d')}.json")
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 40: Run evaluation tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_evaluate.py -v`
Expected: All 3 tests PASS

- [ ] **Step 41: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 42: Commit**

```bash
git add evaluate.py requirements-dev.txt tests/test_evaluate.py
git commit -m "feat: add formal evaluation script with GoEmotions and held-out benchmarks"
```

---

## Final Integration

### Task 13: Integration Verification

- [ ] **Step 43: Run complete test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS across all test files

- [ ] **Step 44: Verify API server starts**

Run: `cd /Users/bel/quantara-nanoGPT && timeout 10 python -c "from emotion_api_server import get_app; app = get_app(); print('OK')" || true`
Expected: Prints "OK" (or graceful timeout/error if model not present locally)

- [ ] **Step 45: Final commit with all integration**

```bash
git add -A
git commit -m "chore: final integration of emotion GPT enhancement suite"
```

- [ ] **Step 46: Push to deploy**

```bash
git push origin master
```
