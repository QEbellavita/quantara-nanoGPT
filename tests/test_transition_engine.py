# tests/test_transition_engine.py
"""Tests for the Emotion Transition Engine (graph-based pathfinding + adaptive weights)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_transition_engine import (
    TransitionGraph,
    TransitionSession,
    AdaptiveWeightTracker,
    EmotionTransitionEngine,
    CURATED_EDGES,
    STEP_TYPES,
)

# The canonical 32 emotions
ALL_EMOTIONS = [
    'joy', 'excitement', 'enthusiasm', 'fun', 'gratitude', 'pride',
    'sadness', 'grief', 'boredom', 'nostalgia',
    'anger', 'frustration', 'hate', 'contempt', 'disgust', 'jealousy',
    'fear', 'anxiety', 'worry', 'overwhelmed', 'stressed',
    'love', 'compassion',
    'calm', 'relief', 'mindfulness', 'resilience', 'hope',
    'guilt', 'shame',
    'surprise',
    'neutral',
]


# ─── TransitionGraph Tests ────────────────────────────────────────────────────

class TestTransitionGraph:
    """Tests for the directed weighted graph over 32 emotions."""

    @pytest.fixture
    def graph(self):
        return TransitionGraph()

    def test_all_32_emotions_are_nodes(self, graph):
        """Every canonical emotion must be a node in the graph."""
        for em in ALL_EMOTIONS:
            assert em in graph.nodes, f"Missing node: {em}"

    def test_has_at_least_50_edges(self, graph):
        """Graph should have at least 50 directed edges."""
        assert graph.edge_count >= 50

    def test_find_path_anxiety_to_calm(self, graph):
        """Pathfinding from anxiety to calm should produce a valid path."""
        path = graph.find_path('anxiety', 'calm')
        assert len(path) >= 1
        # First step should start from anxiety
        assert path[0]['from'] == 'anxiety'
        # Last step should end at calm
        assert path[-1]['to'] == 'calm'

    def test_path_steps_have_required_fields(self, graph):
        """Each step in a path must have technique, exercise, duration, step_type."""
        path = graph.find_path('anger', 'calm')
        assert len(path) >= 1
        for step in path:
            assert 'technique' in step
            assert 'exercise' in step
            assert 'duration_min' in step
            assert 'step_type' in step
            assert 'from' in step
            assert 'to' in step

    def test_same_emotion_returns_empty(self, graph):
        """Path from an emotion to itself should be empty."""
        path = graph.find_path('joy', 'joy')
        assert path == []

    def test_all_emotions_are_reachable(self, graph):
        """Every emotion should be reachable from at least one other emotion."""
        for em in ALL_EMOTIONS:
            # Find at least one other emotion that can reach this one
            reachable = False
            for other in ALL_EMOTIONS:
                if other == em:
                    continue
                path = graph.find_path(other, em)
                if path:
                    reachable = True
                    break
            assert reachable, f"Emotion '{em}' is unreachable from any other emotion"

    def test_unknown_emotion_raises(self, graph):
        """Querying with unknown emotions should raise ValueError."""
        with pytest.raises(ValueError):
            graph.find_path('nonexistent', 'calm')
        with pytest.raises(ValueError):
            graph.find_path('calm', 'nonexistent')


# ─── TransitionSession Tests ──────────────────────────────────────────────────

class TestTransitionSession:
    """Tests for session tracking through a multi-step pathway."""

    @pytest.fixture
    def sample_path(self):
        return [
            {'from': 'anxiety', 'to': 'stressed', 'technique': 'breathing',
             'exercise': 'Box breathing 4-4-4-4', 'duration_min': 5, 'step_type': 'calming'},
            {'from': 'stressed', 'to': 'calm', 'technique': 'body scan',
             'exercise': 'Progressive muscle relaxation', 'duration_min': 10, 'step_type': 'calming'},
        ]

    @pytest.fixture
    def session(self, sample_path):
        return TransitionSession('user123', 'anxiety', 'calm', sample_path)

    def test_starts_at_step_zero(self, session):
        assert session.current_step == 0

    def test_advance_increments_step(self, session):
        session.advance()
        assert session.current_step == 1

    def test_is_complete_after_all_steps(self, session):
        session.advance()
        session.advance()
        assert session.is_complete

    def test_not_complete_initially(self, session):
        assert not session.is_complete

    def test_get_current_step_returns_info(self, session):
        step = session.get_current_step()
        assert step['from'] == 'anxiety'
        assert step['to'] == 'stressed'
        assert step['technique'] == 'breathing'
        assert step['step_type'] == 'calming'

    def test_get_current_step_none_when_complete(self, session):
        session.advance()
        session.advance()
        assert session.get_current_step() is None

    def test_check_biometric_criteria_hr_drop(self, session):
        """Calming step passes if HR drops >= 5."""
        assert session.check_biometric_criteria({'hr_delta': -6})

    def test_check_biometric_criteria_hrv_rise(self, session):
        """Calming step passes if HRV rises >= 10."""
        assert session.check_biometric_criteria({'hrv_delta': 12})

    def test_check_biometric_criteria_fails(self, session):
        """Calming step fails if neither condition met."""
        assert not session.check_biometric_criteria({'hr_delta': -2, 'hrv_delta': 3})

    def test_check_biometric_criteria_no_data(self, session):
        """No biometric data should return None (unknown)."""
        assert session.check_biometric_criteria({}) is None


# ─── AdaptiveWeightTracker Tests ──────────────────────────────────────────────

class TestAdaptiveWeightTracker:
    """Tests for SQLite-backed outcome logging and weight adjustment."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return AdaptiveWeightTracker(db_path=str(tmp_path / 'test_weights.db'))

    def test_log_outcome_writes_to_db(self, tracker):
        """Logging an outcome should not raise and should persist."""
        tracker.log_outcome('anxiety', 'calm', 'breathing', success=True)
        weights = tracker.get_adjusted_weights('anxiety', 'calm')
        assert isinstance(weights, dict)

    def test_get_adjusted_weights_after_successes(self, tracker):
        """After several successes, weights should reflect them."""
        for _ in range(5):
            tracker.log_outcome('anger', 'calm', 'perspective-taking', success=True)
        weights = tracker.get_adjusted_weights('anger', 'calm')
        assert 'perspective-taking' in weights
        assert weights['perspective-taking'] > 0

    def test_get_adjusted_weights_empty(self, tracker):
        """No outcomes logged should return empty dict."""
        weights = tracker.get_adjusted_weights('joy', 'calm')
        assert weights == {}


# ─── EmotionTransitionEngine Tests ────────────────────────────────────────────

class TestEmotionTransitionEngine:
    """Tests for the main interface combining graph + tracker + sessions."""

    @pytest.fixture
    def engine(self, tmp_path):
        return EmotionTransitionEngine(db_path=str(tmp_path / 'test_engine.db'))

    def test_start_session_returns_session(self, engine):
        session = engine.start_session('user1', 'anxiety', 'calm')
        assert isinstance(session, TransitionSession)
        assert session.user_id == 'user1'

    def test_get_session_retrieves(self, engine):
        session = engine.start_session('user2', 'anger', 'calm')
        retrieved = engine.get_session(session.session_id)
        assert retrieved is session

    def test_cleanup_session_removes(self, engine):
        session = engine.start_session('user3', 'fear', 'calm')
        sid = session.session_id
        engine.cleanup_session(sid)
        assert engine.get_session(sid) is None

    def test_get_session_nonexistent(self, engine):
        assert engine.get_session('no-such-id') is None


# ─── STEP_TYPES and CURATED_EDGES basic checks ───────────────────────────────

class TestConstants:
    def test_step_types_has_calming(self):
        assert 'calming' in STEP_TYPES
        assert STEP_TYPES['calming']['hr_delta'] == -5
        assert STEP_TYPES['calming']['hrv_delta'] == 10

    def test_step_types_has_activation(self):
        assert 'activation' in STEP_TYPES
        assert STEP_TYPES['activation']['hr_delta'] == 5

    def test_step_types_has_cognitive(self):
        assert 'cognitive' in STEP_TYPES

    def test_curated_edges_count(self):
        assert len(CURATED_EDGES) >= 45
