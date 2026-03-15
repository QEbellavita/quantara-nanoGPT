"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Emotion Transition Tracker (Therapist Dashboard)
===============================================================================
Tracks per-user emotion shifts over time using Markov chain transition
probabilities. Detects concerning patterns (rapid cycling, negative spirals,
emotional flatline) and generates actionable insights for the therapist
dashboard.

Integrates with:
- Neural Workflow AI Engine
- Therapist Dashboard Engine
- Real-time Dashboard Data
- ML Training & Prediction Systems
- AI Conversational Coach
- Biometric Integration Engine

Pattern Detection Rules:
  rapid_cycling       — 3+ family changes within 30 minutes
  negative_spiral     — 3+ consecutive negative emotions
  emotional_flatline  — same emotion sustained for 2+ hours
  high_arousal_sustained — Fear/Anger family for 1+ hour
  positive_recovery   — transition from negative to positive (good sign)

Storage: in-memory with optional JSON persistence to data/emotion_transitions/
Thread-safe via threading.Lock per user.
===============================================================================
"""

import json
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── Emotion Taxonomy (mirrored from emotion_classifier.py) ─────────────────

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

_EMOTION_TO_FAMILY: Dict[str, str] = {}
for _fam, _ems in EMOTION_FAMILIES.items():
    for _e in _ems:
        _EMOTION_TO_FAMILY[_e] = _fam

NEGATIVE_FAMILIES = {'Anger', 'Fear', 'Sadness', 'Self-Conscious'}
POSITIVE_FAMILIES = {'Joy', 'Love', 'Calm'}
HIGH_AROUSAL_FAMILIES = {'Fear', 'Anger'}


# ─── Data Structures ────────────────────────────────────────────────────────

class EmotionRecord:
    """Single timestamped emotion reading for a user."""

    __slots__ = ('emotion', 'family', 'confidence', 'timestamp')

    def __init__(self, emotion: str, family: str, confidence: float,
                 timestamp: Optional[str] = None):
        self.emotion = emotion.lower()
        self.family = family
        self.confidence = confidence
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'emotion': self.emotion,
            'family': self.family,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'EmotionRecord':
        return EmotionRecord(
            emotion=d['emotion'],
            family=d['family'],
            confidence=d['confidence'],
            timestamp=d.get('timestamp'),
        )

    @property
    def dt(self) -> datetime:
        """Parse timestamp to datetime (handles both Z-suffix and plain ISO)."""
        ts = self.timestamp.replace('Z', '+00:00')
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return datetime.utcnow()


# ─── Transition Tracker ─────────────────────────────────────────────────────

class EmotionTransitionTracker:
    """
    Tracks per-user emotion history, computes transition probabilities,
    detects concerning patterns, and generates therapist dashboard summaries.

    Thread-safe: uses a per-user lock to avoid race conditions when multiple
    API calls write to the same user history concurrently.
    """

    # Pattern detection thresholds
    RAPID_CYCLING_WINDOW_MIN = 30       # minutes
    RAPID_CYCLING_MIN_CHANGES = 3
    NEGATIVE_SPIRAL_MIN_COUNT = 3
    FLATLINE_HOURS = 2
    HIGH_AROUSAL_HOURS = 1

    # Persistence directory
    DEFAULT_PERSIST_DIR = 'data/emotion_transitions'

    def __init__(self, persist_dir: Optional[str] = None,
                 auto_persist: bool = True,
                 max_history_per_user: int = 5000):
        """
        Args:
            persist_dir: Directory for JSON persistence. None = in-memory only.
            auto_persist: If True, write to disk after every record().
            max_history_per_user: Cap per-user history to bound memory.
        """
        self._persist_dir = persist_dir or self.DEFAULT_PERSIST_DIR
        self._auto_persist = auto_persist
        self._max_history = max_history_per_user

        # user_id -> list[EmotionRecord]
        self._history: Dict[str, List[EmotionRecord]] = {}
        # user_id -> threading.Lock
        self._locks: Dict[str, threading.Lock] = {}
        # Global lock for creating per-user locks
        self._meta_lock = threading.Lock()

        # Load any persisted data
        self._load_all()

    # ── Lock helpers ─────────────────────────────────────────────────────

    def _user_lock(self, user_id: str) -> threading.Lock:
        """Get or create a per-user lock."""
        if user_id not in self._locks:
            with self._meta_lock:
                if user_id not in self._locks:
                    self._locks[user_id] = threading.Lock()
        return self._locks[user_id]

    # ── Core API ─────────────────────────────────────────────────────────

    def record(self, user_id: str, emotion: str, family: Optional[str] = None,
               confidence: float = 1.0,
               timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Log an emotion reading for a user.

        Args:
            user_id: Unique user/session identifier.
            emotion: One of the 32 taxonomy emotions.
            family: Emotion family (auto-resolved if omitted).
            confidence: Model confidence [0..1].
            timestamp: ISO-8601 timestamp (defaults to now UTC).

        Returns:
            Dict with recorded entry and any immediate alerts.
        """
        emotion = emotion.lower()
        if family is None:
            family = _EMOTION_TO_FAMILY.get(emotion, 'Neutral')

        rec = EmotionRecord(emotion, family, confidence, timestamp)

        with self._user_lock(user_id):
            if user_id not in self._history:
                self._history[user_id] = []
            self._history[user_id].append(rec)

            # Trim to cap
            if len(self._history[user_id]) > self._max_history:
                self._history[user_id] = self._history[user_id][-self._max_history:]

        if self._auto_persist:
            self._persist_user(user_id)

        # Quick pattern check for real-time alerts
        alerts = self.detect_patterns(user_id)

        return {
            'recorded': rec.to_dict(),
            'user_id': user_id,
            'total_records': len(self._history.get(user_id, [])),
            'alerts': alerts,
        }

    def get_transitions(self, user_id: str,
                        window_hours: float = 24) -> Dict[str, Any]:
        """
        Compute Markov-style transition probabilities between emotion families
        within a recent time window.

        Returns:
            Dict with transition_matrix (family->family->probability),
            transition_counts, and total_transitions.
        """
        records = self._windowed_records(user_id, window_hours)
        if len(records) < 2:
            return {
                'user_id': user_id,
                'window_hours': window_hours,
                'transition_matrix': {},
                'transition_counts': {},
                'total_transitions': 0,
            }

        # Count family-to-family transitions
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(len(records) - 1):
            src = records[i].family
            dst = records[i + 1].family
            counts[src][dst] += 1

        # Convert to probabilities
        matrix: Dict[str, Dict[str, float]] = {}
        total_transitions = 0
        for src, destinations in counts.items():
            total = sum(destinations.values())
            total_transitions += total
            matrix[src] = {dst: round(c / total, 4) for dst, c in destinations.items()}

        # Serialize counts (defaultdict -> dict)
        count_dict = {src: dict(dst) for src, dst in counts.items()}

        return {
            'user_id': user_id,
            'window_hours': window_hours,
            'transition_matrix': matrix,
            'transition_counts': count_dict,
            'total_transitions': total_transitions,
        }

    def get_trajectory(self, user_id: str,
                       window_hours: float = 24) -> Dict[str, Any]:
        """
        Get the emotion timeline for a user within a window.

        Returns:
            Dict with timeline list, summary stats, and family distribution.
        """
        records = self._windowed_records(user_id, window_hours)

        timeline = [r.to_dict() for r in records]

        # Family distribution
        family_counts = Counter(r.family for r in records)
        total = len(records) or 1
        family_distribution = {
            fam: round(cnt / total, 4)
            for fam, cnt in family_counts.most_common()
        }

        # Emotion distribution
        emotion_counts = Counter(r.emotion for r in records)
        emotion_distribution = {
            em: round(cnt / total, 4)
            for em, cnt in emotion_counts.most_common(10)
        }

        # Avg confidence
        avg_confidence = (
            round(sum(r.confidence for r in records) / total, 4)
            if records else 0.0
        )

        # Valence trend: fraction of time in positive vs negative families
        positive_frac = sum(
            1 for r in records if r.family in POSITIVE_FAMILIES
        ) / total
        negative_frac = sum(
            1 for r in records if r.family in NEGATIVE_FAMILIES
        ) / total

        return {
            'user_id': user_id,
            'window_hours': window_hours,
            'timeline': timeline,
            'record_count': len(records),
            'family_distribution': family_distribution,
            'emotion_distribution': emotion_distribution,
            'avg_confidence': avg_confidence,
            'valence': {
                'positive_fraction': round(positive_frac, 4),
                'negative_fraction': round(negative_frac, 4),
                'neutral_fraction': round(1.0 - positive_frac - negative_frac, 4),
            },
        }

    def detect_patterns(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Detect concerning (and positive) patterns in the user's emotion history.

        Returns:
            List of alert dicts with keys: pattern, severity, message, details.
        """
        alerts: List[Dict[str, Any]] = []
        history = self._history.get(user_id, [])
        if not history:
            return alerts

        now = datetime.utcnow()

        # --- rapid_cycling: 3+ family changes within 30 minutes ---
        window_start = now - timedelta(minutes=self.RAPID_CYCLING_WINDOW_MIN)
        recent = [r for r in history if r.dt >= window_start]
        if len(recent) >= 2:
            family_changes = sum(
                1 for i in range(1, len(recent))
                if recent[i].family != recent[i - 1].family
            )
            if family_changes >= self.RAPID_CYCLING_MIN_CHANGES:
                alerts.append({
                    'pattern': 'rapid_cycling',
                    'severity': 'warning',
                    'message': (
                        f'{family_changes} emotion family changes in the last '
                        f'{self.RAPID_CYCLING_WINDOW_MIN} minutes. '
                        'May indicate emotional instability or dysregulation.'
                    ),
                    'details': {
                        'changes': family_changes,
                        'window_minutes': self.RAPID_CYCLING_WINDOW_MIN,
                        'families': [r.family for r in recent],
                    },
                    'timestamp': now.isoformat(),
                })

        # --- negative_spiral: 3+ consecutive negative emotions ---
        if len(history) >= self.NEGATIVE_SPIRAL_MIN_COUNT:
            tail = history[-self.NEGATIVE_SPIRAL_MIN_COUNT:]
            if all(r.family in NEGATIVE_FAMILIES for r in tail):
                run_length = 0
                for r in reversed(history):
                    if r.family in NEGATIVE_FAMILIES:
                        run_length += 1
                    else:
                        break
                alerts.append({
                    'pattern': 'negative_spiral',
                    'severity': 'critical',
                    'message': (
                        f'{run_length} consecutive readings in negative emotion '
                        f'families ({", ".join(set(r.family for r in tail))}). '
                        'Consider therapeutic intervention.'
                    ),
                    'details': {
                        'consecutive_negative': run_length,
                        'families': [r.family for r in history[-run_length:]],
                        'emotions': [r.emotion for r in history[-run_length:]],
                    },
                    'timestamp': now.isoformat(),
                })

        # --- emotional_flatline: same emotion for 2+ hours ---
        if len(history) >= 2:
            latest = history[-1]
            flatline_start = latest.dt
            for r in reversed(history[:-1]):
                if r.emotion == latest.emotion:
                    flatline_start = r.dt
                else:
                    break
            duration_hours = (latest.dt - flatline_start).total_seconds() / 3600
            if duration_hours >= self.FLATLINE_HOURS:
                alerts.append({
                    'pattern': 'emotional_flatline',
                    'severity': 'warning',
                    'message': (
                        f'Same emotion ({latest.emotion}) sustained for '
                        f'{duration_hours:.1f} hours. May indicate dissociation '
                        'or emotional suppression.'
                    ),
                    'details': {
                        'emotion': latest.emotion,
                        'family': latest.family,
                        'duration_hours': round(duration_hours, 2),
                    },
                    'timestamp': now.isoformat(),
                })

        # --- high_arousal_sustained: Fear/Anger for 1+ hour ---
        if len(history) >= 2:
            latest = history[-1]
            if latest.family in HIGH_AROUSAL_FAMILIES:
                arousal_start = latest.dt
                for r in reversed(history[:-1]):
                    if r.family in HIGH_AROUSAL_FAMILIES:
                        arousal_start = r.dt
                    else:
                        break
                duration_hours = (latest.dt - arousal_start).total_seconds() / 3600
                if duration_hours >= self.HIGH_AROUSAL_HOURS:
                    alerts.append({
                        'pattern': 'high_arousal_sustained',
                        'severity': 'critical',
                        'message': (
                            f'High-arousal state ({latest.family}) sustained for '
                            f'{duration_hours:.1f} hours. Risk of burnout or '
                            'panic escalation.'
                        ),
                        'details': {
                            'family': latest.family,
                            'duration_hours': round(duration_hours, 2),
                            'latest_emotion': latest.emotion,
                        },
                        'timestamp': now.isoformat(),
                    })

        # --- positive_recovery: transition from negative to positive ---
        if len(history) >= 2:
            prev = history[-2]
            curr = history[-1]
            if (prev.family in NEGATIVE_FAMILIES and
                    curr.family in POSITIVE_FAMILIES):
                alerts.append({
                    'pattern': 'positive_recovery',
                    'severity': 'info',
                    'message': (
                        f'Positive transition detected: {prev.family} '
                        f'({prev.emotion}) -> {curr.family} ({curr.emotion}). '
                        'Encouraging sign of emotional recovery.'
                    ),
                    'details': {
                        'from_family': prev.family,
                        'from_emotion': prev.emotion,
                        'to_family': curr.family,
                        'to_emotion': curr.emotion,
                    },
                    'timestamp': now.isoformat(),
                })

        return alerts

    def get_dominant_state(self, user_id: str,
                          window_hours: float = 1.0) -> Dict[str, Any]:
        """
        Get the most frequent recent emotion within a time window.

        Returns:
            Dict with dominant emotion, family, frequency, and runner-up.
        """
        records = self._windowed_records(user_id, window_hours)
        if not records:
            return {
                'user_id': user_id,
                'window_hours': window_hours,
                'dominant_emotion': None,
                'dominant_family': None,
                'frequency': 0.0,
                'record_count': 0,
            }

        emotion_counts = Counter(r.emotion for r in records)
        top_emotion, top_count = emotion_counts.most_common(1)[0]
        total = len(records)

        runner_up = None
        if len(emotion_counts) >= 2:
            ru_emotion, ru_count = emotion_counts.most_common(2)[1]
            runner_up = {
                'emotion': ru_emotion,
                'family': _EMOTION_TO_FAMILY.get(ru_emotion, 'Neutral'),
                'frequency': round(ru_count / total, 4),
            }

        return {
            'user_id': user_id,
            'window_hours': window_hours,
            'dominant_emotion': top_emotion,
            'dominant_family': _EMOTION_TO_FAMILY.get(top_emotion, 'Neutral'),
            'frequency': round(top_count / total, 4),
            'record_count': total,
            'runner_up': runner_up,
        }

    def get_dashboard_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary dict for the therapist dashboard.
        Combines trajectory, transitions, patterns, and dominant state.
        """
        trajectory = self.get_trajectory(user_id, window_hours=24)
        transitions = self.get_transitions(user_id, window_hours=24)
        patterns = self.detect_patterns(user_id)
        dominant = self.get_dominant_state(user_id, window_hours=1)

        # Severity summary
        severity_counts = Counter(a['severity'] for a in patterns)

        # Session stats
        history = self._history.get(user_id, [])
        session_start = history[0].timestamp if history else None
        session_end = history[-1].timestamp if history else None

        return {
            'user_id': user_id,
            'generated_at': datetime.utcnow().isoformat(),
            'session': {
                'first_record': session_start,
                'last_record': session_end,
                'total_records': len(history),
            },
            'current_state': dominant,
            'trajectory_24h': {
                'family_distribution': trajectory['family_distribution'],
                'emotion_distribution': trajectory['emotion_distribution'],
                'valence': trajectory['valence'],
                'avg_confidence': trajectory['avg_confidence'],
                'record_count': trajectory['record_count'],
            },
            'transition_matrix_24h': transitions['transition_matrix'],
            'total_transitions_24h': transitions['total_transitions'],
            'alerts': patterns,
            'alert_summary': {
                'total': len(patterns),
                'critical': severity_counts.get('critical', 0),
                'warning': severity_counts.get('warning', 0),
                'info': severity_counts.get('info', 0),
            },
            'recommendations': self._generate_recommendations(patterns, dominant),
        }

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _windowed_records(self, user_id: str,
                          window_hours: float) -> List[EmotionRecord]:
        """Get records within a time window, sorted by timestamp."""
        history = self._history.get(user_id, [])
        if not history:
            return []
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        return [r for r in history if r.dt >= cutoff]

    def _generate_recommendations(self, alerts: List[Dict[str, Any]],
                                  dominant: Dict[str, Any]) -> List[str]:
        """Generate therapist-facing recommendations based on patterns."""
        recs: List[str] = []
        pattern_types = {a['pattern'] for a in alerts}

        if 'rapid_cycling' in pattern_types:
            recs.append(
                'Consider grounding exercises — patient shows rapid emotional '
                'cycling which may indicate dysregulation or external stressors.'
            )
        if 'negative_spiral' in pattern_types:
            recs.append(
                'Prioritize de-escalation techniques. Patient has been in '
                'sustained negative states — CBT cognitive restructuring or '
                'behavioral activation may help.'
            )
        if 'emotional_flatline' in pattern_types:
            recs.append(
                'Assess for dissociation or alexithymia. Emotional flatline '
                'detected — gentle affect labeling exercises recommended.'
            )
        if 'high_arousal_sustained' in pattern_types:
            recs.append(
                'High arousal state sustained — consider relaxation response '
                'training (progressive muscle relaxation, box breathing). '
                'Assess for acute stressor.'
            )
        if 'positive_recovery' in pattern_types:
            recs.append(
                'Positive recovery detected — reinforce coping strategies '
                'that contributed to the shift. Document what worked.'
            )

        # Dominant state recommendation
        dom_family = dominant.get('dominant_family')
        if dom_family in NEGATIVE_FAMILIES and not pattern_types:
            recs.append(
                f'Dominant state is in {dom_family} family — monitor for '
                'pattern development. May benefit from mood tracking homework.'
            )

        if not recs:
            recs.append(
                'No concerning patterns detected. Emotional state appears '
                'stable. Continue regular monitoring.'
            )

        return recs

    # ── Persistence ──────────────────────────────────────────────────────

    def _persist_path(self, user_id: str) -> Path:
        """Get JSON file path for a user's emotion history."""
        safe_id = user_id.replace('/', '_').replace('\\', '_')
        return Path(self._persist_dir) / f'{safe_id}.json'

    def _persist_user(self, user_id: str) -> None:
        """Write a single user's history to disk."""
        try:
            path = self._persist_path(user_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'user_id': user_id,
                'updated_at': datetime.utcnow().isoformat(),
                'records': [r.to_dict() for r in self._history.get(user_id, [])],
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Non-fatal — in-memory data is still valid
            print(f"[EmotionTransitionTracker] Persist failed for {user_id}: {e}")

    def _load_all(self) -> None:
        """Load all persisted user histories from disk."""
        persist_dir = Path(self._persist_dir)
        if not persist_dir.exists():
            return
        for path in persist_dir.glob('*.json'):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                user_id = data.get('user_id', path.stem)
                records = [
                    EmotionRecord.from_dict(r)
                    for r in data.get('records', [])
                ]
                if records:
                    self._history[user_id] = records[-self._max_history:]
            except Exception as e:
                print(f"[EmotionTransitionTracker] Load failed for {path}: {e}")

    def persist_all(self) -> None:
        """Persist all users to disk (call on shutdown)."""
        for user_id in list(self._history.keys()):
            self._persist_user(user_id)

    def get_all_users(self) -> List[str]:
        """Return list of tracked user IDs."""
        return list(self._history.keys())

    def clear_user(self, user_id: str) -> None:
        """Clear all history for a user."""
        with self._user_lock(user_id):
            self._history.pop(user_id, None)
        # Remove persisted file
        path = self._persist_path(user_id)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
