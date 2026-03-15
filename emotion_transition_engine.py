"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Emotion Transition Engine
===============================================================================
Graph-based pathfinding engine for multi-step emotion transitions.
Uses Dijkstra shortest path over 32 canonical emotions with adaptive weights.

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Biometric Integration Engine
- Therapist Dashboard Engine

Components:
  TransitionGraph        - Directed weighted graph, Dijkstra pathfinding
  TransitionSession      - Tracks user through multi-step pathway
  AdaptiveWeightTracker  - SQLite-backed outcome logging & weight adjustment
  EmotionTransitionEngine - Main interface combining all components
===============================================================================
"""

import heapq
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ─── Step Types ───────────────────────────────────────────────────────────────

STEP_TYPES = {
    'calming': {
        'hr_delta': -5,      # HR should drop by at least 5 bpm
        'hrv_delta': 10,     # HRV should rise by at least 10 ms
        'description': 'Calming step — biometric: HR drops or HRV rises',
    },
    'activation': {
        'hr_delta': 5,       # HR should rise by at least 5 bpm
        'eda_delta': 0.5,    # EDA should rise by at least 0.5 µS
        'description': 'Activation step — biometric: HR or EDA rises',
    },
    'cognitive': {
        'time_based': True,
        'description': 'Cognitive step — time-based completion',
    },
}


# ─── Curated Edges ───────────────────────────────────────────────────────────
# Format: (from, to, weight, technique, exercise, duration_min, step_type)

CURATED_EDGES = [
    # === Joy family transitions ===
    ('joy', 'gratitude', 1.0, 'Savoring', 'Pause and list 3 things you appreciate about this moment', 5, 'cognitive'),
    ('joy', 'enthusiasm', 1.0, 'Momentum Building', 'Channel this feeling into your next goal step', 5, 'activation'),
    ('excitement', 'joy', 1.0, 'Grounding', 'Feel feet on ground, take 3 slow breaths to settle', 5, 'calming'),
    ('excitement', 'calm', 2.0, 'Grounding Sequence', 'Box breathing 4-4-4-4 for 4 rounds', 8, 'calming'),
    ('enthusiasm', 'joy', 1.0, 'Values Alignment', 'Connect this energy to what matters most to you', 5, 'cognitive'),
    ('enthusiasm', 'hope', 1.5, 'Goal Bridging', 'Write 3 concrete next steps toward your vision', 7, 'cognitive'),
    ('fun', 'gratitude', 1.0, 'Reflection', 'What made this moment special? Who shared it?', 5, 'cognitive'),
    ('fun', 'joy', 0.5, 'Savoring', 'Notice sensory details of the fun — sounds, colors, feelings', 3, 'cognitive'),
    ('gratitude', 'compassion', 1.0, 'Loving-Kindness', 'Extend warmth to someone who helped you today', 7, 'cognitive'),
    ('gratitude', 'love', 1.0, 'Appreciation Expression', 'Send a message of thanks to someone you care about', 5, 'cognitive'),
    ('pride', 'resilience', 1.0, 'Strength Inventory', 'Name 3 strengths that led to this achievement', 5, 'cognitive'),
    ('pride', 'gratitude', 1.0, 'Humble Reflection', 'Who supported your success? Acknowledge them', 5, 'cognitive'),

    # === Sadness family transitions ===
    ('sadness', 'relief', 2.0, 'Behavioral Activation', 'Do one small meaningful activity — walk, call a friend', 15, 'activation'),
    ('sadness', 'hope', 2.5, 'Future Visioning', 'Imagine one thing you look forward to, even if small', 10, 'cognitive'),
    ('sadness', 'calm', 2.0, 'Self-Compassion', 'Place hand on heart, say: this is hard and I am here for myself', 8, 'calming'),
    ('grief', 'sadness', 1.0, 'Meaning-Making', 'What did this relationship or experience teach you?', 15, 'cognitive'),
    ('grief', 'calm', 3.0, 'Grounding', '5-4-3-2-1 sensory grounding: name things you see, hear, feel', 10, 'calming'),
    ('boredom', 'enthusiasm', 1.5, 'Micro-Goals', 'Set a 10-minute challenge — learn, create, or explore something new', 10, 'activation'),
    ('boredom', 'enthusiasm', 2.0, 'Values Reconnection', 'What genuinely interests you? Take one step toward it', 10, 'cognitive'),
    ('nostalgia', 'gratitude', 1.0, 'Bridge Past-Present', 'What from that memory do you still carry? Honor it', 7, 'cognitive'),

    # === Anger family transitions ===
    ('anger', 'frustration', 0.5, 'Labeling', 'Name the specific trigger: I feel angry because...', 3, 'cognitive'),
    ('anger', 'calm', 3.0, 'Perspective-Taking', 'Take 5 deep breaths, then ask: what need is underneath?', 10, 'calming'),
    ('frustration', 'hope', 2.0, 'Action Planning', 'List what you can control, pick one small action', 10, 'cognitive'),
    ('frustration', 'calm', 2.5, 'Autonomic Reset', '6 exhales longer than inhales, shake out tension', 8, 'calming'),
    ('hate', 'anger', 1.0, 'Intensity Reduction', 'Name the hurt underneath — what was violated?', 10, 'cognitive'),
    ('hate', 'calm', 4.0, 'Perspective-Taking', 'Write a letter (unsent) expressing what you need', 15, 'cognitive'),
    ('contempt', 'compassion', 2.5, 'Perspective Shift', 'What if they have a reason I haven\'t considered?', 10, 'cognitive'),
    ('contempt', 'calm', 3.0, 'Grounding', 'Return to breath — 4 counts in, 7 counts out', 8, 'calming'),
    ('disgust', 'calm', 2.5, 'Graduated Exposure', 'Notice the sensation without judgment, let it pass', 10, 'calming'),
    ('disgust', 'neutral', 2.0, 'Detachment', 'Step back mentally — observe without reacting', 7, 'cognitive'),
    ('jealousy', 'gratitude', 2.5, 'Self-Comparison', 'List 3 things you have that past-you would envy', 10, 'cognitive'),
    ('jealousy', 'hope', 2.0, 'Aspiration Reframe', 'What do they have that you want? Make it a goal', 10, 'cognitive'),

    # === Fear family transitions ===
    ('fear', 'calm', 2.5, 'Grounding', 'Feet on floor, name 5 things you see right now', 8, 'calming'),
    ('fear', 'hope', 3.0, 'Cognitive Reappraisal', 'What is the most likely outcome? (Not worst case)', 10, 'cognitive'),
    ('anxiety', 'calm', 2.0, 'Body-Down Regulation', 'Slow exhale breathing: 4 in, 7 out, 5 rounds', 8, 'calming'),
    ('anxiety', 'relief', 2.5, 'Containment', 'Write worries on paper, fold it away — contained for now', 10, 'cognitive'),
    ('worry', 'calm', 2.0, 'Containment', 'Schedule a 15-min worry window, then let go until then', 5, 'cognitive'),
    ('worry', 'hope', 2.5, 'Problem-Solving', 'Pick the top worry — what is one thing you can do today?', 10, 'cognitive'),
    ('overwhelmed', 'calm', 2.5, 'Smallest Next Step', 'What is the tiniest possible action? Do just that', 5, 'calming'),
    ('overwhelmed', 'relief', 3.0, 'Triage', 'List everything, circle the 1 most urgent, ignore the rest for now', 10, 'cognitive'),
    ('stressed', 'calm', 2.0, 'Autonomic Reset', 'Physiological sigh: double inhale through nose, long exhale', 5, 'calming'),
    ('stressed', 'relief', 2.5, 'Pressure Release', 'Progressive muscle relaxation — tense and release each group', 12, 'calming'),

    # === Love family transitions ===
    ('love', 'compassion', 1.0, 'Appreciation Expression', 'Express what you value about someone you love', 5, 'cognitive'),
    ('love', 'joy', 1.0, 'Savoring', 'Notice the warmth in your body, let it expand', 5, 'calming'),
    ('compassion', 'love', 1.0, 'Service Action', 'Do one kind act for someone today', 10, 'activation'),
    ('compassion', 'calm', 1.0, 'Loving-Kindness Meditation', 'May I be happy, may they be happy, may all beings be happy', 10, 'calming'),

    # === Calm family transitions ===
    ('calm', 'mindfulness', 1.0, 'Deepening Awareness', 'Sit quietly, observe breath without changing it', 10, 'calming'),
    ('calm', 'joy', 1.5, 'Gratitude Practice', 'List 3 good things from today, savor each', 5, 'cognitive'),
    ('relief', 'gratitude', 1.0, 'Acknowledging Shift', 'What changed? Name the relief and thank yourself', 5, 'cognitive'),
    ('relief', 'calm', 0.5, 'Settling', 'Let the relief settle — 3 slow breaths', 3, 'calming'),
    ('mindfulness', 'calm', 0.5, 'Non-Reactive Observation', 'Continue observing, notice thoughts like clouds passing', 10, 'calming'),
    ('mindfulness', 'joy', 1.5, 'Insight Discovery', 'What insight emerged? Let it bring a quiet smile', 5, 'cognitive'),
    ('resilience', 'hope', 1.0, 'Past-Success Recall', 'Remember 3 times you overcame difficulty', 7, 'cognitive'),
    ('resilience', 'pride', 1.0, 'Strength Recognition', 'What strength got you through? Name it proudly', 5, 'cognitive'),
    ('hope', 'joy', 1.5, 'Concrete Steps', 'Pick one hopeful thing and plan the first step', 7, 'cognitive'),
    ('hope', 'enthusiasm', 1.0, 'Vision Building', 'Describe your ideal outcome in vivid detail', 7, 'cognitive'),

    # === Self-Conscious family transitions ===
    ('guilt', 'calm', 2.5, 'Amends Planning', 'What can you do to repair? Plan one concrete step', 10, 'cognitive'),
    ('guilt', 'compassion', 2.0, 'Self-Forgiveness', 'You made a mistake — that doesn\'t define you. What did you learn?', 10, 'cognitive'),
    ('shame', 'calm', 3.0, 'Identity Separation', 'Separate what you did from who you are — you are not your mistake', 12, 'cognitive'),
    ('shame', 'compassion', 2.5, 'Self-Worth Rebuilding', 'List 5 qualities you genuinely like about yourself', 10, 'cognitive'),

    # === Surprise & Neutral transitions ===
    ('surprise', 'enthusiasm', 1.0, 'Mindful Observation', 'What just happened? Observe with openness', 5, 'cognitive'),
    ('surprise', 'joy', 1.5, 'Positive Reframe', 'Could this be a delightful surprise? Find the gift', 5, 'cognitive'),
    ('neutral', 'enthusiasm', 1.5, 'Values Clarification', 'What matters to you right now? Explore one thread', 7, 'cognitive'),
    ('neutral', 'calm', 1.0, 'Mindful Settling', 'Neutral is fine — deepen it into peaceful awareness', 5, 'calming'),
    ('neutral', 'joy', 2.0, 'Engagement Spark', 'What brought you joy recently? Revisit it briefly', 5, 'cognitive'),

    # === Cross-family bridge edges (ensure full connectivity) ===
    ('relief', 'hope', 1.5, 'Forward Looking', 'From relief, what new door opens?', 7, 'cognitive'),

    # === Intra-family & reverse edges (ensure all 32 are reachable) ===
    ('joy', 'excitement', 1.0, 'Energy Amplification', 'Let joy build into excitement — what thrills you?', 5, 'activation'),
    ('joy', 'fun', 1.0, 'Playfulness', 'Turn this joy into play — do something lighthearted', 5, 'activation'),
    ('sadness', 'grief', 1.5, 'Deepening', 'What loss underlies this sadness? Honor it', 10, 'cognitive'),
    ('sadness', 'nostalgia', 1.0, 'Memory Bridge', 'What happy memory connects to this feeling?', 7, 'cognitive'),
    ('calm', 'boredom', 2.0, 'Stillness Shift', 'Too much calm can drift to boredom — notice it', 5, 'cognitive'),
    ('neutral', 'boredom', 1.5, 'Disengagement Notice', 'Neutral fading to boredom — what spark is missing?', 5, 'cognitive'),
    ('anger', 'hate', 2.0, 'Intensity Escalation', 'When anger deepens — pause and notice the shift', 5, 'cognitive'),
    ('anger', 'contempt', 1.5, 'Moral Judgment', 'Anger can harden into contempt — catch it early', 5, 'cognitive'),
    ('anger', 'disgust', 1.5, 'Aversion', 'Anger mixing with rejection — name the boundary', 5, 'cognitive'),
    ('frustration', 'jealousy', 2.0, 'Comparison Trap', 'Frustration can trigger comparison — notice it', 7, 'cognitive'),
    ('anxiety', 'fear', 1.0, 'Root Identification', 'What specific fear underlies this anxiety?', 7, 'cognitive'),
    ('anxiety', 'worry', 0.5, 'Cognitive Narrowing', 'Anxiety becoming worry — name the specific concern', 5, 'cognitive'),
    ('stressed', 'overwhelmed', 1.0, 'Load Assessment', 'Stress building up — how much is on your plate?', 5, 'cognitive'),
    ('worry', 'anxiety', 1.0, 'Escalation Notice', 'Worry spreading — ground yourself before it grows', 5, 'calming'),
    ('calm', 'love', 1.5, 'Heart Opening', 'From calm, let warmth flow toward someone you care about', 7, 'cognitive'),
    ('compassion', 'guilt', 2.0, 'Responsibility Check', 'Compassion revealing where you fell short — gently', 10, 'cognitive'),
    ('shame', 'guilt', 1.0, 'Specificity', 'Move from global shame to specific guilt — what action?', 7, 'cognitive'),
    ('guilt', 'shame', 1.5, 'Internalization', 'When guilt about actions becomes shame about self — pause', 7, 'cognitive'),
    ('calm', 'surprise', 2.0, 'Openness', 'From calm, notice what is unexpected around you', 5, 'cognitive'),
    ('joy', 'surprise', 1.5, 'Delight', 'Let joy open you to delightful surprises', 5, 'cognitive'),
    ('fear', 'anxiety', 0.5, 'Generalization', 'Fear spreading into general anxiety — name the source', 5, 'cognitive'),
    ('fear', 'stressed', 1.5, 'Activation', 'Fear triggering stress response — notice body signals', 5, 'activation'),
    ('nostalgia', 'sadness', 1.0, 'Bittersweet', 'The sweet ache of nostalgia deepening — sit with it', 7, 'cognitive'),
]

# Emotion family roots — used to build bridge edges through neutral
_FAMILY_ROOTS = [
    'joy', 'sadness', 'anger', 'fear', 'love', 'calm',
    'guilt', 'surprise', 'boredom', 'hope', 'relief',
    'resilience', 'mindfulness', 'pride', 'compassion',
]

# The canonical 32 emotions
CANONICAL_EMOTIONS = [
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


# ─── TransitionGraph ─────────────────────────────────────────────────────────

class TransitionGraph:
    """Directed weighted graph over emotions with Dijkstra pathfinding."""

    def __init__(self, edges=None, adaptive_weights: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None):
        self.nodes = set(CANONICAL_EMOTIONS)
        # adjacency: node -> [(neighbor, weight, metadata)]
        self._adj: Dict[str, List[Tuple[str, float, dict]]] = {e: [] for e in self.nodes}

        self._edge_count = 0
        for edge in (edges or CURATED_EDGES):
            src, dst, weight, technique, exercise, duration, step_type = edge
            # Skip edges involving non-canonical emotions
            if src not in self.nodes or dst not in self.nodes:
                continue
            # Apply adaptive weight adjustment if available
            if adaptive_weights:
                adj = adaptive_weights.get((src, dst), {})
                success_rate = adj.get(technique)
                if success_rate is not None:
                    # Lower weight for higher success rate (weight is cost)
                    weight = weight * (1.5 - success_rate)
            self._adj[src].append((dst, weight, {
                'technique': technique,
                'exercise': exercise,
                'duration_min': duration,
                'step_type': step_type,
            }))
            self._edge_count += 1

        # Ensure strong connectivity via neutral as universal hub
        self._add_neutral_bridges()

    def _add_neutral_bridges(self):
        """Add bridge edges through neutral to ensure strong connectivity.

        For every canonical emotion, ensure there is an edge to neutral and
        from neutral to that emotion (using high weights so Dijkstra prefers
        natural paths when available).
        """
        # Collect existing edges from/to neutral to avoid duplicates
        existing_to_neutral = set()
        existing_from_neutral = set()
        for neighbor, _, _ in self._adj.get('neutral', []):
            existing_from_neutral.add(neighbor)
        for emotion in self.nodes:
            for neighbor, _, _ in self._adj.get(emotion, []):
                if neighbor == 'neutral':
                    existing_to_neutral.add(emotion)

        bridge_meta = {
            'technique': 'Neutral Bridge',
            'exercise': 'Pause, breathe, and return to a neutral baseline',
            'duration_min': 5,
            'step_type': 'calming',
        }

        for emotion in CANONICAL_EMOTIONS:
            if emotion == 'neutral':
                continue
            # emotion -> neutral
            if emotion not in existing_to_neutral:
                self._adj[emotion].append(('neutral', 3.5, bridge_meta.copy()))
                self._edge_count += 1
            # neutral -> emotion
            if emotion not in existing_from_neutral:
                self._adj['neutral'].append((emotion, 3.5, bridge_meta.copy()))
                self._edge_count += 1

    @property
    def edge_count(self) -> int:
        return self._edge_count

    def find_path(self, from_emotion: str, to_emotion: str) -> Optional[List[dict]]:
        """Find shortest path between two emotions using Dijkstra.
        Returns list of step dicts, empty list if same emotion, or None if unreachable.
        Raises ValueError for unknown emotions.
        """
        if from_emotion not in self.nodes:
            raise ValueError(f"Unknown emotion: {from_emotion}")
        if to_emotion not in self.nodes:
            raise ValueError(f"Unknown emotion: {to_emotion}")
        if from_emotion == to_emotion:
            return []

        # Dijkstra
        dist = {n: float('inf') for n in self.nodes}
        prev = {n: None for n in self.nodes}
        prev_edge = {n: None for n in self.nodes}
        dist[from_emotion] = 0
        # (distance, counter, node) — counter breaks ties
        pq = [(0, 0, from_emotion)]
        counter = 1

        while pq:
            d, _, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == to_emotion:
                break
            for neighbor, weight, meta in self._adj.get(u, []):
                nd = d + weight
                if nd < dist[neighbor]:
                    dist[neighbor] = nd
                    prev[neighbor] = u
                    prev_edge[neighbor] = meta
                    heapq.heappush(pq, (nd, counter, neighbor))
                    counter += 1

        if dist[to_emotion] == float('inf'):
            return None  # No path exists

        # Reconstruct path
        path = []
        node = to_emotion
        while prev[node] is not None:
            meta = prev_edge[node]
            path.append({
                'from': prev[node],
                'to': node,
                'technique': meta['technique'],
                'exercise': meta['exercise'],
                'duration_min': meta['duration_min'],
                'step_type': meta['step_type'],
            })
            node = prev[node]

        path.reverse()
        return path


# ─── TransitionSession ───────────────────────────────────────────────────────

class TransitionSession:
    """Tracks a user through a multi-step emotion transition pathway."""

    def __init__(self, user_id: str, from_emotion: str, to_emotion: str, path: List[dict]):
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id
        self.from_emotion = from_emotion
        self.to_emotion = to_emotion
        self.path = path
        self.current_step = 0
        self.created_at = datetime.now().isoformat()

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.path)

    def advance(self):
        """Move to the next step."""
        if not self.is_complete:
            self.current_step += 1

    def get_current_step(self) -> Optional[dict]:
        """Return current step info, or None if complete."""
        if self.is_complete:
            return None
        step = self.path[self.current_step].copy()
        step['step_number'] = self.current_step
        step['total_steps'] = len(self.path)
        return step

    def check_biometric_criteria(self, biometrics: dict) -> Optional[bool]:
        """Check if current step's biometric criteria are met.
        Returns True if met, False if not met, None if no relevant data.
        """
        if self.is_complete:
            return None
        step = self.path[self.current_step]
        step_type = step.get('step_type', 'cognitive')
        criteria = STEP_TYPES.get(step_type, {})

        if step_type == 'calming':
            hr_delta = biometrics.get('hr_delta')
            hrv_delta = biometrics.get('hrv_delta')
            if hr_delta is None and hrv_delta is None:
                return None
            # HR should drop by threshold or HRV should rise by threshold
            hr_ok = hr_delta is not None and hr_delta <= criteria['hr_delta']
            hrv_ok = hrv_delta is not None and hrv_delta >= criteria['hrv_delta']
            return hr_ok or hrv_ok

        elif step_type == 'activation':
            hr_delta = biometrics.get('hr_delta')
            eda_delta = biometrics.get('eda_delta')
            if hr_delta is None and eda_delta is None:
                return None
            hr_ok = hr_delta is not None and hr_delta >= criteria['hr_delta']
            eda_ok = eda_delta is not None and eda_delta >= criteria['eda_delta']
            return hr_ok or eda_ok

        else:  # cognitive — time-based
            return None

    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'from_emotion': self.from_emotion,
            'to_emotion': self.to_emotion,
            'current_step': self.current_step,
            'total_steps': len(self.path),
            'is_complete': self.is_complete,
            'path': self.path,
            'created_at': self.created_at,
        }


# ─── AdaptiveWeightTracker ───────────────────────────────────────────────────

class AdaptiveWeightTracker:
    """SQLite-backed tracker that logs transition outcomes and adjusts weights."""

    def __init__(self, db_path: str = 'transition_outcomes.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_emotion TEXT NOT NULL,
                to_emotion TEXT NOT NULL,
                technique TEXT NOT NULL,
                success INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def log_outcome(self, from_emotion: str, to_emotion: str, technique: str, success: bool):
        """Log a transition outcome."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute(
            'INSERT INTO outcomes (from_emotion, to_emotion, technique, success, timestamp) VALUES (?, ?, ?, ?, ?)',
            (from_emotion, to_emotion, technique, int(success), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    def get_adjusted_weights(self, from_emotion: str, to_emotion: str) -> Dict[str, float]:
        """Get adjusted weights for techniques between two emotions.
        Returns dict of technique -> adjusted_weight.
        Higher weight = more successful historically.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA busy_timeout=5000')
        rows = conn.execute(
            '''SELECT technique, SUM(success) as successes, COUNT(*) as total
               FROM outcomes
               WHERE from_emotion = ? AND to_emotion = ?
               GROUP BY technique''',
            (from_emotion, to_emotion)
        ).fetchall()
        conn.close()

        result = {}
        for technique, successes, total in rows:
            result[technique] = successes / total if total > 0 else 0.5
        return result


# ─── EmotionTransitionEngine ─────────────────────────────────────────────────

class EmotionTransitionEngine:
    """Main interface combining graph pathfinding, adaptive weights, and sessions.

    Integrates with Neural Workflow AI Engine for emotion-aware transitions.
    """

    def __init__(self, db_path: str = 'transition_outcomes.db'):
        self.tracker = AdaptiveWeightTracker(db_path=db_path)
        self.graph = self._build_graph()
        self._sessions: Dict[str, TransitionSession] = {}

    def _build_graph(self) -> TransitionGraph:
        """Build the transition graph with adaptive weights applied."""
        # Collect adaptive weights for all edges
        adaptive_weights: Dict[Tuple[str, str], Dict[str, float]] = {}
        for edge in CURATED_EDGES:
            src, dst = edge[0], edge[1]
            key = (src, dst)
            if key not in adaptive_weights:
                weights = self.tracker.get_adjusted_weights(src, dst)
                if weights:
                    adaptive_weights[key] = weights
        return TransitionGraph(adaptive_weights=adaptive_weights)

    def start_session(self, user_id: str, from_emotion: str, to_emotion: str) -> TransitionSession:
        """Start a new transition session with graph-based pathfinding."""
        path = self.graph.find_path(from_emotion, to_emotion)
        session = TransitionSession(user_id, from_emotion, to_emotion, path)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[TransitionSession]:
        """Retrieve an active session by ID."""
        return self._sessions.get(session_id)

    def cleanup_session(self, session_id: str):
        """Remove a session."""
        self._sessions.pop(session_id, None)

    def find_path(self, from_emotion: str, to_emotion: str) -> Optional[List[dict]]:
        """Find optimal path without creating a session."""
        return self.graph.find_path(from_emotion, to_emotion)

    def log_feedback(self, from_emotion: str, to_emotion: str, technique: str, success: bool):
        """Log transition outcome feedback and rebuild graph with updated weights."""
        self.tracker.log_outcome(from_emotion, to_emotion, technique, success=success)
        self.graph = self._build_graph()
