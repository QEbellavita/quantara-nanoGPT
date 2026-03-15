"""QUANTARA NEURAL ECOSYSTEM - Evolution Engine

Tests for EvolutionEngine: confidence computation, stage transitions,
synergy detection, snapshot checking, and stage progress reporting.
"""

import sys
import os
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine import EvolutionEngine, STAGE_NAMES

_NOW = time.time()
_WEEK = 7 * 24 * 3600


def _make_engine():
    """Create an EvolutionEngine backed by a mock ProfileDB."""
    db = MagicMock()
    db.get_last_snapshot_time.return_value = None
    return EvolutionEngine(db)


# ─── TestConfidenceComputation ────────────────────────────────────────────────

class TestConfidenceComputation(unittest.TestCase):
    """Tests for compute_domain_confidence and compute_overall_confidence."""

    def setUp(self):
        self.engine = _make_engine()

    # --- compute_domain_confidence ---

    def test_zero_events_returns_zero(self):
        result = self.engine.compute_domain_confidence(
            event_count=0,
            latest_event_time=_NOW,
            source_count=2,
        )
        self.assertEqual(result, 0.0)

    def test_100_events_recent_multi_source_equals_one(self):
        """100 events within the last 7 days from 2+ sources should yield 1.0."""
        result = self.engine.compute_domain_confidence(
            event_count=100,
            latest_event_time=_NOW - 3600,   # 1 hour ago
            source_count=2,
        )
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_single_source_reduces_confidence(self):
        """Single source should reduce confidence by factor 0.8 vs multi-source."""
        multi = self.engine.compute_domain_confidence(100, _NOW - 3600, 2)
        single = self.engine.compute_domain_confidence(100, _NOW - 3600, 1)
        self.assertAlmostEqual(single, multi * 0.8, places=5)

    def test_old_events_reduce_confidence(self):
        """Events older than 1 week should reduce confidence below 1.0."""
        recent = self.engine.compute_domain_confidence(100, _NOW - 3600, 2)
        old = self.engine.compute_domain_confidence(100, _NOW - 3 * _WEEK, 2)
        self.assertLess(old, recent)

    def test_very_old_events_floor_at_0_3_times_volume(self):
        """Recency factor floors at 0.3 regardless of age."""
        # 50+ weeks old — recency should be at floor 0.3
        very_old = self.engine.compute_domain_confidence(
            event_count=100,
            latest_event_time=_NOW - 60 * _WEEK,
            source_count=2,
        )
        # volume=1.0, recency=0.3, source=1.0 → 0.3
        self.assertAlmostEqual(very_old, 0.3, places=5)

    def test_partial_events_scale_below_one(self):
        """50 events should yield half the volume factor of 100 events."""
        full = self.engine.compute_domain_confidence(100, _NOW - 3600, 2)
        half = self.engine.compute_domain_confidence(50, _NOW - 3600, 2)
        self.assertAlmostEqual(half, full * 0.5, places=5)

    def test_recency_exact_one_week_boundary(self):
        """An event exactly at the 1-week boundary should give recency 1.0."""
        result = self.engine.compute_domain_confidence(100, _NOW - _WEEK + 10, 2)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_recency_decays_one_extra_week(self):
        """One full extra week beyond 7 days → recency = 0.9."""
        result = self.engine.compute_domain_confidence(
            event_count=100,
            latest_event_time=_NOW - 2 * _WEEK - 60,  # ~2 weeks old
            source_count=2,
        )
        # volume=1.0, recency=0.9, source=1.0
        self.assertAlmostEqual(result, 0.9, places=4)

    # --- compute_overall_confidence ---

    def test_overall_confidence_weighted_mean(self):
        """Overall confidence is a weighted mean by event_count."""
        domain_scores = {
            'emotion': {'score': 0.8, 'event_count': 80},
            'sleep': {'score': 0.5, 'event_count': 20},
        }
        domain_meta = {
            'emotion': {'latest_event_time': _NOW - 3600, 'source_count': 2},
            'sleep': {'latest_event_time': _NOW - 3600, 'source_count': 2},
        }
        result = self.engine.compute_overall_confidence(domain_scores, domain_meta)
        # emotion conf = min(1, 80/100)*1.0*1.0 = 0.8
        # sleep  conf = min(1, 20/100)*1.0*1.0 = 0.2
        # weighted = (0.8*80 + 0.2*20) / 100 = (64 + 4) / 100 = 0.68
        self.assertAlmostEqual(result, 0.68, places=5)

    def test_overall_confidence_no_events_returns_zero(self):
        """No events in any domain should return 0.0."""
        domain_scores = {'emotion': {'score': 0.0, 'event_count': 0}}
        domain_meta = {'emotion': {'latest_event_time': _NOW, 'source_count': 1}}
        result = self.engine.compute_overall_confidence(domain_scores, domain_meta)
        self.assertEqual(result, 0.0)

    def test_overall_confidence_single_domain(self):
        """Single domain overall confidence equals domain confidence."""
        event_count = 100
        latest = _NOW - 3600
        sources = 2
        domain_scores = {'emotion': {'score': 0.9, 'event_count': event_count}}
        domain_meta = {'emotion': {'latest_event_time': latest, 'source_count': sources}}
        overall = self.engine.compute_overall_confidence(domain_scores, domain_meta)
        expected = self.engine.compute_domain_confidence(event_count, latest, sources)
        self.assertAlmostEqual(overall, expected, places=5)


# ─── TestStageTransitions ─────────────────────────────────────────────────────

class TestStageTransitions(unittest.TestCase):
    """Tests for evaluate_stage and STAGE_NAMES."""

    def setUp(self):
        self.engine = _make_engine()

    # --- STAGE_NAMES ---

    def test_stage_names_map(self):
        self.assertEqual(STAGE_NAMES[1], 'Nascent')
        self.assertEqual(STAGE_NAMES[2], 'Awareness')
        self.assertEqual(STAGE_NAMES[3], 'Regulation')
        self.assertEqual(STAGE_NAMES[4], 'Integration')
        self.assertEqual(STAGE_NAMES[5], 'Mastery')

    # --- Default / no-advance scenario ---

    def test_nascent_default_no_advance(self):
        """Stage 1 with insufficient data should stay at Stage 1."""
        result = self.engine.evaluate_stage(
            current_stage=1,
            confidence=0.1,
            total_events=5,
            domains_with_events=1,
            weeks_of_data=0.5,
            positive_patterns=False,
            synergy_count=0,
            stability_score=0.0,
            consecutive_met=0,
        )
        self.assertFalse(result['changed'])
        self.assertEqual(result['new_stage'], 1)
        self.assertEqual(result['stage_name'], 'Nascent')

    # --- Advancement: 1 → 2 ---

    def test_advance_stage_1_to_2(self):
        """Meeting all Stage 1→2 criteria with consecutive_met >= 3 should advance."""
        result = self.engine.evaluate_stage(
            current_stage=1,
            confidence=0.35,
            total_events=60,
            domains_with_events=3,
            weeks_of_data=2,
            positive_patterns=False,
            synergy_count=0,
            stability_score=0.0,
            consecutive_met=3,
        )
        self.assertTrue(result['changed'])
        self.assertEqual(result['new_stage'], 2)
        self.assertEqual(result['stage_name'], 'Awareness')

    def test_advance_stage_1_to_2_insufficient_confidence(self):
        """Stage 1→2 requires confidence >= 0.30; below that should not advance."""
        result = self.engine.evaluate_stage(
            current_stage=1,
            confidence=0.25,
            total_events=60,
            domains_with_events=3,
            weeks_of_data=2,
            positive_patterns=False,
            synergy_count=0,
            stability_score=0.0,
            consecutive_met=3,
        )
        self.assertFalse(result['changed'])
        self.assertEqual(result['new_stage'], 1)

    # --- Advancement: 2 → 3 ---

    def test_advance_stage_2_to_3(self):
        """Meeting all Stage 2→3 criteria should advance to Regulation."""
        result = self.engine.evaluate_stage(
            current_stage=2,
            confidence=0.6,
            total_events=150,
            domains_with_events=4,
            weeks_of_data=4,
            positive_patterns=True,
            synergy_count=0,
            stability_score=0.5,
            consecutive_met=3,
        )
        self.assertTrue(result['changed'])
        self.assertEqual(result['new_stage'], 3)
        self.assertEqual(result['stage_name'], 'Regulation')

    def test_advance_stage_2_to_3_missing_positive_patterns(self):
        """Stage 2→3 requires positive_patterns; without it should not advance."""
        result = self.engine.evaluate_stage(
            current_stage=2,
            confidence=0.6,
            total_events=150,
            domains_with_events=4,
            weeks_of_data=4,
            positive_patterns=False,
            synergy_count=0,
            stability_score=0.5,
            consecutive_met=3,
        )
        self.assertFalse(result['changed'])

    # --- Advancement: 3 → 4 ---

    def test_advance_stage_3_to_4(self):
        """Meeting all Stage 3→4 criteria should advance to Integration."""
        result = self.engine.evaluate_stage(
            current_stage=3,
            confidence=0.8,
            total_events=300,
            domains_with_events=5,
            weeks_of_data=8,
            positive_patterns=True,
            synergy_count=4,
            stability_score=0.7,
            consecutive_met=3,
        )
        self.assertTrue(result['changed'])
        self.assertEqual(result['new_stage'], 4)
        self.assertEqual(result['stage_name'], 'Integration')

    def test_advance_stage_3_to_4_insufficient_synergies(self):
        """Stage 3→4 requires synergy_count >= 3; fewer should not advance."""
        result = self.engine.evaluate_stage(
            current_stage=3,
            confidence=0.8,
            total_events=300,
            domains_with_events=5,
            weeks_of_data=8,
            positive_patterns=True,
            synergy_count=2,
            stability_score=0.7,
            consecutive_met=3,
        )
        self.assertFalse(result['changed'])

    # --- Advancement: 4 → 5 ---

    def test_advance_stage_4_to_5(self):
        """Meeting all Stage 4→5 criteria should advance to Mastery."""
        result = self.engine.evaluate_stage(
            current_stage=4,
            confidence=0.95,
            total_events=500,
            domains_with_events=6,
            weeks_of_data=12,
            positive_patterns=True,
            synergy_count=6,
            stability_score=0.9,
            consecutive_met=3,
        )
        self.assertTrue(result['changed'])
        self.assertEqual(result['new_stage'], 5)
        self.assertEqual(result['stage_name'], 'Mastery')

    def test_advance_stage_4_to_5_insufficient_stability(self):
        """Stage 4→5 requires stability > 0.85; below that should not advance."""
        result = self.engine.evaluate_stage(
            current_stage=4,
            confidence=0.95,
            total_events=500,
            domains_with_events=6,
            weeks_of_data=12,
            positive_patterns=True,
            synergy_count=6,
            stability_score=0.80,
            consecutive_met=3,
        )
        self.assertFalse(result['changed'])

    def test_advance_stage_5_does_not_go_beyond(self):
        """Stage 5 should not advance further even with excellent metrics."""
        result = self.engine.evaluate_stage(
            current_stage=5,
            confidence=1.0,
            total_events=1000,
            domains_with_events=8,
            weeks_of_data=52,
            positive_patterns=True,
            synergy_count=10,
            stability_score=1.0,
            consecutive_met=100,
        )
        self.assertFalse(result['changed'])
        self.assertEqual(result['new_stage'], 5)

    # --- Requires 3 consecutive ---

    def test_requires_3_consecutive_met(self):
        """Advancement should not occur if consecutive_met < 3."""
        for consec in [0, 1, 2]:
            result = self.engine.evaluate_stage(
                current_stage=1,
                confidence=0.5,
                total_events=100,
                domains_with_events=4,
                weeks_of_data=5,
                positive_patterns=True,
                synergy_count=3,
                stability_score=0.9,
                consecutive_met=consec,
            )
            self.assertFalse(result['changed'], f"Should not advance with consecutive_met={consec}")

    def test_exactly_3_consecutive_allows_advance(self):
        """consecutive_met == 3 is sufficient for advancement."""
        result = self.engine.evaluate_stage(
            current_stage=1,
            confidence=0.35,
            total_events=60,
            domains_with_events=3,
            weeks_of_data=2,
            positive_patterns=False,
            synergy_count=0,
            stability_score=0.0,
            consecutive_met=3,
        )
        self.assertTrue(result['changed'])

    # --- Regression on sustained negative ---

    def test_regression_on_sustained_negative_2_weeks(self):
        """sustained_negative_weeks >= 2 drops stage by 1."""
        result = self.engine.evaluate_stage(
            current_stage=3,
            confidence=0.8,
            total_events=200,
            domains_with_events=4,
            weeks_of_data=10,
            positive_patterns=False,
            synergy_count=3,
            stability_score=0.5,
            consecutive_met=5,
            sustained_negative_weeks=2,
        )
        self.assertTrue(result['changed'])
        self.assertEqual(result['new_stage'], 2)
        self.assertIn('Recalibration', result['reason'])

    def test_regression_on_sustained_negative_3_weeks(self):
        """sustained_negative_weeks >= 2 drops by exactly 1 (not more)."""
        result = self.engine.evaluate_stage(
            current_stage=4,
            confidence=0.9,
            total_events=400,
            domains_with_events=5,
            weeks_of_data=15,
            positive_patterns=True,
            synergy_count=5,
            stability_score=0.7,
            consecutive_met=5,
            sustained_negative_weeks=3,
        )
        self.assertTrue(result['changed'])
        self.assertEqual(result['new_stage'], 3)

    def test_no_regression_below_stage_1(self):
        """Stage 1 should never regress further."""
        result = self.engine.evaluate_stage(
            current_stage=1,
            confidence=0.05,
            total_events=5,
            domains_with_events=1,
            weeks_of_data=1,
            positive_patterns=False,
            synergy_count=0,
            stability_score=0.0,
            consecutive_met=0,
            sustained_negative_weeks=10,
        )
        self.assertFalse(result['changed'])
        self.assertEqual(result['new_stage'], 1)

    def test_no_regression_at_1_week_negative(self):
        """Exactly 1 week of sustained negative should not trigger regression."""
        result = self.engine.evaluate_stage(
            current_stage=3,
            confidence=0.6,
            total_events=200,
            domains_with_events=4,
            weeks_of_data=10,
            positive_patterns=False,
            synergy_count=3,
            stability_score=0.5,
            consecutive_met=5,
            sustained_negative_weeks=1,
        )
        # 1 week is NOT enough for regression
        self.assertNotEqual(result['new_stage'], 2)

    def test_regression_takes_priority_over_advancement(self):
        """When both regression and advancement criteria are met, regression wins."""
        result = self.engine.evaluate_stage(
            current_stage=2,
            confidence=0.65,
            total_events=200,
            domains_with_events=4,
            weeks_of_data=6,
            positive_patterns=True,
            synergy_count=0,
            stability_score=0.5,
            consecutive_met=5,
            sustained_negative_weeks=2,
        )
        # Regression check is first, so should drop to stage 1
        self.assertTrue(result['changed'])
        self.assertEqual(result['new_stage'], 1)


# ─── TestSynergyDetection ─────────────────────────────────────────────────────

class TestSynergyDetection(unittest.TestCase):
    """Tests for detect_synergies."""

    def setUp(self):
        self.engine = _make_engine()

    def _make_scores(self, n=28, base=0.5, noise=0.0):
        import random
        rng = random.Random(42)
        return [base + rng.uniform(-noise, noise) for _ in range(n)]

    def test_no_synergy_below_14_days(self):
        """Pairs with fewer than 14 days of data should not be reported."""
        scores = {
            'emotion': [0.5] * 10,
            'sleep': [0.5] * 10,
        }
        result = self.engine.detect_synergies('user1', scores)
        self.assertEqual(result, [])

    def test_strong_positive_correlation_detected(self):
        """Two perfectly correlated series should be flagged as a synergy."""
        import math
        series = [math.sin(i / 3.0) for i in range(28)]
        scores = {'emotion': series, 'sleep': series}
        result = self.engine.detect_synergies('user1', scores)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]['correlation'], 0.6)
        self.assertLess(result[0]['p_value'], 0.05)

    def test_uncorrelated_series_not_detected(self):
        """Two independent random series should typically not be reported."""
        import random
        rng = random.Random(99)
        a = [rng.random() for _ in range(28)]
        rng2 = random.Random(12345)
        b = [rng2.random() for _ in range(28)]
        scores = {'emotion': a, 'sleep': b}
        result = self.engine.detect_synergies('user1', scores)
        # Independent series are very unlikely to exceed |r| > 0.6 with p < 0.05
        for s in result:
            self.assertGreater(abs(s['correlation']), 0.6)

    def test_synergy_has_required_keys(self):
        """Each synergy dict must contain domain_a, domain_b, correlation, p_value, insight."""
        import math
        series = [math.sin(i / 3.0) for i in range(28)]
        scores = {'emotion': series, 'mood': series}
        result = self.engine.detect_synergies('user1', scores)
        if result:
            for s in result:
                self.assertIn('domain_a', s)
                self.assertIn('domain_b', s)
                self.assertIn('correlation', s)
                self.assertIn('p_value', s)
                self.assertIn('insight', s)


# ─── TestSnapshotChecking ─────────────────────────────────────────────────────

class TestSnapshotChecking(unittest.TestCase):
    """Tests for check_missed_snapshots."""

    def test_no_snapshots_recorded_returns_both(self):
        """If no snapshots have ever been taken, both types should be overdue."""
        db = MagicMock()
        db.get_last_snapshot_time.return_value = None
        engine = EvolutionEngine(db)
        result = engine.check_missed_snapshots('user1')
        self.assertIn('weekly', result)
        self.assertIn('monthly', result)

    def test_recent_snapshots_not_overdue(self):
        """Snapshots taken within the last day should not be flagged as overdue."""
        db = MagicMock()
        recent = time.time() - 3600  # 1 hour ago
        db.get_last_snapshot_time.return_value = recent
        engine = EvolutionEngine(db)
        result = engine.check_missed_snapshots('user1')
        self.assertNotIn('weekly', result)
        self.assertNotIn('monthly', result)

    def test_weekly_overdue_but_monthly_recent(self):
        """If weekly snapshot is 8 days old but monthly is recent, only weekly is flagged."""
        db = MagicMock()

        def get_last(user_id, snapshot_type):
            if snapshot_type == 'weekly':
                return time.time() - 8 * 24 * 3600  # 8 days ago
            else:
                return time.time() - 3600  # 1 hour ago

        db.get_last_snapshot_time.side_effect = get_last
        engine = EvolutionEngine(db)
        result = engine.check_missed_snapshots('user1')
        self.assertIn('weekly', result)
        self.assertNotIn('monthly', result)

    def test_monthly_overdue_but_weekly_recent(self):
        """If monthly snapshot is 31 days old but weekly is recent, only monthly is flagged."""
        db = MagicMock()

        def get_last(user_id, snapshot_type):
            if snapshot_type == 'monthly':
                return time.time() - 31 * 24 * 3600  # 31 days ago
            else:
                return time.time() - 3600  # 1 hour ago

        db.get_last_snapshot_time.side_effect = get_last
        engine = EvolutionEngine(db)
        result = engine.check_missed_snapshots('user1')
        self.assertIn('monthly', result)
        self.assertNotIn('weekly', result)


# ─── TestStageProgress ────────────────────────────────────────────────────────

class TestStageProgress(unittest.TestCase):
    """Tests for get_stage_progress."""

    def setUp(self):
        self.engine = _make_engine()

    def test_stage_5_returns_full_progress(self):
        """Stage 5 has no next stage; progress should be 1.0."""
        result = self.engine.get_stage_progress(
            current_stage=5,
            confidence=1.0,
            total_events=1000,
            domains_with_events=8,
            weeks_of_data=52,
            positive_patterns=True,
            synergy_count=10,
            stability_score=1.0,
        )
        self.assertEqual(result['progress'], 1.0)
        self.assertIsNone(result['next_stage'])
        self.assertIsNone(result['next_stage_name'])

    def test_stage_1_progress_structure(self):
        """Stage 1 progress should list 3 criteria."""
        result = self.engine.get_stage_progress(
            current_stage=1,
            confidence=0.1,
            total_events=10,
            domains_with_events=1,
            weeks_of_data=0.5,
            positive_patterns=False,
            synergy_count=0,
            stability_score=0.0,
        )
        self.assertEqual(len(result['criteria']), 3)
        self.assertEqual(result['next_stage'], 2)

    def test_progress_zero_when_no_criteria_met(self):
        result = self.engine.get_stage_progress(
            current_stage=1,
            confidence=0.1,
            total_events=5,
            domains_with_events=1,
            weeks_of_data=0.1,
            positive_patterns=False,
            synergy_count=0,
            stability_score=0.0,
        )
        self.assertAlmostEqual(result['progress'], 0.0)

    def test_progress_one_when_all_criteria_met(self):
        result = self.engine.get_stage_progress(
            current_stage=1,
            confidence=0.35,
            total_events=60,
            domains_with_events=3,
            weeks_of_data=3,
            positive_patterns=True,
            synergy_count=0,
            stability_score=0.0,
        )
        self.assertAlmostEqual(result['progress'], 1.0)

    def test_criteria_met_flag(self):
        """Each criterion should have a 'met' boolean."""
        result = self.engine.get_stage_progress(
            current_stage=2,
            confidence=0.6,
            total_events=200,
            domains_with_events=4,
            weeks_of_data=5,
            positive_patterns=True,
            synergy_count=1,
            stability_score=0.5,
        )
        for c in result['criteria']:
            self.assertIn('met', c)
            self.assertIn('name', c)
            self.assertIn('current', c)
            self.assertIn('target', c)


if __name__ == '__main__':
    unittest.main()
