"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Evolution Engine
===============================================================================
User profile evolution tracking: stage advancement, confidence scoring,
synergy detection, and snapshot management.

Integrates with:
- Neural Workflow AI Engine
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.

Components:
  STAGE_NAMES          - Stage index to human-readable name mapping
  EvolutionEngine      - Core engine: confidence, stage transitions, synergies
===============================================================================
"""

import time
from typing import Dict, List, Optional

STAGE_NAMES: Dict[int, str] = {
    1: 'Nascent',
    2: 'Awareness',
    3: 'Regulation',
    4: 'Integration',
    5: 'Mastery',
}

_SECONDS_PER_WEEK = 7 * 24 * 3600


class EvolutionEngine:
    """Core engine for user profile evolution across five stages.

    Tracks confidence, stage transitions, cross-domain synergies, and
    snapshot scheduling for the Quantara User Profile Engine.
    """

    def __init__(self, db):
        """Initialise with a ProfileDB instance.

        Args:
            db: ProfileDB — provides get_or_create_profile(),
                get_last_snapshot_time(), etc.
        """
        self.db = db

    # ─── Confidence ──────────────────────────────────────────────────────────

    def compute_domain_confidence(
        self,
        event_count: int,
        latest_event_time: float,
        source_count: int,
    ) -> float:
        """Compute [0, 1] confidence for a single domain.

        Formula: min(1.0, event_count / 100) * recency_factor * source_factor

        recency_factor:
            1.0 if latest event was within the last 7 days,
            decays 0.1 per full week beyond that, floor at 0.3.

        source_factor:
            1.0 if source_count >= 2, else 0.8.

        Returns 0.0 when event_count == 0.
        """
        if event_count == 0:
            return 0.0

        volume_factor = min(1.0, event_count / 100)

        now = time.time()
        age_seconds = now - latest_event_time
        weeks_old = age_seconds / _SECONDS_PER_WEEK
        if weeks_old <= 1.0:
            recency_factor = 1.0
        else:
            # Each full week beyond 1 reduces by 0.1
            extra_weeks = int(weeks_old - 1.0)
            recency_factor = max(0.3, 1.0 - 0.1 * extra_weeks)

        source_factor = 1.0 if source_count >= 2 else 0.8

        return volume_factor * recency_factor * source_factor

    def compute_overall_confidence(
        self,
        domain_scores: Dict[str, Dict],
        domain_meta: Dict[str, Dict],
    ) -> float:
        """Compute weighted-mean confidence across all domains.

        Args:
            domain_scores: {domain: {'score': float, 'event_count': int}}
            domain_meta:   {domain: {'latest_event_time': float, 'source_count': int}}

        Returns weighted mean of per-domain confidences, weighted by event_count.
        Returns 0.0 if there are no events across all domains.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for domain, ds in domain_scores.items():
            event_count = ds.get('event_count', 0)
            if event_count == 0:
                continue
            meta = domain_meta.get(domain, {})
            latest = meta.get('latest_event_time', time.time())
            sources = meta.get('source_count', 1)
            conf = self.compute_domain_confidence(event_count, latest, sources)
            weighted_sum += conf * event_count
            total_weight += event_count

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    # ─── Stage Evaluation ────────────────────────────────────────────────────

    def evaluate_stage(
        self,
        current_stage: int,
        confidence: float,
        total_events: int,
        domains_with_events: int,
        weeks_of_data: float,
        positive_patterns: bool,
        synergy_count: int,
        stability_score: float,
        consecutive_met: int,
        sustained_negative_weeks: int = 0,
    ) -> Dict:
        """Evaluate whether the user should advance, regress, or stay at their stage.

        Regression is checked before advancement. If the user has been in a
        sustained negative pattern for 2+ weeks and is above Stage 1, they
        drop one stage (framed as supportive recalibration).

        Advancement criteria (all require consecutive_met >= 3):
            1 → 2 : total_events >= 50, domains_with_events >= 3, confidence >= 0.30
            2 → 3 : weeks_of_data >= 4, positive_patterns, confidence > 0.55
            3 → 4 : synergy_count >= 3, weeks_of_data >= 8, confidence > 0.75
            4 → 5 : weeks_of_data >= 12, stability_score > 0.85, confidence > 0.90

        Returns:
            {
                'new_stage': int,
                'changed': bool,
                'reason': str,
                'stage_name': str,
            }
        """
        # --- Regression check ---
        if current_stage > 1 and sustained_negative_weeks >= 2:
            new_stage = current_stage - 1
            return {
                'new_stage': new_stage,
                'changed': True,
                'reason': (
                    'Recalibration: a period of challenge is an opportunity to '
                    'rebuild foundation. You are not going backwards — you are '
                    'deepening your roots.'
                ),
                'stage_name': STAGE_NAMES[new_stage],
            }

        # --- Advancement check ---
        if consecutive_met < 3:
            return {
                'new_stage': current_stage,
                'changed': False,
                'reason': 'Criteria not yet sustained for 3 consecutive periods.',
                'stage_name': STAGE_NAMES[current_stage],
            }

        advanced = False
        reason = ''

        if current_stage == 1:
            if total_events >= 50 and domains_with_events >= 3 and confidence >= 0.30:
                advanced = True
                reason = (
                    f'Reached {total_events} events across {domains_with_events} domains '
                    f'with confidence {confidence:.2f}.'
                )
        elif current_stage == 2:
            if weeks_of_data >= 4 and positive_patterns and confidence > 0.55:
                advanced = True
                reason = (
                    f'{weeks_of_data:.1f} weeks of data with positive patterns '
                    f'and confidence {confidence:.2f}.'
                )
        elif current_stage == 3:
            if synergy_count >= 3 and weeks_of_data >= 8 and confidence > 0.75:
                advanced = True
                reason = (
                    f'{synergy_count} cross-domain synergies detected over '
                    f'{weeks_of_data:.1f} weeks with confidence {confidence:.2f}.'
                )
        elif current_stage == 4:
            if weeks_of_data >= 12 and stability_score > 0.85 and confidence > 0.90:
                advanced = True
                reason = (
                    f'Stability {stability_score:.2f} over {weeks_of_data:.1f} weeks '
                    f'with confidence {confidence:.2f}.'
                )

        if advanced and current_stage < 5:
            new_stage = current_stage + 1
            return {
                'new_stage': new_stage,
                'changed': True,
                'reason': reason,
                'stage_name': STAGE_NAMES[new_stage],
            }

        return {
            'new_stage': current_stage,
            'changed': False,
            'reason': 'Advancement criteria not yet fully met.',
            'stage_name': STAGE_NAMES[current_stage],
        }

    # ─── Synergy Detection ───────────────────────────────────────────────────

    def detect_synergies(
        self,
        user_id: str,
        daily_domain_scores: Dict[str, List[float]],
    ) -> List[Dict]:
        """Detect statistically significant correlations between domain score series.

        For each pair of domains with 14+ days of overlapping data, compute
        Pearson r. If |r| > 0.6 and p_value < 0.05 the pair is reported as a
        synergy.

        Args:
            user_id: Identifier for logging/context (not used for filtering here).
            daily_domain_scores: {domain: [28 daily float scores]}

        Returns:
            List of dicts with keys:
                domain_a, domain_b, correlation (float), p_value (float), insight (str)
        """
        try:
            from scipy.stats import pearsonr
        except ImportError:
            return []

        domains = list(daily_domain_scores.keys())
        synergies: List[Dict] = []

        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                da = domains[i]
                db = domains[j]
                scores_a = daily_domain_scores[da]
                scores_b = daily_domain_scores[db]

                # Only consider days where both series have data
                pairs = [
                    (a, b)
                    for a, b in zip(scores_a, scores_b)
                    if a is not None and b is not None
                ]
                if len(pairs) < 14:
                    continue

                xa = [p[0] for p in pairs]
                xb = [p[1] for p in pairs]

                try:
                    corr, p_val = pearsonr(xa, xb)
                except Exception:
                    continue

                if abs(corr) > 0.6 and p_val < 0.05:
                    direction = 'co-elevate' if corr > 0 else 'inversely balance'
                    insight = (
                        f'{da.capitalize()} and {db.capitalize()} tend to '
                        f'{direction} (r={corr:.2f}, p={p_val:.4f}). '
                        'Strengthening one domain may benefit the other.'
                    )
                    synergies.append({
                        'domain_a': da,
                        'domain_b': db,
                        'correlation': corr,
                        'p_value': p_val,
                        'insight': insight,
                    })

        return synergies

    # ─── Snapshot Management ─────────────────────────────────────────────────

    def check_missed_snapshots(self, user_id: str) -> List[str]:
        """Return a list of snapshot types that are overdue for the given user.

        A weekly snapshot is overdue if the last one was taken more than 7 days ago.
        A monthly snapshot is overdue if the last one was taken more than 30 days ago.

        Returns:
            List of strings from {'weekly', 'monthly'} that need to be caught up.
        """
        now = time.time()
        overdue: List[str] = []

        weekly_last = self.db.get_last_snapshot_time(user_id, 'weekly')
        if weekly_last is None or (now - weekly_last) > 7 * 24 * 3600:
            overdue.append('weekly')

        monthly_last = self.db.get_last_snapshot_time(user_id, 'monthly')
        if monthly_last is None or (now - monthly_last) > 30 * 24 * 3600:
            overdue.append('monthly')

        return overdue

    # ─── Stage Progress ──────────────────────────────────────────────────────

    def get_stage_progress(
        self,
        current_stage: int,
        confidence: float,
        total_events: int,
        domains_with_events: int,
        weeks_of_data: float,
        positive_patterns: bool,
        synergy_count: int,
        stability_score: float,
    ) -> Dict:
        """Return a progress checklist toward the next stage.

        Each criterion is represented as:
            {'name': str, 'met': bool, 'current': value, 'target': value}

        Overall progress = met_count / total_criteria.

        Returns the checklist dict with:
            'criteria': list of criterion dicts
            'progress': float in [0, 1]
            'next_stage': int or None if already at max
            'next_stage_name': str or None
        """
        if current_stage >= 5:
            return {
                'criteria': [],
                'progress': 1.0,
                'next_stage': None,
                'next_stage_name': None,
            }

        next_stage = current_stage + 1

        if current_stage == 1:
            criteria = [
                {
                    'name': 'Total events',
                    'met': total_events >= 50,
                    'current': total_events,
                    'target': 50,
                },
                {
                    'name': 'Domains with events',
                    'met': domains_with_events >= 3,
                    'current': domains_with_events,
                    'target': 3,
                },
                {
                    'name': 'Confidence',
                    'met': confidence >= 0.30,
                    'current': round(confidence, 3),
                    'target': 0.30,
                },
            ]
        elif current_stage == 2:
            criteria = [
                {
                    'name': 'Weeks of data',
                    'met': weeks_of_data >= 4,
                    'current': round(weeks_of_data, 1),
                    'target': 4,
                },
                {
                    'name': 'Positive patterns detected',
                    'met': positive_patterns,
                    'current': positive_patterns,
                    'target': True,
                },
                {
                    'name': 'Confidence',
                    'met': confidence > 0.55,
                    'current': round(confidence, 3),
                    'target': 0.55,
                },
            ]
        elif current_stage == 3:
            criteria = [
                {
                    'name': 'Cross-domain synergies',
                    'met': synergy_count >= 3,
                    'current': synergy_count,
                    'target': 3,
                },
                {
                    'name': 'Weeks of data',
                    'met': weeks_of_data >= 8,
                    'current': round(weeks_of_data, 1),
                    'target': 8,
                },
                {
                    'name': 'Confidence',
                    'met': confidence > 0.75,
                    'current': round(confidence, 3),
                    'target': 0.75,
                },
            ]
        elif current_stage == 4:
            criteria = [
                {
                    'name': 'Weeks of data',
                    'met': weeks_of_data >= 12,
                    'current': round(weeks_of_data, 1),
                    'target': 12,
                },
                {
                    'name': 'Stability score',
                    'met': stability_score > 0.85,
                    'current': round(stability_score, 3),
                    'target': 0.85,
                },
                {
                    'name': 'Confidence',
                    'met': confidence > 0.90,
                    'current': round(confidence, 3),
                    'target': 0.90,
                },
            ]
        else:
            criteria = []

        met_count = sum(1 for c in criteria if c['met'])
        progress = met_count / len(criteria) if criteria else 0.0

        return {
            'criteria': criteria,
            'progress': progress,
            'next_stage': next_stage,
            'next_stage_name': STAGE_NAMES[next_stage],
        }
