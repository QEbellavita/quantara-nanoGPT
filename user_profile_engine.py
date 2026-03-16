"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - User Profile Engine
===============================================================================
Central orchestrator for the genetic fingerprint system.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from biometrics, emotions, therapy sessions
===============================================================================
"""

import dataclasses
import json
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from profile_db import ProfileDB
from evolution_engine import EvolutionEngine, STAGE_NAMES
from domain_processors import get_all_processors

logger = logging.getLogger(__name__)

# Biometric event types subject to rate limiting (1 per minute per domain/user)
_RATE_LIMITED_EVENTS = frozenset({
    'hr_reading', 'hrv_reading', 'eda_reading', 'breathing_reading',
})

_SECONDS_PER_WEEK = 7 * 24 * 3600


@dataclasses.dataclass
class ProfileSnapshot:
    """Lightweight in-memory summary of a user's emotional profile.
    Updated incrementally on every emotion_classified event.
    Used by the personalization tiebreaker in emotion_classifier.py.
    """
    user_id: str
    evolution_stage: int
    overall_confidence: float
    emotion_prior: Dict[str, float]
    dominant_family: str
    event_count: int
    family_counts: Dict[str, int]
    last_updated: float


class UserProfileEngine:
    """Central orchestrator for the Quantara User Profile / Genetic Fingerprint system.

    Wires together ProfileDB, EvolutionEngine, and all domain processors to
    produce a holistic user fingerprint from raw events.
    """

    def __init__(self, db_path: str = 'data/profile.db'):
        self.db = ProfileDB(db_path)
        self.evolution = EvolutionEngine(self.db)
        self.processors: Dict[str, Any] = {
            p.domain: p for p in get_all_processors()
        }
        # Rate limit cache: (user_id, domain) -> last_timestamp
        self._rate_cache: Dict[tuple, float] = {}
        self._rate_lock = threading.Lock()
        # Consecutive-met counter per user for stage evaluation
        self._consecutive_met: Dict[str, int] = {}
        self._closed = False
        self._event_bus = None
        self._intelligence_publisher = None
        self._alert_engine = None
        self._ecosystem_connector = None
        self._snapshots: Dict[str, 'ProfileSnapshot'] = {}
        self._snapshot_lock = threading.Lock()

    # ─── Event Ingestion ─────────────────────────────────────────────────

    def log_event(
        self,
        user_id: str,
        domain: str,
        event_type: str,
        payload: Optional[dict] = None,
        source: str = 'nanogpt',
        confidence: Optional[float] = None,
    ) -> Optional[int]:
        """Log a raw event into the profile database.

        Returns the event_id on success, or None if the event was skipped
        (rate-limited, engine closed, or error).
        """
        if self._closed:
            return None

        try:
            # Rate-limit biometric high-frequency events
            if event_type in _RATE_LIMITED_EVENTS:
                now = time.time()
                key = (user_id, domain)
                with self._rate_lock:
                    last = self._rate_cache.get(key)
                    if last is not None and (now - last) < 60.0:
                        return None
                    self._rate_cache[key] = now

            # Ensure profile exists
            self.db.get_or_create_profile(user_id)

            # Log the event
            event_id = self.db.log_event(
                user_id=user_id,
                domain=domain,
                event_type=event_type,
                payload=payload,
                source=source,
                confidence=confidence if confidence is not None else 1.0,
            )
            # Update in-memory snapshot for personalization
            if domain == 'emotional' and event_type == 'emotion_classified':
                family = (payload or {}).get('family')
                if family:
                    self._update_snapshot(user_id, family)
            return event_id
        except Exception:
            logger.exception("Error logging event for user %s", user_id)
            return None

    def _update_snapshot(self, user_id: str, family: str) -> None:
        """Incrementally update the in-memory profile snapshot."""
        profile_data = None
        with self._snapshot_lock:
            snap = self._snapshots.get(user_id)
            if snap is not None:
                snap.family_counts[family] = snap.family_counts.get(family, 0) + 1
                snap.event_count += 1
                total = sum(snap.family_counts.values())
                snap.emotion_prior = {f: c / total for f, c in snap.family_counts.items()}
                snap.dominant_family = max(snap.family_counts, key=snap.family_counts.get)
                snap.last_updated = time.time()
                return

        # Slow path: create new snapshot (outside lock)
        profile_data = self.db.get_or_create_profile(user_id)
        with self._snapshot_lock:
            if user_id in self._snapshots:
                snap = self._snapshots[user_id]
                snap.family_counts[family] = snap.family_counts.get(family, 0) + 1
                snap.event_count += 1
                total = sum(snap.family_counts.values())
                snap.emotion_prior = {f: c / total for f, c in snap.family_counts.items()}
                snap.dominant_family = max(snap.family_counts, key=snap.family_counts.get)
                snap.last_updated = time.time()
                return
            snap = ProfileSnapshot(
                user_id=user_id,
                evolution_stage=profile_data.get('evolution_stage', 1),
                overall_confidence=profile_data.get('confidence', 0.0),
                emotion_prior={family: 1.0},
                dominant_family=family,
                event_count=1,
                family_counts={family: 1},
                last_updated=time.time(),
            )
            self._snapshots[user_id] = snap

    def get_profile_snapshot(self, user_id: str) -> Optional['ProfileSnapshot']:
        """Return the cached profile snapshot, rebuilding from DB if needed.
        Returns None only if user has zero emotional events.
        Does NOT conflict with get_snapshot() which returns DB snapshots.
        """
        with self._snapshot_lock:
            snap = self._snapshots.get(user_id)
            if snap is not None:
                return snap

        # Rebuild from DB outside the lock
        events = self.db.get_events(user_id, domain='emotional', limit=10000)
        family_counts: Dict[str, int] = {}
        for evt in events:
            if evt.get('event_type') != 'emotion_classified':
                continue
            payload = evt.get('payload')
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    continue
            family = (payload or {}).get('family')
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1

        if not family_counts:
            return None

        total = sum(family_counts.values())
        profile = self.db.get_or_create_profile(user_id)
        snap = ProfileSnapshot(
            user_id=user_id,
            evolution_stage=profile.get('evolution_stage', 1),
            overall_confidence=profile.get('confidence', 0.0),
            emotion_prior={f: c / total for f, c in family_counts.items()},
            dominant_family=max(family_counts, key=family_counts.get),
            event_count=total,
            family_counts=family_counts,
            last_updated=time.time(),
        )

        with self._snapshot_lock:
            if user_id not in self._snapshots:
                self._snapshots[user_id] = snap
            return self._snapshots[user_id]

    # ─── Core Processing ─────────────────────────────────────────────────

    def process(self, user_id: str) -> Dict[str, Any]:
        """Run the full fingerprint computation pipeline for a user.

        Returns a fingerprint dict with domain scores, confidence, stage info,
        synergies, and metadata.
        """
        profile = self.db.get_or_create_profile(user_id)
        current_stage = profile.get('evolution_stage', 1)
        created_at = profile.get('created_at', time.time())

        domain_scores: Dict[str, Dict] = {}
        domain_meta: Dict[str, Dict] = {}

        # Process each domain
        for domain_name, processor in self.processors.items():
            events = self.db.get_events(user_id, domain=domain_name, limit=10000)
            # Reverse to oldest-first for processor computation
            events = list(reversed(events))

            if events:
                result = processor.compute(events)
            else:
                result = processor.get_empty_score()

            domain_scores[domain_name] = result

            # Collect metadata for confidence computation
            latest_time = self.db.get_latest_event_time(user_id, domain=domain_name)
            sources = self.db.get_domain_sources(user_id, domain_name)
            domain_meta[domain_name] = {
                'latest_event_time': latest_time if latest_time else time.time(),
                'source_count': len(sources),
            }

        # Overall confidence
        confidence = self.evolution.compute_overall_confidence(domain_scores, domain_meta)

        # Stage evaluation inputs
        total_events = sum(ds.get('event_count', 0) for ds in domain_scores.values())
        domains_with_events = sum(
            1 for ds in domain_scores.values() if ds.get('event_count', 0) > 0
        )
        weeks_of_data = (time.time() - created_at) / _SECONDS_PER_WEEK

        # Positive patterns: check if any domain has recovery_rate > 0.1
        positive_patterns = False
        for ds in domain_scores.values():
            metrics = ds.get('metrics', {})
            if metrics.get('recovery_rate', 0) > 0.1:
                positive_patterns = True
                break

        # Synergy count from DB
        existing_synergies = self.db.get_synergies(user_id)
        synergy_count = len(existing_synergies)

        # Stability: 1 - average volatility across domains
        volatilities = []
        for ds in domain_scores.values():
            metrics = ds.get('metrics', {})
            v = metrics.get('volatility')
            if v is not None:
                volatilities.append(v)
        avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0.0
        stability = 1.0 - avg_volatility

        # Consecutive met counter
        consecutive_met = self._consecutive_met.get(user_id, 0)

        # Evaluate stage
        stage_result = self.evolution.evaluate_stage(
            current_stage=current_stage,
            confidence=confidence,
            total_events=total_events,
            domains_with_events=domains_with_events,
            weeks_of_data=weeks_of_data,
            positive_patterns=positive_patterns,
            synergy_count=synergy_count,
            stability_score=stability,
            consecutive_met=consecutive_met,
        )

        # Update consecutive met counter
        # If criteria are being met (stage would advance if consecutive >= 3),
        # increment; otherwise reset
        if stage_result['changed'] or (not stage_result['changed'] and
                                        stage_result['reason'] == 'Criteria not yet sustained for 3 consecutive periods.'):
            # Criteria are met but not sustained yet, or stage changed
            if not stage_result['changed']:
                self._consecutive_met[user_id] = consecutive_met + 1
            else:
                self._consecutive_met[user_id] = 0
        else:
            self._consecutive_met[user_id] = 0

        new_stage = stage_result['new_stage']

        # Synergy detection via daily domain scores
        daily_scores = self._get_daily_domain_scores(user_id)
        detected_synergies = self.evolution.detect_synergies(user_id, daily_scores)
        for syn in detected_synergies:
            self.db.save_synergy(
                user_id=user_id,
                domain_a=syn['domain_a'],
                domain_b=syn['domain_b'],
                correlation=syn['correlation'],
                insight=syn.get('insight', ''),
            )

        # Build fingerprint
        fingerprint = {
            'user_id': user_id,
            'domains': domain_scores,
            'domain_meta': domain_meta,
            'confidence': round(confidence, 4),
            'stage': new_stage,
            'stage_name': STAGE_NAMES.get(new_stage, 'Unknown'),
            'stage_changed': stage_result['changed'],
            'stage_reason': stage_result['reason'],
            'total_events': total_events,
            'domains_with_events': domains_with_events,
            'weeks_of_data': round(weeks_of_data, 2),
            'stability': round(stability, 4),
            'synergy_count': len(detected_synergies) + synergy_count,
            'synergies': detected_synergies,
            'timestamp': time.time(),
        }

        # Update profile in DB
        self.db.update_profile(
            user_id,
            fingerprint_json=json.dumps(fingerprint),
            confidence=confidence,
            evolution_stage=new_stage,
            evolution_count=profile.get('evolution_count', 0) + 1,
            last_evolved=time.time(),
            last_synced=time.time(),
        )

        # Save stage_change snapshot if stage changed
        if stage_result['changed']:
            self.db.save_snapshot(
                user_id=user_id,
                snapshot_type='stage_change',
                fingerprint_json=json.dumps(fingerprint),
                stage=new_stage,
                confidence=confidence,
            )

        # Check and create missed snapshots
        self._check_snapshots(user_id, fingerprint, new_stage, confidence)

        # Publish to event bus
        if self._event_bus:
            try:
                self._event_bus.publish('profile.updated', {
                    'user_id': user_id, 'confidence': confidence,
                    'domains_changed': list(fingerprint.keys()),
                })
                if stage_result['changed']:
                    self._event_bus.publish('profile.stage.changed', {
                        'user_id': user_id, 'old_stage': current_stage,
                        'new_stage': new_stage, 'reason': stage_result['reason'],
                    })
            except Exception as e:
                logger.warning("Bus publish failed: %s", e)

        # Publish intelligence
        if self._intelligence_publisher:
            try:
                self._intelligence_publisher.publish_for_user(user_id, fingerprint, new_stage, confidence)
            except Exception as e:
                logger.warning("Intelligence publish failed: %s", e)

        # Run predictive alerts
        if self._alert_engine:
            try:
                self._alert_engine.check_predictive(user_id, fingerprint)
            except Exception as e:
                logger.warning("Predictive alert check failed: %s", e)

        return fingerprint

    # ─── Daily Domain Scores ─────────────────────────────────────────────

    def _get_daily_domain_scores(self, user_id: str) -> Dict[str, List[Optional[float]]]:
        """Get daily aggregate scores per domain over the last 28 days.

        Returns {domain: [28 daily float scores or None]} where index 0 is
        the oldest day and index 27 is today.
        """
        now = time.time()
        start_time = now - 28 * 24 * 3600
        result: Dict[str, List[Optional[float]]] = {}

        for domain_name, processor in self.processors.items():
            daily: List[Optional[float]] = [None] * 28
            events = self.db.get_events(
                user_id, domain=domain_name, start_time=start_time, limit=10000,
            )
            if not events:
                result[domain_name] = daily
                continue

            # Group events by day index
            day_events: Dict[int, list] = defaultdict(list)
            for evt in events:
                ts = evt.get('timestamp', now)
                day_idx = 27 - int((now - ts) / (24 * 3600))
                if 0 <= day_idx < 28:
                    day_events[day_idx].append(evt)

            # Compute score per day
            for day_idx, devts in day_events.items():
                try:
                    day_result = processor.compute(devts)
                    daily[day_idx] = day_result.get('score', 0.0)
                except Exception:
                    daily[day_idx] = None

            result[domain_name] = daily

        return result

    # ─── Snapshot Management ─────────────────────────────────────────────

    def _check_snapshots(
        self,
        user_id: str,
        fingerprint: Dict,
        stage: int,
        confidence: float,
    ) -> None:
        """Check for missed snapshots and create them."""
        overdue = self.evolution.check_missed_snapshots(user_id)
        fp_json = json.dumps(fingerprint)
        for snapshot_type in overdue:
            self.db.save_snapshot(
                user_id=user_id,
                snapshot_type=snapshot_type,
                fingerprint_json=fp_json,
                stage=stage,
                confidence=confidence,
            )

    # ─── Event Bus Integration ───────────────────────────────────────────

    def set_event_bus(self, bus, connector=None):
        """Wire the event bus for real-time intelligence and alerts."""
        self._event_bus = bus
        self._ecosystem_connector = connector
        try:
            from intelligence_publisher import IntelligencePublisher
            self._intelligence_publisher = IntelligencePublisher(bus, connector)
        except Exception as e:
            logger.warning("IntelligencePublisher init failed: %s", e)
        try:
            from alert_engine import AlertEngine
            self._alert_engine = AlertEngine(bus, self.db)
        except Exception as e:
            logger.warning("AlertEngine init failed: %s", e)

    # ─── Query Methods ───────────────────────────────────────────────────

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored profile for a user, or None if not found."""
        try:
            profile = self.db.get_or_create_profile(user_id)
            if profile and profile.get('fingerprint_json'):
                profile['fingerprint'] = json.loads(profile['fingerprint_json'])
            return profile
        except Exception:
            logger.exception("Error getting profile for %s", user_id)
            return None

    def get_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent snapshot for a user."""
        snapshots = self.db.get_snapshots(user_id, limit=1)
        if snapshots:
            snap = snapshots[0]
            if snap.get('fingerprint_json'):
                snap['fingerprint'] = json.loads(snap['fingerprint_json'])
            return snap
        return None

    def get_evolution(self, user_id: str) -> Dict[str, Any]:
        """Return evolution history for a user."""
        profile = self.db.get_or_create_profile(user_id)
        snapshots = self.db.get_snapshots(user_id, snapshot_type='stage_change')
        synergies = self.db.get_synergies(user_id)
        return {
            'user_id': user_id,
            'current_stage': profile.get('evolution_stage', 1),
            'stage_name': STAGE_NAMES.get(profile.get('evolution_stage', 1), 'Unknown'),
            'confidence': profile.get('confidence', 0.0),
            'evolution_count': profile.get('evolution_count', 0),
            'stage_history': snapshots,
            'synergies': synergies,
        }

    def get_stage_progress(self, user_id: str) -> Dict[str, Any]:
        """Return progress toward the next evolution stage."""
        profile = self.db.get_or_create_profile(user_id)
        current_stage = profile.get('evolution_stage', 1)
        confidence = profile.get('confidence', 0.0)
        created_at = profile.get('created_at', time.time())

        total_events = self.db.get_event_count(user_id)
        domain_counts = self.db.get_domain_event_counts(user_id)
        domains_with_events = sum(1 for c in domain_counts.values() if c > 0)
        weeks_of_data = (time.time() - created_at) / _SECONDS_PER_WEEK
        synergies = self.db.get_synergies(user_id)

        # Approximate positive_patterns and stability from stored fingerprint
        positive_patterns = False
        stability = 0.5
        fp_json = profile.get('fingerprint_json')
        if fp_json:
            try:
                fp = json.loads(fp_json)
                stability = fp.get('stability', 0.5)
                for ds in fp.get('domains', {}).values():
                    if ds.get('metrics', {}).get('recovery_rate', 0) > 0.1:
                        positive_patterns = True
                        break
            except (json.JSONDecodeError, TypeError):
                pass

        return self.evolution.get_stage_progress(
            current_stage=current_stage,
            confidence=confidence,
            total_events=total_events,
            domains_with_events=domains_with_events,
            weeks_of_data=weeks_of_data,
            positive_patterns=positive_patterns,
            synergy_count=len(synergies),
            stability_score=stability,
        )

    # ─── User Deletion ───────────────────────────────────────────────────

    def delete_user(self, user_id: str) -> None:
        """Purge all data for a user and clear caches."""
        self.db.delete_user(user_id)
        # Clear rate limit cache entries for this user
        with self._rate_lock:
            keys_to_remove = [k for k in self._rate_cache if k[0] == user_id]
            for k in keys_to_remove:
                del self._rate_cache[k]
        # Clear consecutive met counter
        self._consecutive_met.pop(user_id, None)
        with self._snapshot_lock:
            self._snapshots.pop(user_id, None)

    # ─── Lifecycle ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down the engine and release resources."""
        self._closed = True
        self.db.close()
