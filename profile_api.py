"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile API Blueprint
===============================================================================
Flask Blueprint exposing the User Profile Engine over REST.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from biometrics, emotions, therapy sessions
===============================================================================
"""

import json
import logging
import os
import time
import uuid
from functools import wraps
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _load_service_keys() -> Dict[str, str]:
    """Load service API keys from environment variables.

    Returns a dict mapping key-value -> service-name so we can look up
    which service authenticated.
    """
    keys: Dict[str, str] = {}
    mapping = {
        'PROFILE_SERVICE_KEY_BACKEND': 'backend',
        'PROFILE_SERVICE_KEY_FRONTEND': 'frontend',
        'PROFILE_SERVICE_KEY_MASTER': 'master',
    }
    for env_var, service_name in mapping.items():
        val = os.environ.get(env_var)
        if val:
            keys[val] = service_name
    return keys


def require_service_key(f):
    """Decorator that checks X-Service-Key header against known service keys."""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-Service-Key', '')
        service_keys = _load_service_keys()
        if not key or key not in service_keys:
            return jsonify({'error': 'Unauthorized', 'message': 'Valid X-Service-Key header required'}), 401
        request.service_name = service_keys[key]
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _generate_insights(fingerprint: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate human-readable insights from the fingerprint and profile."""
    insights: List[Dict[str, Any]] = []
    domains = fingerprint.get('domains', {})

    # Emotional domain insights
    emotional = domains.get('emotion', {})
    if emotional:
        metrics = emotional.get('metrics', {})
        dominant = metrics.get('dominant_emotion', 'neutral')
        volatility = metrics.get('volatility', 0.0)
        insights.append({
            'domain': 'emotional',
            'type': 'dominant_emotion',
            'value': dominant,
            'description': f'Your dominant emotional state is {dominant}.',
        })
        insights.append({
            'domain': 'emotional',
            'type': 'volatility',
            'value': round(volatility, 4),
            'description': f'Emotional volatility is {"high" if volatility > 0.5 else "moderate" if volatility > 0.25 else "low"} ({round(volatility, 2)}).',
        })

    # Biometric domain insights
    biometric = domains.get('biometric', {})
    if biometric:
        metrics = biometric.get('metrics', {})
        resting_hr = metrics.get('resting_hr', 0)
        stress = metrics.get('stress_index', 0.0)
        if resting_hr:
            insights.append({
                'domain': 'biometric',
                'type': 'resting_hr',
                'value': resting_hr,
                'description': f'Resting heart rate is {resting_hr} bpm.',
            })
        insights.append({
            'domain': 'biometric',
            'type': 'stress_level',
            'value': round(stress, 4),
            'description': f'Current stress index is {round(stress, 2)}.',
        })

    # Temporal / behavioral insights
    behavioral = domains.get('behavioral', {})
    if behavioral:
        metrics = behavioral.get('metrics', {})
        chronotype = metrics.get('chronotype', 'unknown')
        if chronotype != 'unknown':
            insights.append({
                'domain': 'temporal',
                'type': 'chronotype',
                'value': chronotype,
                'description': f'Your chronotype is {chronotype}.',
            })

    return insights


def _compute_big_five(fingerprint: Dict[str, Any]) -> Dict[str, float]:
    """Map genetic fingerprint DNA to Big Five personality traits."""
    domains = fingerprint.get('domains', {})

    emotional = domains.get('emotion', {}).get('metrics', {})
    behavioral = domains.get('behavioral', {}).get('metrics', {})

    emotional_range = emotional.get('emotional_range', 5.0)
    completion_rate = behavioral.get('completion_rate', 0.5)
    daily_interaction_rate = behavioral.get('daily_interaction_rate', 5.0)
    intervention_response_rate = behavioral.get('intervention_response_rate', 0.5)
    volatility = emotional.get('volatility', 0.3)

    return {
        'openness': round(min(emotional_range / 15.0, 1.0), 4),
        'conscientiousness': round(min(max(completion_rate, 0.0), 1.0), 4),
        'extraversion': round(min(daily_interaction_rate / 10.0, 1.0), 4),
        'agreeableness': round(min(max(intervention_response_rate, 0.0), 1.0), 4),
        'neuroticism': round(min(max(volatility, 0.0), 1.0), 4),
    }


def _compute_predictions(fingerprint: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compute forward-looking predictions from the fingerprint."""
    domains = fingerprint.get('domains', {})

    # Predicted dominant emotion family
    emotional = domains.get('emotion', {}).get('metrics', {})
    dominant = emotional.get('dominant_emotion', 'neutral')

    # Expected stress level
    biometric = domains.get('biometric', {}).get('metrics', {})
    stress = biometric.get('stress_index', 0.3)

    # Optimal intervention hours based on behavioral patterns
    behavioral = domains.get('behavioral', {}).get('metrics', {})
    chronotype = behavioral.get('chronotype', 'unknown')
    if chronotype == 'morning':
        optimal_hours = [9, 10, 11]
    elif chronotype == 'evening':
        optimal_hours = [16, 17, 18]
    else:
        optimal_hours = [10, 14, 16]

    # Risk factors
    risk_factors: List[str] = []
    volatility = emotional.get('volatility', 0.0)
    if volatility > 0.6:
        risk_factors.append('high_emotional_volatility')
    if stress > 0.7:
        risk_factors.append('elevated_stress')
    stability = fingerprint.get('stability', 1.0)
    if stability < 0.3:
        risk_factors.append('low_stability')

    return {
        'predicted_dominant_family': dominant,
        'expected_stress_level': round(stress, 4),
        'optimal_intervention_hours': optimal_hours,
        'risk_factors': risk_factors,
        'window_hours': 48,
    }


# ---------------------------------------------------------------------------
# Blueprint factory
# ---------------------------------------------------------------------------

def create_profile_blueprint(engine) -> Blueprint:
    """Create and return a Flask Blueprint wired to the given UserProfileEngine."""

    bp = Blueprint('profile', __name__)

    # === Health ============================================================

    @bp.route('/api/profile/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'service': 'profile-engine',
            'timestamp': time.time(),
        })

    # === Ingest (service-key protected) ====================================

    @bp.route('/api/profile/ingest', methods=['POST'])
    @require_service_key
    def ingest():
        data = request.get_json(silent=True) or {}

        user_id = data.get('user_id')
        domain = data.get('domain')
        event_type = data.get('event_type')
        payload = data.get('payload')
        source = data.get('source', 'api')
        confidence = data.get('confidence')

        if not user_id or not domain or not event_type:
            return jsonify({'error': 'Missing required fields: user_id, domain, event_type'}), 400

        event_id = engine.log_event(
            user_id=user_id,
            domain=domain,
            event_type=event_type,
            payload=payload,
            source=source,
            confidence=confidence,
        )

        if event_id is None:
            return jsonify({'event_id': None, 'status': 'skipped'}), 200

        return jsonify({'event_id': event_id, 'status': 'logged'}), 201

    # === Profile endpoints (user-facing) ===================================

    @bp.route('/api/profile/<user_id>/snapshot', methods=['GET'])
    def snapshot(user_id: str):
        try:
            fingerprint = engine.process(user_id)
            snap = engine.get_snapshot(user_id)
            return jsonify({
                'user_id': user_id,
                'fingerprint': fingerprint,
                'snapshot': snap,
            })
        except Exception as e:
            logger.exception("Error computing snapshot for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>/evolution', methods=['GET'])
    def evolution(user_id: str):
        try:
            result = engine.get_evolution(user_id)
            return jsonify(result)
        except Exception as e:
            logger.exception("Error getting evolution for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>/evolution/stage', methods=['GET'])
    def stage_progress(user_id: str):
        try:
            result = engine.get_stage_progress(user_id)
            return jsonify(result)
        except Exception as e:
            logger.exception("Error getting stage progress for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>/domain/<domain>', methods=['GET'])
    def domain_detail(user_id: str, domain: str):
        try:
            fingerprint = engine.process(user_id)
            domain_data = fingerprint.get('domains', {}).get(domain, {})
            recent_events = engine.db.get_events(user_id, domain=domain, limit=50)
            return jsonify({
                'user_id': user_id,
                'domain': domain,
                'fingerprint': domain_data,
                'recent_events': recent_events,
            })
        except Exception as e:
            logger.exception("Error getting domain %s for %s", domain, user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>/synergies', methods=['GET'])
    def synergies(user_id: str):
        try:
            result = engine.db.get_synergies(user_id)
            return jsonify({'user_id': user_id, 'synergies': result})
        except Exception as e:
            logger.exception("Error getting synergies for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>/predictions', methods=['GET'])
    def predictions(user_id: str):
        try:
            fingerprint = engine.process(user_id)
            profile = engine.get_profile(user_id) or {}
            preds = _compute_predictions(fingerprint, profile)
            return jsonify({'user_id': user_id, 'predictions': preds})
        except Exception as e:
            logger.exception("Error computing predictions for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>/events', methods=['GET'])
    def events(user_id: str):
        try:
            domain = request.args.get('domain')
            start_time = request.args.get('start_time', type=float)
            end_time = request.args.get('end_time', type=float)
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)

            result = engine.db.get_events(
                user_id,
                domain=domain,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                offset=offset,
            )
            return jsonify({'user_id': user_id, 'events': result, 'count': len(result)})
        except Exception as e:
            logger.exception("Error getting events for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>/export', methods=['GET'])
    def export_profile(user_id: str):
        try:
            profile = engine.get_profile(user_id) or {}
            snapshots = engine.db.get_snapshots(user_id)
            synergies = engine.db.get_synergies(user_id)
            event_count = engine.db.get_event_count(user_id)
            return jsonify({
                'user_id': user_id,
                'profile': profile,
                'snapshots': snapshots,
                'synergies': synergies,
                'event_count': event_count,
            })
        except Exception as e:
            logger.exception("Error exporting profile for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>', methods=['DELETE'])
    def delete_user(user_id: str):
        try:
            engine.delete_user(user_id)
            return jsonify({'status': 'deleted', 'user_id': user_id})
        except Exception as e:
            logger.exception("Error deleting user %s", user_id)
            return jsonify({'error': str(e)}), 500

    # === Ecosystem endpoints (service-key protected) =======================

    @bp.route('/api/profile/<user_id>/context', methods=['GET'])
    @require_service_key
    def context(user_id: str):
        try:
            profile = engine.get_profile(user_id) or {}
            domain_counts = engine.db.get_domain_event_counts(user_id)

            # Build lightweight per-domain summary
            fp = {}
            fp_json = profile.get('fingerprint_json')
            if fp_json:
                try:
                    fp = json.loads(fp_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            domain_summary: Dict[str, Any] = {}
            for domain_name, count in domain_counts.items():
                domain_data = fp.get('domains', {}).get(domain_name, {})
                domain_summary[domain_name] = {
                    'score': domain_data.get('score', 0.0),
                    'event_count': count,
                }

            return jsonify({
                'user_id': user_id,
                'domains': domain_summary,
                'stage': profile.get('evolution_stage', 1),
                'confidence': profile.get('confidence', 0.0),
            })
        except Exception as e:
            logger.exception("Error getting context for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/profile/<user_id>/domain/<domain>/score', methods=['GET'])
    @require_service_key
    def domain_score(user_id: str, domain: str):
        try:
            profile = engine.get_profile(user_id) or {}
            fp = {}
            fp_json = profile.get('fingerprint_json')
            if fp_json:
                try:
                    fp = json.loads(fp_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            domain_data = fp.get('domains', {}).get(domain, {})
            return jsonify({
                'user_id': user_id,
                'domain': domain,
                'score': domain_data.get('score', 0.0),
            })
        except Exception as e:
            logger.exception("Error getting domain score for %s/%s", user_id, domain)
            return jsonify({'error': str(e)}), 500

    # === Frontend-compatible endpoints =====================================

    @bp.route('/api/v1/users/<user_id>/genetic-fingerprint/sync', methods=['POST'])
    def fingerprint_sync(user_id: str):
        try:
            fingerprint = engine.process(user_id)
            synergies = engine.db.get_synergies(user_id)
            return jsonify({
                'user_id': user_id,
                'fingerprint': fingerprint,
                'confidence': fingerprint.get('confidence', 0.0),
                'stage': fingerprint.get('stage', 1),
                'stage_name': fingerprint.get('stage_name', 'Unknown'),
                'synergies': synergies,
            })
        except Exception as e:
            logger.exception("Error syncing fingerprint for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/v1/users/<user_id>/insights', methods=['POST'])
    def user_insights(user_id: str):
        try:
            fingerprint = engine.process(user_id)
            profile = engine.get_profile(user_id) or {}
            insights = _generate_insights(fingerprint, profile)
            return jsonify({
                'user_id': user_id,
                'insights': insights,
            })
        except Exception as e:
            logger.exception("Error generating insights for %s", user_id)
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/v1/cognitive/theory-of-mind', methods=['POST'])
    def theory_of_mind():
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        try:
            fingerprint = engine.process(user_id)
            big_five = _compute_big_five(fingerprint)
            return jsonify({
                'user_id': user_id,
                'big_five': big_five,
            })
        except Exception as e:
            logger.exception("Error computing theory-of-mind for %s", user_id)
            return jsonify({'error': str(e)}), 500

    return bp
