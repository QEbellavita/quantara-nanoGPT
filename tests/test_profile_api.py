"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile API Blueprint Tests
===============================================================================
Tests for profile_api.py Flask Blueprint.
===============================================================================
"""

import json
import os
import sys
import tempfile
import time
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask

from profile_api import create_profile_blueprint
from user_profile_engine import UserProfileEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app(db_path: str):
    """Create a Flask app with the profile blueprint wired to a fresh engine."""
    engine = UserProfileEngine(db_path=db_path)
    app = Flask(__name__)
    app.config['TESTING'] = True
    bp = create_profile_blueprint(engine)
    app.register_blueprint(bp)
    return app, engine


SERVICE_KEY = 'test-backend-key-12345'
FRONTEND_KEY = 'test-frontend-key-67890'
MASTER_KEY = 'test-master-key-99999'


class _BaseAPITest(unittest.TestCase):
    """Shared setup/teardown for all API test classes."""

    def setUp(self):
        # Set service key env vars
        os.environ['PROFILE_SERVICE_KEY_BACKEND'] = SERVICE_KEY
        os.environ['PROFILE_SERVICE_KEY_FRONTEND'] = FRONTEND_KEY
        os.environ['PROFILE_SERVICE_KEY_MASTER'] = MASTER_KEY

        self._tmpfile = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmpfile.close()
        self.app, self.engine = _make_app(self._tmpfile.name)
        self.client = self.app.test_client()

    def tearDown(self):
        self.engine.close()
        try:
            os.unlink(self._tmpfile.name)
        except OSError:
            pass
        # Clean env
        for var in ('PROFILE_SERVICE_KEY_BACKEND', 'PROFILE_SERVICE_KEY_FRONTEND',
                     'PROFILE_SERVICE_KEY_MASTER'):
            os.environ.pop(var, None)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestProfileHealthEndpoint(_BaseAPITest):
    """GET /api/profile/health"""

    def test_health_returns_200(self):
        resp = self.client.get('/api/profile/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['service'], 'profile-engine')
        self.assertIn('timestamp', data)

    def test_health_no_auth_required(self):
        # Should work without any service key
        os.environ.pop('PROFILE_SERVICE_KEY_BACKEND', None)
        resp = self.client.get('/api/profile/health')
        self.assertEqual(resp.status_code, 200)


class TestIngestEndpoint(_BaseAPITest):
    """POST /api/profile/ingest"""

    def test_ingest_requires_service_key(self):
        resp = self.client.post(
            '/api/profile/ingest',
            json={'user_id': 'u1', 'domain': 'emotion', 'event_type': 'mood'},
        )
        self.assertEqual(resp.status_code, 401)

    def test_ingest_rejects_bad_key(self):
        resp = self.client.post(
            '/api/profile/ingest',
            json={'user_id': 'u1', 'domain': 'emotion', 'event_type': 'mood'},
            headers={'X-Service-Key': 'wrong-key'},
        )
        self.assertEqual(resp.status_code, 401)

    def test_ingest_with_valid_key(self):
        resp = self.client.post(
            '/api/profile/ingest',
            json={
                'user_id': 'u1',
                'domain': 'emotion',
                'event_type': 'mood',
                'payload': {'mood': 'happy'},
                'source': 'api',
                'confidence': 0.95,
            },
            headers={'X-Service-Key': SERVICE_KEY},
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertEqual(data['status'], 'logged')
        self.assertIsNotNone(data['event_id'])

    def test_ingest_missing_fields(self):
        resp = self.client.post(
            '/api/profile/ingest',
            json={'user_id': 'u1'},
            headers={'X-Service-Key': SERVICE_KEY},
        )
        self.assertEqual(resp.status_code, 400)

    def test_ingest_frontend_key_works(self):
        resp = self.client.post(
            '/api/profile/ingest',
            json={'user_id': 'u1', 'domain': 'emotion', 'event_type': 'mood'},
            headers={'X-Service-Key': FRONTEND_KEY},
        )
        self.assertEqual(resp.status_code, 201)

    def test_ingest_master_key_works(self):
        resp = self.client.post(
            '/api/profile/ingest',
            json={'user_id': 'u1', 'domain': 'emotion', 'event_type': 'mood'},
            headers={'X-Service-Key': MASTER_KEY},
        )
        self.assertEqual(resp.status_code, 201)


class TestSnapshotEndpoint(_BaseAPITest):
    """GET /api/profile/<user_id>/snapshot"""

    def test_snapshot_new_user(self):
        resp = self.client.get('/api/profile/test-user-1/snapshot')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['user_id'], 'test-user-1')
        self.assertIn('fingerprint', data)

    def test_snapshot_after_events(self):
        # Ingest some events first
        for i in range(3):
            self.client.post(
                '/api/profile/ingest',
                json={
                    'user_id': 'test-user-2',
                    'domain': 'emotion',
                    'event_type': 'mood',
                    'payload': {'mood': 'happy', 'intensity': 0.8},
                },
                headers={'X-Service-Key': SERVICE_KEY},
            )

        resp = self.client.get('/api/profile/test-user-2/snapshot')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('fingerprint', data)
        fp = data['fingerprint']
        self.assertIn('domains', fp)
        self.assertIn('confidence', fp)


class TestDeleteEndpoint(_BaseAPITest):
    """DELETE /api/profile/<user_id>"""

    def test_delete_user(self):
        # Create a user first
        self.client.post(
            '/api/profile/ingest',
            json={'user_id': 'del-user', 'domain': 'emotion', 'event_type': 'mood'},
            headers={'X-Service-Key': SERVICE_KEY},
        )

        # Delete
        resp = self.client.delete('/api/profile/del-user')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'deleted')
        self.assertEqual(data['user_id'], 'del-user')

        # Events should be gone
        events = self.engine.db.get_events('del-user')
        self.assertEqual(len(events), 0)

    def test_delete_nonexistent_user(self):
        resp = self.client.delete('/api/profile/ghost-user')
        self.assertEqual(resp.status_code, 200)


class TestFingerprintSyncEndpoint(_BaseAPITest):
    """POST /api/v1/users/<user_id>/genetic-fingerprint/sync"""

    def test_sync_returns_fingerprint(self):
        resp = self.client.post('/api/v1/users/sync-user/genetic-fingerprint/sync')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['user_id'], 'sync-user')
        self.assertIn('fingerprint', data)
        self.assertIn('confidence', data)
        self.assertIn('stage', data)
        self.assertIn('stage_name', data)
        self.assertIn('synergies', data)

    def test_sync_after_ingestion(self):
        self.client.post(
            '/api/profile/ingest',
            json={
                'user_id': 'sync-user-2',
                'domain': 'emotion',
                'event_type': 'mood',
                'payload': {'mood': 'calm'},
            },
            headers={'X-Service-Key': SERVICE_KEY},
        )
        resp = self.client.post('/api/v1/users/sync-user-2/genetic-fingerprint/sync')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data['confidence'], float)


class TestPredictionsEndpoint(_BaseAPITest):
    """GET /api/profile/<user_id>/predictions"""

    def test_predictions_structure(self):
        resp = self.client.get('/api/profile/pred-user/predictions')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['user_id'], 'pred-user')
        preds = data['predictions']
        self.assertIn('predicted_dominant_family', preds)
        self.assertIn('expected_stress_level', preds)
        self.assertIn('optimal_intervention_hours', preds)
        self.assertIn('risk_factors', preds)
        self.assertEqual(preds['window_hours'], 48)

    def test_predictions_risk_factors_is_list(self):
        resp = self.client.get('/api/profile/pred-user-2/predictions')
        data = resp.get_json()
        self.assertIsInstance(data['predictions']['risk_factors'], list)


class TestEvolutionEndpoint(_BaseAPITest):
    """GET /api/profile/<user_id>/evolution"""

    def test_evolution_returns_structure(self):
        resp = self.client.get('/api/profile/evo-user/evolution')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['user_id'], 'evo-user')
        self.assertIn('current_stage', data)
        self.assertIn('stage_name', data)


class TestEventsEndpoint(_BaseAPITest):
    """GET /api/profile/<user_id>/events"""

    def test_events_empty(self):
        resp = self.client.get('/api/profile/no-events-user/events')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['events'], [])

    def test_events_with_data(self):
        self.client.post(
            '/api/profile/ingest',
            json={'user_id': 'ev-user', 'domain': 'emotion', 'event_type': 'mood'},
            headers={'X-Service-Key': SERVICE_KEY},
        )
        resp = self.client.get('/api/profile/ev-user/events')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertGreater(len(data['events']), 0)


class TestTheoryOfMindEndpoint(_BaseAPITest):
    """POST /api/v1/cognitive/theory-of-mind"""

    def test_big_five_structure(self):
        resp = self.client.post(
            '/api/v1/cognitive/theory-of-mind',
            json={'user_id': 'tom-user'},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        big_five = data['big_five']
        for trait in ('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'):
            self.assertIn(trait, big_five)
            self.assertIsInstance(big_five[trait], float)
            self.assertGreaterEqual(big_five[trait], 0.0)
            self.assertLessEqual(big_five[trait], 1.0)

    def test_missing_user_id(self):
        resp = self.client.post(
            '/api/v1/cognitive/theory-of-mind',
            json={},
        )
        self.assertEqual(resp.status_code, 400)


class TestContextEndpoint(_BaseAPITest):
    """GET /api/profile/<user_id>/context (requires service key)"""

    def test_context_requires_key(self):
        resp = self.client.get('/api/profile/ctx-user/context')
        self.assertEqual(resp.status_code, 401)

    def test_context_with_key(self):
        resp = self.client.get(
            '/api/profile/ctx-user/context',
            headers={'X-Service-Key': SERVICE_KEY},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('stage', data)
        self.assertIn('confidence', data)


class TestDomainScoreEndpoint(_BaseAPITest):
    """GET /api/profile/<user_id>/domain/<domain>/score (requires service key)"""

    def test_domain_score_requires_key(self):
        resp = self.client.get('/api/profile/ds-user/domain/emotion/score')
        self.assertEqual(resp.status_code, 401)

    def test_domain_score_with_key(self):
        resp = self.client.get(
            '/api/profile/ds-user/domain/emotion/score',
            headers={'X-Service-Key': SERVICE_KEY},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['domain'], 'emotion')
        self.assertIn('score', data)


if __name__ == '__main__':
    unittest.main()
