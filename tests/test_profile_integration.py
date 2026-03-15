"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - End-to-End Integration Tests
===============================================================================
Full lifecycle, API, multi-user isolation, and prediction tests for the
User Profile / Genetic Fingerprint system.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from biometrics, emotions, therapy sessions
===============================================================================
"""

import json
import os
import sys
import tempfile
import time
import unittest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask

from profile_api import create_profile_blueprint
from user_profile_engine import UserProfileEngine


SERVICE_KEY = 'integration-test-key-e2e'


class TestProfileIntegration(unittest.TestCase):
    """End-to-end integration tests for the full profile engine pipeline."""

    def setUp(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmpfile.close()
        self.engine = UserProfileEngine(db_path=self._tmpfile.name)

    def tearDown(self):
        self.engine.close()
        try:
            os.unlink(self._tmpfile.name)
        except OSError:
            pass

    # ─── Full Lifecycle ──────────────────────────────────────────────────

    def test_full_lifecycle(self):
        """Log events across 4 domains, process, and verify fingerprint."""
        user_id = 'lifecycle-user'

        # 1. Log 20 emotional events (cycling joy/sadness/calm/anger)
        emotions = ['joy', 'sadness', 'calm', 'anger']
        for i in range(20):
            eid = self.engine.log_event(
                user_id, 'emotional', 'emotion_detected',
                payload={'emotion': emotions[i % 4], 'confidence': 0.8},
            )
            self.assertIsNotNone(eid)

        # 2. Log 15 biometric events (hr 70-84, hrv 55-69, eda 3.0)
        for i in range(15):
            eid = self.engine.log_event(
                user_id, 'biometric', 'biometric_reading',
                payload={'hr': 70 + i, 'hrv': 55 + i, 'eda': 3.0},
            )
            self.assertIsNotNone(eid)

        # 3. Log 10 behavioral session_completed events
        for i in range(10):
            eid = self.engine.log_event(
                user_id, 'behavioral', 'session_completed',
                payload={'duration': 30, 'completed': True},
            )
            self.assertIsNotNone(eid)

        # 4. Log 10 linguistic text_analyzed events
        for i in range(10):
            eid = self.engine.log_event(
                user_id, 'linguistic', 'text_analyzed',
                payload={'sentiment': 0.7, 'word_count': 100},
            )
            self.assertIsNotNone(eid)

        # 5. Process
        fingerprint = self.engine.process(user_id)

        # 6. Assert all 4 domains with data are present in fingerprint
        domains = fingerprint['domains']
        for domain in ('emotional', 'biometric', 'behavioral', 'linguistic'):
            self.assertIn(domain, domains, f"Domain {domain} missing from fingerprint")
            self.assertGreater(
                domains[domain].get('event_count', 0), 0,
                f"Domain {domain} should have events",
            )

        # 7. Assert emotional.dominant_emotion is in the expected set
        emo_metrics = domains['emotional']['metrics']
        self.assertIn(
            emo_metrics['dominant_emotion'],
            {'joy', 'sadness', 'calm', 'anger', 'neutral'},
        )

        # 8. Assert emotional.event_count == 20
        self.assertEqual(domains['emotional']['event_count'], 20)

        # 9. Assert evolution_stage == 1 (Nascent)
        profile = self.engine.get_profile(user_id)
        self.assertEqual(profile['evolution_stage'], 1)

        # 10. Assert confidence > 0.0
        self.assertGreater(profile['confidence'], 0.0)

        # 11. Assert snapshot works (stage, fingerprint present)
        snapshot = self.engine.get_snapshot(user_id)
        self.assertIsNotNone(snapshot)
        self.assertIn('stage', snapshot)
        self.assertIn('fingerprint', snapshot)

    # ─── API Integration ─────────────────────────────────────────────────

    def test_api_integration(self):
        """Test the full API round-trip: ingest, snapshot, sync, delete."""
        user_id = 'api-integration-user'

        # Set service key env var
        os.environ['PROFILE_SERVICE_KEY_BACKEND'] = SERVICE_KEY

        try:
            # Create Flask app with profile blueprint
            app = Flask(__name__)
            app.config['TESTING'] = True
            bp = create_profile_blueprint(self.engine)
            app.register_blueprint(bp)
            client = app.test_client()

            # POST to /api/profile/ingest with service key
            resp = client.post(
                '/api/profile/ingest',
                json={
                    'user_id': user_id,
                    'domain': 'emotional',
                    'event_type': 'emotion_detected',
                    'payload': {'emotion': 'joy', 'confidence': 0.9},
                    'source': 'api',
                },
                headers={'X-Service-Key': SERVICE_KEY},
            )
            self.assertEqual(resp.status_code, 201)
            data = resp.get_json()
            self.assertEqual(data['status'], 'logged')
            self.assertIsNotNone(data['event_id'])

            # GET /api/profile/<user_id>/snapshot — verify 200, has stage
            resp = client.get(f'/api/profile/{user_id}/snapshot')
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn('fingerprint', data)
            self.assertIn('stage', data['fingerprint'])

            # POST /api/v1/users/<user_id>/genetic-fingerprint/sync — verify 200
            resp = client.post(f'/api/v1/users/{user_id}/genetic-fingerprint/sync')
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn('fingerprint', data)
            self.assertIn('stage', data)
            self.assertIn('stage_name', data)

            # DELETE /api/profile/<user_id> — verify 200
            resp = client.delete(f'/api/profile/{user_id}')
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data['status'], 'deleted')

            # Verify user data is gone
            events = self.engine.db.get_events(user_id)
            self.assertEqual(len(events), 0)

        finally:
            os.environ.pop('PROFILE_SERVICE_KEY_BACKEND', None)

    # ─── Multi-User Isolation ────────────────────────────────────────────

    def test_multi_user_isolation(self):
        """Verify events for user1 do not appear in user2's profile."""
        user1 = 'isolation-user-1'
        user2 = 'isolation-user-2'

        # Log distinct events for each user
        for i in range(10):
            self.engine.log_event(
                user1, 'emotional', 'emotion_detected',
                payload={'emotion': 'joy', 'confidence': 0.9},
            )
        for i in range(5):
            self.engine.log_event(
                user2, 'emotional', 'emotion_detected',
                payload={'emotion': 'sadness', 'confidence': 0.7},
            )

        # Process both
        fp1 = self.engine.process(user1)
        fp2 = self.engine.process(user2)

        # Verify event counts are isolated
        self.assertEqual(fp1['domains']['emotional']['event_count'], 10)
        self.assertEqual(fp2['domains']['emotional']['event_count'], 5)

        # Verify events don't cross-contaminate
        user1_events = self.engine.db.get_events(user1, domain='emotional')
        user2_events = self.engine.db.get_events(user2, domain='emotional')
        self.assertEqual(len(user1_events), 10)
        self.assertEqual(len(user2_events), 5)

        # Verify dominant emotions reflect what was logged
        u1_dominant = fp1['domains']['emotional']['metrics']['dominant_emotion']
        u2_dominant = fp2['domains']['emotional']['metrics']['dominant_emotion']
        self.assertEqual(u1_dominant, 'joy')
        self.assertEqual(u2_dominant, 'sadness')

        # Deleting user1 should not affect user2
        self.engine.delete_user(user1)
        user2_events_after = self.engine.db.get_events(user2, domain='emotional')
        self.assertEqual(len(user2_events_after), 5)

    # ─── Predictions Endpoint ────────────────────────────────────────────

    def test_predictions_endpoint(self):
        """Log emotional + biometric events and verify predictions response."""
        user_id = 'predictions-user'

        # Set service key env var
        os.environ['PROFILE_SERVICE_KEY_BACKEND'] = SERVICE_KEY

        try:
            # Log some emotional events
            for i in range(10):
                self.engine.log_event(
                    user_id, 'emotional', 'emotion_detected',
                    payload={'emotion': 'calm', 'confidence': 0.85},
                )

            # Log some biometric events
            for i in range(8):
                self.engine.log_event(
                    user_id, 'biometric', 'biometric_reading',
                    payload={'hr': 72 + i, 'hrv': 60 + i, 'eda': 2.5},
                )

            # Create Flask app
            app = Flask(__name__)
            app.config['TESTING'] = True
            bp = create_profile_blueprint(self.engine)
            app.register_blueprint(bp)
            client = app.test_client()

            # GET /api/profile/<user_id>/predictions
            resp = client.get(f'/api/profile/{user_id}/predictions')
            self.assertEqual(resp.status_code, 200)

            data = resp.get_json()
            self.assertEqual(data['user_id'], user_id)

            preds = data['predictions']
            self.assertIn('predicted_dominant_family', preds)
            self.assertIn('risk_factors', preds)
            self.assertIsInstance(preds['risk_factors'], list)
            self.assertIn('expected_stress_level', preds)
            self.assertIn('optimal_intervention_hours', preds)
            self.assertEqual(preds['window_hours'], 48)

        finally:
            os.environ.pop('PROFILE_SERVICE_KEY_BACKEND', None)


if __name__ == '__main__':
    unittest.main()
