"""
Tests for UserProfileEngine orchestrator.
"""

import os
import sys
import tempfile
import time
import unittest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from user_profile_engine import UserProfileEngine


class TestUserProfileEngine(unittest.TestCase):
    """Integration tests for UserProfileEngine."""

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

    # ─── Basic log_event and get_profile ─────────────────────────────────

    def test_log_event_and_get_profile(self):
        eid = self.engine.log_event(
            user_id='u1',
            domain='emotional',
            event_type='emotion_detected',
            payload={'emotion': 'joy', 'confidence': 0.9},
        )
        self.assertIsNotNone(eid)
        self.assertIsInstance(eid, int)

        profile = self.engine.get_profile('u1')
        self.assertIsNotNone(profile)
        self.assertEqual(profile['user_id'], 'u1')

    # ─── process updates fingerprint ─────────────────────────────────────

    def test_process_updates_fingerprint(self):
        # Log a few events so processing has data
        for i in range(5):
            self.engine.log_event(
                'u2', 'emotional', 'emotion_detected',
                payload={'emotion': 'joy', 'confidence': 0.8},
            )

        fingerprint = self.engine.process('u2')
        self.assertIn('domains', fingerprint)
        self.assertIn('confidence', fingerprint)
        self.assertIn('stage', fingerprint)
        self.assertEqual(fingerprint['stage'], 1)  # Nascent with few events
        self.assertGreater(fingerprint['total_events'], 0)

        # Verify the profile was updated in DB
        profile = self.engine.get_profile('u2')
        self.assertIsNotNone(profile.get('fingerprint'))

    # ─── log_event no-op when closed ─────────────────────────────────────

    def test_log_event_noop_when_closed(self):
        self.engine.close()
        result = self.engine.log_event(
            'u3', 'emotional', 'emotion_detected',
            payload={'emotion': 'joy'},
        )
        self.assertIsNone(result)

    # ─── get_snapshot returns current ────────────────────────────────────

    def test_get_snapshot_returns_current(self):
        self.engine.log_event(
            'u4', 'emotional', 'emotion_detected',
            payload={'emotion': 'sadness'},
        )
        # Process to generate data, but stage won't change with 1 event
        self.engine.process('u4')

        # A snapshot should exist (weekly catch-up since no prior snapshots)
        snapshot = self.engine.get_snapshot('u4')
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot['user_id'], 'u4')

    # ─── evolution stage starts at nascent ───────────────────────────────

    def test_evolution_stage_starts_nascent(self):
        fingerprint = self.engine.process('u5')
        self.assertEqual(fingerprint['stage'], 1)
        self.assertEqual(fingerprint['stage_name'], 'Nascent')

    # ─── biometric rate limiting ─────────────────────────────────────────

    def test_biometric_rate_limiting(self):
        # First HR reading should succeed
        eid1 = self.engine.log_event(
            'u6', 'biometric', 'hr_reading',
            payload={'bpm': 72},
        )
        self.assertIsNotNone(eid1)

        # Second HR reading within 60s should be skipped
        eid2 = self.engine.log_event(
            'u6', 'biometric', 'hr_reading',
            payload={'bpm': 74},
        )
        self.assertIsNone(eid2)

        # Non-rate-limited event on same domain should still work
        eid3 = self.engine.log_event(
            'u6', 'biometric', 'custom_event',
            payload={'value': 1},
        )
        self.assertIsNotNone(eid3)

    # ─── catch-up snapshots ──────────────────────────────────────────────

    def test_catchup_snapshots(self):
        # Create profile with created_at set to 15 days ago
        user_id = 'u7'
        self.engine.db.get_or_create_profile(user_id)

        # Backdate the profile creation
        fifteen_days_ago = time.time() - 15 * 24 * 3600
        conn = self.engine.db._read_conn()
        conn.close()
        # Use writer to update created_at
        self.engine.db._enqueue_write(
            "UPDATE profiles SET created_at = ? WHERE user_id = ?",
            (fifteen_days_ago, user_id),
            wait=True,
        )

        # Log some events
        for i in range(3):
            self.engine.log_event(
                user_id, 'emotional', 'emotion_detected',
                payload={'emotion': 'joy'},
            )

        # Process — should trigger catch-up weekly snapshot
        self.engine.process(user_id)

        # Verify a weekly snapshot was created
        snapshots = self.engine.db.get_snapshots(user_id, snapshot_type='weekly')
        self.assertTrue(len(snapshots) > 0, "Expected at least one weekly snapshot")

    # ─── delete_user clears everything ───────────────────────────────────

    def test_delete_user(self):
        self.engine.log_event(
            'u8', 'emotional', 'emotion_detected',
            payload={'emotion': 'joy'},
        )
        self.engine.process('u8')
        self.engine.delete_user('u8')

        events = self.engine.db.get_events('u8')
        self.assertEqual(len(events), 0)

    # ─── get_stage_progress ──────────────────────────────────────────────

    def test_get_stage_progress(self):
        self.engine.log_event(
            'u9', 'emotional', 'emotion_detected',
            payload={'emotion': 'joy'},
        )
        self.engine.process('u9')

        progress = self.engine.get_stage_progress('u9')
        self.assertIn('criteria', progress)
        self.assertIn('progress', progress)
        self.assertEqual(progress['next_stage'], 2)
        self.assertEqual(progress['next_stage_name'], 'Awareness')


if __name__ == '__main__':
    unittest.main()
