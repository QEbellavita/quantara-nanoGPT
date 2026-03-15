"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Ecosystem Connector
===============================================================================
Unit tests for EcosystemConnector: single-domain routing, multi-domain fan-out,
dead-letter storage, service registration, and failure-based deregistration.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from profile_event_bus import ProfileEventBus
from profile_db import ProfileDB
from ecosystem_connector import EcosystemConnector, MULTI_DOMAIN_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_connector(tmp_path=None):
    """Return a (connector, bus, db) triple backed by a temp DB."""
    if tmp_path is None:
        tmp_path = tempfile.mktemp(suffix=".db")
    db = ProfileDB(db_path=tmp_path)
    bus = ProfileEventBus()
    connector = EcosystemConnector(bus=bus, db=db)
    return connector, bus, db


# ---------------------------------------------------------------------------
# MULTI_DOMAIN_MAP structure
# ---------------------------------------------------------------------------

class TestMultiDomainMap(unittest.TestCase):
    """Verify the MULTI_DOMAIN_MAP constant is correctly structured."""

    def test_map_contains_expected_keys(self):
        self.assertIn("biometric,temporal", MULTI_DOMAIN_MAP)
        self.assertIn("biometric,emotional", MULTI_DOMAIN_MAP)
        self.assertIn("linguistic,social,aspirational", MULTI_DOMAIN_MAP)
        self.assertIn("emotional,behavioral", MULTI_DOMAIN_MAP)
        self.assertIn("social,temporal", MULTI_DOMAIN_MAP)
        self.assertIn("behavioral,cognitive", MULTI_DOMAIN_MAP)

    def test_map_values_are_lists(self):
        for key, value in MULTI_DOMAIN_MAP.items():
            self.assertIsInstance(value, list, f"Value for '{key}' should be a list")
            self.assertTrue(len(value) >= 2, f"Value for '{key}' should have at least 2 domains")


# ---------------------------------------------------------------------------
# Single-domain routing
# ---------------------------------------------------------------------------

class TestSingleDomainRouting(unittest.TestCase):
    """Single-domain events are published to the correct event bus topic."""

    def setUp(self):
        self.connector, self.bus, self.db = _make_connector()
        self.received_topics = []

        def capture(topic, payload):
            self.received_topics.append(topic)

        self.bus.subscribe("event.*", capture, mode="sync")

    def tearDown(self):
        self.bus.shutdown()
        self.db.close()

    def test_single_domain_publishes_correct_topic(self):
        """A single-domain event publishes exactly one topic: event.<domain>."""
        event = {
            "user_id": "user_001",
            "domain": "emotion",
            "event_type": "spike",
            "payload": {"intensity": 0.9},
            "source": "sensor",
        }
        self.connector.route_inbound(event)
        self.assertEqual(self.received_topics, ["event.emotion"])

    def test_single_domain_cognitive_topic(self):
        """Cognitive domain routes to event.cognitive."""
        event = {
            "user_id": "user_002",
            "domain": "cognitive",
            "event_type": "assessment",
            "payload": {},
            "source": "api",
        }
        self.connector.route_inbound(event)
        self.assertIn("event.cognitive", self.received_topics)

    def test_payload_forwarded_to_bus(self):
        """The bus payload includes user_id, event_type, payload, and source."""
        received_payloads = []

        def capture_payload(topic, payload):
            received_payloads.append(payload)

        self.bus.subscribe("event.biometric", capture_payload, mode="sync")

        event = {
            "user_id": "user_003",
            "domain": "biometric",
            "event_type": "heart_rate",
            "payload": {"bpm": 72},
            "source": "wearable",
        }
        self.connector.route_inbound(event)

        self.assertEqual(len(received_payloads), 1)
        p = received_payloads[0]
        self.assertEqual(p["user_id"], "user_003")
        self.assertEqual(p["event_type"], "heart_rate")
        self.assertEqual(p["payload"], {"bpm": 72})
        self.assertEqual(p["source"], "wearable")


# ---------------------------------------------------------------------------
# Multi-domain routing
# ---------------------------------------------------------------------------

class TestMultiDomainRouting(unittest.TestCase):
    """Multi-domain events publish to each sub-domain topic separately."""

    def setUp(self):
        self.connector, self.bus, self.db = _make_connector()
        self.received_topics = []

        def capture(topic, payload):
            self.received_topics.append(topic)

        self.bus.subscribe("event.*", capture, mode="sync")

    def tearDown(self):
        self.bus.shutdown()
        self.db.close()

    def test_biometric_temporal_publishes_two_topics(self):
        """'biometric,temporal' domain fans out to event.biometric and event.temporal."""
        event = {
            "user_id": "user_m01",
            "domain": "biometric,temporal",
            "event_type": "timed_reading",
            "payload": {},
            "source": "sensor",
        }
        self.connector.route_inbound(event)
        self.assertIn("event.biometric", self.received_topics)
        self.assertIn("event.temporal", self.received_topics)
        self.assertEqual(len(self.received_topics), 2)

    def test_linguistic_social_aspirational_publishes_three_topics(self):
        """'linguistic,social,aspirational' domain fans out to three topics."""
        event = {
            "user_id": "user_m02",
            "domain": "linguistic,social,aspirational",
            "event_type": "conversation",
            "payload": {},
            "source": "api",
        }
        self.connector.route_inbound(event)
        self.assertIn("event.linguistic", self.received_topics)
        self.assertIn("event.social", self.received_topics)
        self.assertIn("event.aspirational", self.received_topics)
        self.assertEqual(len(self.received_topics), 3)

    def test_emotional_behavioral_publishes_two_topics(self):
        """'emotional,behavioral' domain fans out to two topics."""
        event = {
            "user_id": "user_m03",
            "domain": "emotional,behavioral",
            "event_type": "pattern",
            "payload": {},
            "source": "ml_pipeline",
        }
        self.connector.route_inbound(event)
        self.assertIn("event.emotional", self.received_topics)
        self.assertIn("event.behavioral", self.received_topics)

    def test_multi_domain_payload_identical_across_topics(self):
        """Each sub-domain topic receives the same payload."""
        received = {}

        def capture(topic, payload):
            received[topic] = payload

        self.bus.subscribe("event.biometric", capture, mode="sync")
        self.bus.subscribe("event.emotional", capture, mode="sync")

        event = {
            "user_id": "user_m04",
            "domain": "biometric,emotional",
            "event_type": "fusion",
            "payload": {"fused": True},
            "source": "sensor",
        }
        self.connector.route_inbound(event)

        self.assertIn("event.biometric", received)
        self.assertIn("event.emotional", received)
        self.assertEqual(received["event.biometric"]["user_id"], "user_m04")
        self.assertEqual(received["event.emotional"]["user_id"], "user_m04")


# ---------------------------------------------------------------------------
# Dead-letter queue
# ---------------------------------------------------------------------------

class TestDeadLetter(unittest.TestCase):
    """Invalid events are stored in the dead-letter queue."""

    def setUp(self):
        self.connector, self.bus, self.db = _make_connector()

    def tearDown(self):
        self.bus.shutdown()
        self.db.close()

    def test_missing_user_id_goes_to_dead_letter(self):
        """An event without user_id is sent to the dead-letter queue."""
        initial_count = self.connector.get_dead_letter_count()
        event = {
            "domain": "emotion",
            "event_type": "spike",
            "payload": {},
            "source": "sensor",
        }
        self.connector.route_inbound(event)
        self.assertEqual(self.connector.get_dead_letter_count(), initial_count + 1)

    def test_missing_domain_goes_to_dead_letter(self):
        """An event without domain is sent to the dead-letter queue."""
        initial_count = self.connector.get_dead_letter_count()
        event = {
            "user_id": "user_dl01",
            "event_type": "update",
            "payload": {},
        }
        self.connector.route_inbound(event)
        self.assertEqual(self.connector.get_dead_letter_count(), initial_count + 1)

    def test_missing_both_user_id_and_domain_goes_to_dead_letter(self):
        """An event missing both user_id and domain is stored as dead letter."""
        initial_count = self.connector.get_dead_letter_count()
        self.connector.route_inbound({})
        self.assertEqual(self.connector.get_dead_letter_count(), initial_count + 1)

    def test_valid_event_does_not_add_dead_letter(self):
        """A valid event does not create a dead-letter entry."""
        initial_count = self.connector.get_dead_letter_count()
        event = {
            "user_id": "user_valid",
            "domain": "emotion",
            "event_type": "baseline",
            "payload": {},
            "source": "api",
        }
        self.connector.route_inbound(event)
        self.assertEqual(self.connector.get_dead_letter_count(), initial_count)

    def test_dead_letter_count_accumulates(self):
        """Multiple invalid events accumulate in the dead-letter queue."""
        initial_count = self.connector.get_dead_letter_count()
        for _ in range(3):
            self.connector.route_inbound({"event_type": "orphan"})
        self.assertEqual(self.connector.get_dead_letter_count(), initial_count + 3)


# ---------------------------------------------------------------------------
# Service registration
# ---------------------------------------------------------------------------

class TestServiceRegistration(unittest.TestCase):
    """Service registration and URL allowlist validation."""

    def setUp(self):
        self.connector, self.bus, self.db = _make_connector()
        # Ensure no allowlist env var interferes by default
        os.environ.pop("PROFILE_ALLOWED_WEBHOOK_HOSTS", None)

    def tearDown(self):
        os.environ.pop("PROFILE_ALLOWED_WEBHOOK_HOSTS", None)
        self.bus.shutdown()
        self.db.close()

    def test_register_service_stored(self):
        """Registered service appears in get_registered_services()."""
        self.connector.register_service("analytics", "http://analytics.local/webhook")
        services = self.connector.get_registered_services()
        self.assertIn("analytics", services)
        self.assertEqual(services["analytics"], "http://analytics.local/webhook")

    def test_register_multiple_services(self):
        """Multiple services can be registered independently."""
        self.connector.register_service("svc_a", "http://svc-a.local/hook")
        self.connector.register_service("svc_b", "http://svc-b.local/hook")
        services = self.connector.get_registered_services()
        self.assertIn("svc_a", services)
        self.assertIn("svc_b", services)

    def test_allowlist_blocks_disallowed_host(self):
        """Registration raises ValueError when host is not in the allowlist."""
        os.environ["PROFILE_ALLOWED_WEBHOOK_HOSTS"] = "trusted.host.com"
        with self.assertRaises(ValueError):
            self.connector.register_service("bad_svc", "http://evil.host.com/hook")

    def test_allowlist_permits_allowed_host(self):
        """Registration succeeds when host is in the allowlist."""
        os.environ["PROFILE_ALLOWED_WEBHOOK_HOSTS"] = "trusted.host.com,other.host.com"
        self.connector.register_service("good_svc", "http://trusted.host.com/hook")
        self.assertIn("good_svc", self.connector.get_registered_services())

    def test_no_allowlist_permits_any_host(self):
        """When PROFILE_ALLOWED_WEBHOOK_HOSTS is unset, any URL is accepted."""
        self.connector.register_service("open_svc", "http://anywhere.example.com/hook")
        self.assertIn("open_svc", self.connector.get_registered_services())


# ---------------------------------------------------------------------------
# Failure tracking and deregistration
# ---------------------------------------------------------------------------

class TestDeregistrationAfterFailures(unittest.TestCase):
    """Services are deregistered after 5 consecutive delivery failures."""

    def setUp(self):
        self.connector, self.bus, self.db = _make_connector()
        os.environ.pop("PROFILE_ALLOWED_WEBHOOK_HOSTS", None)

    def tearDown(self):
        self.bus.shutdown()
        self.db.close()

    def test_service_deregistered_after_five_failures(self):
        """Calling record_delivery_failure 5 times removes the service."""
        self.connector.register_service("fragile_svc", "http://fragile.local/hook")
        self.assertIn("fragile_svc", self.connector.get_registered_services())

        for _ in range(5):
            self.connector.record_delivery_failure("fragile_svc")

        self.assertNotIn("fragile_svc", self.connector.get_registered_services())

    def test_service_survives_four_failures(self):
        """A service with 4 failures is not yet deregistered."""
        self.connector.register_service("sturdy_svc", "http://sturdy.local/hook")

        for _ in range(4):
            self.connector.record_delivery_failure("sturdy_svc")

        self.assertIn("sturdy_svc", self.connector.get_registered_services())

    def test_failure_count_resets_on_success(self):
        """The failure counter is reset after a successful delivery."""
        self.connector.register_service("recover_svc", "http://recover.local/hook")

        # Accumulate 3 failures manually
        for _ in range(3):
            self.connector.record_delivery_failure("recover_svc")

        # Simulate a successful delivery resetting the counter
        self.connector._services["recover_svc"]["failures"] = 0

        # Now add 4 more failures — should still be registered (total 4 < 5)
        for _ in range(4):
            self.connector.record_delivery_failure("recover_svc")

        self.assertIn("recover_svc", self.connector.get_registered_services())

    def test_record_failure_for_unknown_service_is_silent(self):
        """Calling record_delivery_failure for an unregistered service doesn't raise."""
        try:
            self.connector.record_delivery_failure("nonexistent_service")
        except Exception as exc:
            self.fail(f"record_delivery_failure raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
