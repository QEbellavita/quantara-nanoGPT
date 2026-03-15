"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Next-Level End-to-End Integration Tests
===============================================================================
End-to-end test: event in -> bus -> process -> intelligence out -> alerts.

Validates the full event bus pipeline across all wired components:
ProfileEventBus, UserProfileEngine, EcosystemConnector, ProcessScheduler,
IntelligencePublisher, and AlertEngine.

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
import time

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from profile_event_bus import ProfileEventBus
from user_profile_engine import UserProfileEngine
from ecosystem_connector import EcosystemConnector
from process_scheduler import ProcessScheduler


SERVICE_KEY = 'next-level-integration-key'


class TestNextLevelIntegration:
    """End-to-end integration tests for the full event bus pipeline."""

    @pytest.fixture()
    def system(self, tmp_path):
        """Create the full wired system: bus, engine, connector, scheduler."""
        db_path = str(tmp_path / 'integration_test.db')

        bus = ProfileEventBus()
        engine = UserProfileEngine(db_path=db_path)
        connector = EcosystemConnector(bus, engine.db)
        engine.set_event_bus(bus, connector)
        scheduler = ProcessScheduler(
            process_fn=engine.process,
            debounce_seconds=0.2,
            count_threshold=100,
            periodic_seconds=600,
        )
        scheduler.start()

        yield {
            'bus': bus,
            'engine': engine,
            'connector': connector,
            'scheduler': scheduler,
        }

        scheduler.stop()
        bus.shutdown()
        engine.close()

    # ------------------------------------------------------------------
    # Test 1: event flows through bus to profile
    # ------------------------------------------------------------------

    def test_event_flows_through_bus_to_profile(self, system):
        """Route an inbound emotional event via connector, wait for debounce,
        verify the profile exists in the database."""
        bus = system['bus']
        engine = system['engine']
        connector = system['connector']

        user_id = 'flow-test-user'

        # Subscribe to event.emotional so we can also log it into the engine
        received = []

        def on_event(topic, payload):
            received.append((topic, payload))
            # Log the event into the engine as the bus subscriber would
            engine.log_event(
                user_id=payload['user_id'],
                domain='emotional',
                event_type=payload.get('event_type', 'emotion_detected'),
                payload=payload.get('payload', {}),
                source=payload.get('source', 'bus'),
            )

        bus.subscribe('event.emotional', on_event, mode='sync')

        # Route an emotional event through the connector
        connector.route_inbound({
            'user_id': user_id,
            'domain': 'emotional',
            'event_type': 'emotion_detected',
            'payload': {'emotion': 'joy', 'confidence': 0.85},
            'source': 'integration-test',
        })

        # Wait for processing
        time.sleep(0.5)

        # Verify the event was received on the bus
        assert len(received) >= 1
        assert received[0][0] == 'event.emotional'
        assert received[0][1]['user_id'] == user_id

        # Verify the profile exists in the database
        profile = engine.get_profile(user_id)
        assert profile is not None
        assert profile['user_id'] == user_id

    # ------------------------------------------------------------------
    # Test 2: intelligence published after process
    # ------------------------------------------------------------------

    def test_intelligence_published_after_process(self, system):
        """Subscribe to intelligence.*, log events, call process(), and
        verify intelligence.therapy and intelligence.coaching were published."""
        bus = system['bus']
        engine = system['engine']

        user_id = 'intel-test-user'

        # Collect intelligence events
        intel_events = []

        def on_intelligence(topic, payload):
            intel_events.append((topic, payload))

        bus.subscribe('intelligence.*', on_intelligence, mode='sync')

        # Log 5 emotional events
        for i in range(5):
            engine.log_event(
                user_id, 'emotional', 'emotion_detected',
                payload={'emotion': 'calm', 'confidence': 0.8},
            )

        # Call process directly
        engine.process(user_id)

        # Wait for async delivery
        time.sleep(0.3)

        # Collect topics that were published
        published_topics = [t for t, _ in intel_events]
        assert 'intelligence.therapy' in published_topics, (
            f"Expected intelligence.therapy in {published_topics}"
        )
        assert 'intelligence.coaching' in published_topics, (
            f"Expected intelligence.coaching in {published_topics}"
        )

    # ------------------------------------------------------------------
    # Test 3: multi-domain event routing
    # ------------------------------------------------------------------

    def test_multi_domain_event_routing(self, system):
        """Route an event with domain='biometric,temporal' and verify both
        event.biometric and event.temporal are received."""
        bus = system['bus']
        connector = system['connector']

        received = []

        def on_event(topic, payload):
            received.append((topic, payload))

        bus.subscribe('event.*', on_event, mode='sync')

        connector.route_inbound({
            'user_id': 'multi-domain-user',
            'domain': 'biometric,temporal',
            'event_type': 'multi_reading',
            'payload': {'hr': 72, 'time_of_day': 'morning'},
            'source': 'wearable',
        })

        # Filter for our specific topics
        topics = [t for t, _ in received]
        assert 'event.biometric' in topics, (
            f"Expected event.biometric in {topics}"
        )
        assert 'event.temporal' in topics, (
            f"Expected event.temporal in {topics}"
        )

    # ------------------------------------------------------------------
    # Test 4: alert fires on negative spiral
    # ------------------------------------------------------------------

    def test_alert_fires_on_negative_spiral(self, system):
        """Log 6 emotional events with family='Sadness' and verify the
        emotional_spiral alert is published to alert.reactive."""
        bus = system['bus']

        user_id = 'spiral-test-user'

        # Collect reactive alerts
        alerts = []

        def on_alert(topic, payload):
            alerts.append((topic, payload))

        bus.subscribe('alert.reactive', on_alert, mode='sync')

        # Publish 6 negative emotional events directly to the bus with
        # the shape that AlertEngine's ReactiveDetector expects:
        # top-level user_id, emotion_family, and timestamp fields.
        now = time.time()
        for i in range(6):
            bus.publish('event.emotional', {
                'user_id': user_id,
                'event_type': 'emotion_detected',
                'emotion_family': 'Sadness',
                'emotion': 'deep_sadness',
                'timestamp': now - (5 - i),  # within last 2 hours
                'source': 'integration-test',
            })

        # Wait for async AlertEngine processing (it subscribes in async mode)
        time.sleep(0.5)

        # Check for emotional_spiral alert
        alert_types = [p.get('alert_type') for _, p in alerts]
        assert 'emotional_spiral' in alert_types, (
            f"Expected emotional_spiral alert in {alert_types}. "
            f"Got {len(alerts)} alerts total."
        )

    # ------------------------------------------------------------------
    # Test 5: full API with bus
    # ------------------------------------------------------------------

    def test_full_api_with_bus(self, system):
        """Create a Flask app with the profile blueprint, POST ingest,
        GET snapshot, and POST fingerprint sync - all return 200/201."""
        from flask import Flask
        from profile_api import create_profile_blueprint

        engine = system['engine']
        user_id = 'api-bus-user'

        os.environ['PROFILE_SERVICE_KEY_BACKEND'] = SERVICE_KEY

        try:
            app = Flask(__name__)
            app.config['TESTING'] = True
            bp = create_profile_blueprint(engine)
            app.register_blueprint(bp)
            client = app.test_client()

            # POST to /api/profile/ingest
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
            assert resp.status_code == 201, (
                f"Ingest expected 201, got {resp.status_code}: {resp.get_json()}"
            )

            # GET snapshot
            resp = client.get(f'/api/profile/{user_id}/snapshot')
            assert resp.status_code == 200, (
                f"Snapshot expected 200, got {resp.status_code}: {resp.get_json()}"
            )

            # POST fingerprint sync
            resp = client.post(f'/api/v1/users/{user_id}/genetic-fingerprint/sync')
            assert resp.status_code == 200, (
                f"Sync expected 200, got {resp.status_code}: {resp.get_json()}"
            )

        finally:
            os.environ.pop('PROFILE_SERVICE_KEY_BACKEND', None)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
