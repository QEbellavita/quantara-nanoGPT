"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Ecosystem Connector
===============================================================================
Multi-domain event routing, dead-letter queue management, and intelligent
service delivery for the Quantara Neural Ecosystem. Bridges the event bus
with registered downstream services, handling retries, failure tracking, and
webhook allowlist validation.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-domain routing map
# ---------------------------------------------------------------------------

MULTI_DOMAIN_MAP: Dict[str, List[str]] = {
    "biometric,temporal": ["biometric", "temporal"],
    "biometric,emotional": ["biometric", "emotional"],
    "linguistic,social,aspirational": ["linguistic", "social", "aspirational"],
    "emotional,behavioral": ["emotional", "behavioral"],
    "social,temporal": ["social", "temporal"],
    "behavioral,cognitive": ["behavioral", "cognitive"],
}

# Retry delay schedule (seconds)
_RETRY_DELAYS = [1, 2, 4]


# ---------------------------------------------------------------------------
# EcosystemConnector
# ---------------------------------------------------------------------------

class EcosystemConnector:
    """
    Routes inbound events across domain topics, delivers intelligence to
    registered downstream services with retry logic, and maintains a
    dead-letter queue for events that cannot be processed.

    Parameters
    ----------
    bus : ProfileEventBus
        The shared event bus used for topic-based pub/sub.
    db : ProfileDB
        The profile database used for dead-letter persistence.
    """

    def __init__(self, bus, db) -> None:
        self._bus = bus
        self._db = db
        # {name: {'url': str, 'failures': int}}
        self._services: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Inbound routing
    # ------------------------------------------------------------------

    def route_inbound(self, event_data: dict) -> None:
        """
        Route an inbound event onto the event bus.

        Extracts ``user_id``, ``domain``, ``event_type``, ``payload``, and
        ``source`` from *event_data*. If the domain key matches a key in
        :data:`MULTI_DOMAIN_MAP` the event is published to each sub-domain
        topic individually. Otherwise it is published to ``event.<domain>``.

        Events missing ``user_id`` or ``domain`` are sent to the dead-letter
        queue instead of being routed.

        Parameters
        ----------
        event_data : dict
            Raw inbound event dictionary.
        """
        user_id = event_data.get("user_id")
        domain = event_data.get("domain")
        event_type = event_data.get("event_type", "unknown")
        payload = event_data.get("payload", {})
        source = event_data.get("source", "external")

        if not user_id or not domain:
            missing = "user_id" if not user_id else "domain"
            error_msg = f"Missing required field: {missing}"
            logger.warning("EcosystemConnector.route_inbound: %s — sending to dead letter", error_msg)
            self._store_dead_letter(event_data, error_msg)
            return

        bus_payload = {
            "user_id": user_id,
            "event_type": event_type,
            "payload": payload,
            "source": source,
        }

        if domain in MULTI_DOMAIN_MAP:
            for sub_domain in MULTI_DOMAIN_MAP[domain]:
                topic = f"event.{sub_domain}"
                logger.debug("EcosystemConnector: publishing to topic=%s (multi-domain)", topic)
                self._bus.publish(topic, bus_payload)
        else:
            topic = f"event.{domain}"
            logger.debug("EcosystemConnector: publishing to topic=%s", topic)
            self._bus.publish(topic, bus_payload)

    # ------------------------------------------------------------------
    # Intelligence delivery
    # ------------------------------------------------------------------

    def deliver_intelligence(self, intelligence_type: str, payload: dict) -> None:
        """
        Deliver an intelligence update to all registered services.

        Each service receives a POST with ``schema_version=1`` included in
        the delivery payload. Delivery failures are tracked via
        :meth:`record_delivery_failure`.

        Parameters
        ----------
        intelligence_type : str
            Categorisation label for the intelligence being delivered.
        payload : dict
            Intelligence data to forward.
        """
        services = dict(self._services)
        for name, info in services.items():
            self._deliver_to_service(name, info["url"], intelligence_type, payload)

    def _deliver_to_service(
        self,
        name: str,
        url: str,
        intelligence_type: str,
        payload: dict,
    ) -> None:
        """
        Attempt HTTP delivery to a single registered service with up to 3
        retries using delays of [1, 2, 4] seconds between attempts.

        On success the service's failure counter is reset to 0. On exhausting
        all retries :meth:`record_delivery_failure` is called.

        Parameters
        ----------
        name : str
            Registered service name (used for failure tracking).
        url : str
            Target URL for HTTP POST delivery.
        intelligence_type : str
            Delivery type label included in the POST body.
        payload : dict
            Intelligence data to deliver.
        """
        body = json.dumps({
            "intelligence_type": intelligence_type,
            "schema_version": 1,
            "payload": payload,
        }).encode("utf-8")

        last_exc: Optional[Exception] = None
        for attempt, delay in enumerate(_RETRY_DELAYS):
            if attempt > 0:
                time.sleep(delay)
            try:
                req = Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=10) as resp:
                    status = resp.status
                if status < 400:
                    logger.debug(
                        "EcosystemConnector: delivered to service=%s status=%s", name, status
                    )
                    # Reset failures on success
                    if name in self._services:
                        self._services[name]["failures"] = 0
                    return
                last_exc = RuntimeError(f"HTTP {status}")
            except (URLError, OSError, RuntimeError) as exc:
                last_exc = exc
                logger.warning(
                    "EcosystemConnector: delivery attempt %d/%d to service=%s failed: %s",
                    attempt + 1,
                    len(_RETRY_DELAYS),
                    name,
                    exc,
                )

        logger.error(
            "EcosystemConnector: all delivery attempts failed for service=%s last_error=%s",
            name,
            last_exc,
        )
        self.record_delivery_failure(name)

    # ------------------------------------------------------------------
    # Service registry
    # ------------------------------------------------------------------

    def register_service(self, name: str, url: str) -> None:
        """
        Register a downstream service for intelligence delivery.

        If the ``PROFILE_ALLOWED_WEBHOOK_HOSTS`` environment variable is set
        it must contain a comma-separated allowlist of hostnames. The supplied
        *url* is validated against this list; an invalid URL raises
        ``ValueError``.

        Parameters
        ----------
        name : str
            Human-readable service identifier.
        url : str
            Target webhook URL (must be HTTP or HTTPS).

        Raises
        ------
        ValueError
            If the URL's hostname is not in the allowlist when one is configured.
        """
        allowlist_env = os.environ.get("PROFILE_ALLOWED_WEBHOOK_HOSTS", "")
        if allowlist_env:
            allowed_hosts = [h.strip() for h in allowlist_env.split(",") if h.strip()]
            parsed = urlparse(url)
            if parsed.hostname not in allowed_hosts:
                raise ValueError(
                    f"URL hostname '{parsed.hostname}' is not in the allowed webhook hosts: {allowed_hosts}"
                )

        self._services[name] = {"url": url, "failures": 0}
        logger.info("EcosystemConnector: registered service name=%s url=%s", name, url)

    def get_registered_services(self) -> Dict[str, str]:
        """
        Return a mapping of registered service names to their URLs.

        Returns
        -------
        dict
            ``{name: url}`` for each registered service.
        """
        return {name: info["url"] for name, info in self._services.items()}

    def record_delivery_failure(self, name: str) -> None:
        """
        Increment a service's failure count and deregister it after 5 failures.

        Parameters
        ----------
        name : str
            Registered service name.
        """
        if name not in self._services:
            logger.debug("EcosystemConnector: record_delivery_failure for unknown service=%s", name)
            return

        self._services[name]["failures"] += 1
        failures = self._services[name]["failures"]
        logger.warning(
            "EcosystemConnector: service=%s failure count=%d", name, failures
        )

        if failures >= 5:
            del self._services[name]
            logger.error(
                "EcosystemConnector: service=%s deregistered after %d consecutive failures",
                name,
                failures,
            )

    # ------------------------------------------------------------------
    # Dead-letter queue
    # ------------------------------------------------------------------

    def _store_dead_letter(self, event_data: dict, error: str) -> None:
        """
        Persist an unroutable event to the ``dead_letter_events`` table.

        Parameters
        ----------
        event_data : dict
            The original event that could not be routed.
        error : str
            Human-readable description of why routing failed.
        """
        now = time.time()
        payload_json = json.dumps(event_data)
        self._db._enqueue_write(
            "INSERT INTO dead_letter_events (timestamp, payload, error, retries) VALUES (?, ?, ?, 0)",
            (now, payload_json, error),
            wait=True,
        )
        logger.debug("EcosystemConnector: stored dead letter event error=%r", error)

    def get_dead_letter_count(self) -> int:
        """
        Return the number of rows in the dead-letter queue.

        Returns
        -------
        int
            Total count of dead-letter events.
        """
        conn = self._db._read_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM dead_letter_events").fetchone()
            return row[0]
        finally:
            conn.close()

    def replay_dead_letters(self) -> Dict[str, int]:
        """
        Attempt to re-route all pending dead-letter events.

        Successfully routed events are deleted from the queue. Events that
        still fail have their ``retries`` counter incremented.

        Returns
        -------
        dict
            ``{'replayed': int, 'failed': int}`` counts.
        """
        conn = self._db._read_conn()
        try:
            rows = conn.execute(
                "SELECT id, payload, retries FROM dead_letter_events"
            ).fetchall()
        finally:
            conn.close()

        replayed = 0
        failed = 0

        for row in rows:
            row_id = row["id"]
            try:
                event_data = json.loads(row["payload"])
            except (json.JSONDecodeError, TypeError) as exc:
                logger.error(
                    "EcosystemConnector: dead letter id=%s has invalid JSON: %s", row_id, exc
                )
                failed += 1
                self._db._enqueue_write(
                    "UPDATE dead_letter_events SET retries = retries + 1 WHERE id = ?",
                    (row_id,),
                    wait=True,
                )
                continue

            # Check whether the event is now routable (has user_id and domain)
            user_id = event_data.get("user_id")
            domain = event_data.get("domain")

            if user_id and domain:
                try:
                    self.route_inbound(event_data)
                    # Delete on success
                    self._db._enqueue_write(
                        "DELETE FROM dead_letter_events WHERE id = ?",
                        (row_id,),
                        wait=True,
                    )
                    replayed += 1
                    logger.debug("EcosystemConnector: replayed dead letter id=%s", row_id)
                except Exception as exc:
                    logger.warning(
                        "EcosystemConnector: replay failed for dead letter id=%s: %s", row_id, exc
                    )
                    self._db._enqueue_write(
                        "UPDATE dead_letter_events SET retries = retries + 1 WHERE id = ?",
                        (row_id,),
                        wait=True,
                    )
                    failed += 1
            else:
                self._db._enqueue_write(
                    "UPDATE dead_letter_events SET retries = retries + 1 WHERE id = ?",
                    (row_id,),
                    wait=True,
                )
                failed += 1

        logger.info(
            "EcosystemConnector.replay_dead_letters: replayed=%d failed=%d", replayed, failed
        )
        return {"replayed": replayed, "failed": failed}
