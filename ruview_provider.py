"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - RuView WiFi Sensing Provider
===============================================================================
Integrates RuView WiFi DensePose data as a passive biometric input channel.
Consumes real-time vital signs (HR, breathing rate), presence detection,
and body pose data via WebSocket/REST — no cameras, no wearables.

Integrates with:
- Neural Workflow AI Engine
- Biometric Integration Engine
- Emotion-Aware Training Engine
- Dashboard Data Integration
- Real-time Data

Ghost Protocol: RuView operates on WiFi CSI signals only — no optical
capture, no video, no images. All processing is local/edge.
===============================================================================
"""

import os
import time
import json
import logging
import threading
from collections import deque

import requests

logger = logging.getLogger(__name__)

# Optional: websocket-client for real-time streaming
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    logger.info("[RuViewProvider] websocket-client not installed. "
                "Install with: pip install websocket-client. "
                "REST polling will be used as fallback.")

API_TIMEOUT = 5  # seconds
DEFAULT_RUVIEW_HOST = 'localhost'
DEFAULT_HTTP_PORT = 8080
DEFAULT_WS_PORT = 8765


class RuViewProvider:
    """
    Consumes RuView WiFi sensing data and maps it to Quantara's
    biometric input format for emotion classification.

    Supports two modes:
    - WebSocket streaming (preferred, ~20 Hz real-time updates)
    - REST polling (fallback, on-demand queries)

    Connected to:
    - Neural Workflow AI Engine
    - Biometric Integration Engine
    - Dashboard Data Integration
    - Real-time Data
    """

    # Buffer size for rolling vital signs average
    VITALS_BUFFER_SIZE = 20  # ~1 second at 20 Hz

    # Cache TTL for REST polling mode
    CACHE_TTL = 5  # seconds

    # Breathing rate to HRV approximation
    # Higher breathing rate correlates with lower HRV (stress response)
    # Normal: 12-20 BPM → HRV 40-70 ms
    # Fast: 20-30 BPM → HRV 20-40 ms
    # Slow: 6-12 BPM → HRV 60-95 ms (relaxation/meditation)
    HRV_FROM_BREATHING = {
        'min_br': 6.0, 'max_br': 30.0,
        'min_hrv': 20.0, 'max_hrv': 95.0,
    }

    # Motion level to EDA approximation
    # Higher motion correlates with higher sympathetic activation
    # Still: 0.0-0.2 → EDA 0.5-2.0 µS
    # Active: 0.5-1.0 → EDA 3.0-8.0 µS
    EDA_FROM_MOTION = {
        'min_motion': 0.0, 'max_motion': 1.0,
        'min_eda': 0.5, 'max_eda': 8.0,
    }

    def __init__(
        self,
        host: str = None,
        http_port: int = None,
        ws_port: int = None,
        use_websocket: bool = True,
    ):
        self.host = host or os.environ.get('RUVIEW_HOST', DEFAULT_RUVIEW_HOST)
        self.http_port = int(http_port or os.environ.get('RUVIEW_HTTP_PORT', DEFAULT_HTTP_PORT))
        self.ws_port = int(ws_port or os.environ.get('RUVIEW_WS_PORT', DEFAULT_WS_PORT))

        self.base_url = f"http://{self.host}:{self.http_port}"
        self.ws_url = f"ws://{self.host}:{self.ws_port}/ws/sensing"

        # State
        self._connected = False
        self._latest_data = None
        self._latest_timestamp = 0
        self._vitals_buffer = deque(maxlen=self.VITALS_BUFFER_SIZE)
        self._pose_data = None
        self._presence_data = None

        # REST cache
        self._rest_cache = {}
        self._rest_cache_time = 0

        # WebSocket
        self._ws = None
        self._ws_thread = None
        self._use_websocket = use_websocket and HAS_WEBSOCKET
        self._reconnect_delay = 1  # seconds, doubles on failure up to 30s

        # Mood signals derived from RuView data
        self._mood_signals = []

    # ─── Connection Management ─────────────────────────────────────────

    def connect(self) -> bool:
        """
        Connect to RuView. Tries WebSocket first, falls back to REST health check.
        Returns True if connection established.
        """
        if self._use_websocket:
            return self._connect_websocket()
        return self._check_rest_health()

    def disconnect(self):
        """Disconnect from RuView WebSocket stream."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False
        logger.info("[RuViewProvider] Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _check_rest_health(self) -> bool:
        """Check if RuView REST API is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=API_TIMEOUT)
            resp.raise_for_status()
            self._connected = True
            logger.info(f"[RuViewProvider] REST API connected at {self.base_url}")
            return True
        except Exception as e:
            logger.warning(f"[RuViewProvider] REST health check failed: {e}")
            self._connected = False
            return False

    def _connect_websocket(self) -> bool:
        """Connect to RuView WebSocket stream in background thread."""
        if not HAS_WEBSOCKET:
            logger.warning("[RuViewProvider] websocket-client not available, falling back to REST")
            return self._check_rest_health()

        try:
            self._ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close,
                on_open=self._on_ws_open,
            )

            self._ws_thread = threading.Thread(
                target=self._ws.run_forever,
                kwargs={'reconnect': 5},
                daemon=True,
                name='ruview-ws'
            )
            self._ws_thread.start()

            # Wait briefly for connection
            for _ in range(10):
                if self._connected:
                    return True
                time.sleep(0.2)

            # If WS didn't connect, try REST
            if not self._connected:
                logger.warning("[RuViewProvider] WebSocket timeout, trying REST fallback")
                return self._check_rest_health()

            return self._connected

        except Exception as e:
            logger.warning(f"[RuViewProvider] WebSocket connection failed: {e}")
            return self._check_rest_health()

    def _on_ws_open(self, ws):
        self._connected = True
        self._reconnect_delay = 1
        logger.info(f"[RuViewProvider] WebSocket connected at {self.ws_url}")

    def _on_ws_message(self, ws, message):
        """Process incoming RuView WebSocket data."""
        try:
            data = json.loads(message)
            self._latest_data = data
            self._latest_timestamp = time.time()

            # Extract and buffer vital signs
            vitals = data.get('vital_signs') or data.get('vitals') or {}
            if vitals:
                self._vitals_buffer.append({
                    'heart_rate': vitals.get('heart_rate', vitals.get('hr_bpm', 0)),
                    'breathing_rate': vitals.get('breathing_rate', vitals.get('br_bpm', 0)),
                    'confidence': vitals.get('confidence', 0),
                    'timestamp': time.time(),
                })

            # Extract pose data
            pose = data.get('pose') or data.get('keypoints')
            if pose:
                self._pose_data = pose

            # Extract presence data
            raw_presence = data.get('presence')
            presence_obj = raw_presence if isinstance(raw_presence, dict) else {}
            if raw_presence is not None or 'occupancy' in data:
                self._presence_data = {
                    'detected': raw_presence if isinstance(raw_presence, bool)
                                else presence_obj.get('detected', False),
                    'occupancy': data.get('occupancy', presence_obj.get('count', 0)),
                    'motion_level': data.get('motion_level', presence_obj.get('motion', 0)),
                }

            # Derive mood signals
            self._update_mood_signals()

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"[RuViewProvider] Parse error: {e}")

    def _on_ws_error(self, ws, error):
        logger.warning(f"[RuViewProvider] WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg):
        self._connected = False
        logger.info(f"[RuViewProvider] WebSocket closed: {close_status_code} {close_msg}")

    # ─── Data Access ───────────────────────────────────────────────────

    def get_vital_signs(self) -> dict | None:
        """
        Get current vital signs from RuView.
        Returns averaged values from buffer (WebSocket) or single poll (REST).
        Returns None if unavailable.
        """
        # Try WebSocket buffer first
        if self._vitals_buffer:
            return self._averaged_vitals()

        # Fall back to REST polling
        return self._poll_vitals()

    def get_presence(self) -> dict | None:
        """
        Get current presence/occupancy data.
        Returns None if unavailable.
        """
        if self._presence_data:
            return self._presence_data

        # REST fallback
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/sensing/latest",
                timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                'detected': data.get('presence', False),
                'occupancy': data.get('occupancy', 0),
                'motion_level': data.get('motion_level', 0),
            }
        except Exception as e:
            logger.warning(f"[RuViewProvider] Presence fetch failed: {e}")
            return None

    def get_pose(self) -> dict | None:
        """
        Get current 17-keypoint body pose.
        Returns None if unavailable.
        """
        if self._pose_data:
            return self._pose_data

        # REST fallback
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/pose/current",
                timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"[RuViewProvider] Pose fetch failed: {e}")
            return None

    def get_biometrics(self) -> dict | None:
        """
        Get RuView data mapped to Quantara's biometric input format.
        This is the primary integration point for emotion classification.

        Returns dict compatible with BiometricEncoder:
            {heart_rate, hrv, eda} + RuView extras
        Or None if RuView is unavailable.
        """
        vitals = self.get_vital_signs()
        if not vitals:
            return None

        hr = vitals.get('heart_rate', 0)
        br = vitals.get('breathing_rate', 0)
        confidence = vitals.get('confidence', 0)
        motion = vitals.get('motion_level', 0)

        # Skip low-confidence readings
        if confidence < 0.3:
            logger.debug("[RuViewProvider] Low confidence reading, skipping")
            return None

        # Map breathing rate → HRV estimate
        hrv = self._breathing_to_hrv(br)

        # Map motion level → EDA estimate
        eda = self._motion_to_eda(motion)

        return {
            'heart_rate': hr,
            'hrv': hrv,
            'eda': eda,
            # RuView-specific extras (not consumed by BiometricEncoder but
            # available for dashboard/logging)
            'source': 'ruview_wifi',
            'breathing_rate': br,
            'motion_level': motion,
            'confidence': confidence,
            'occupancy': self._presence_data.get('occupancy', 0) if self._presence_data else 0,
            'presence': self._presence_data.get('detected', False) if self._presence_data else False,
            'mood_signals': list(self._mood_signals),
        }

    def get_mood_signals(self) -> list:
        """Get current mood signals derived from RuView data."""
        return list(self._mood_signals)

    # ─── Internal Helpers ──────────────────────────────────────────────

    def _poll_vitals(self) -> dict | None:
        """Poll vital signs via REST API with caching."""
        now = time.time()
        if self._rest_cache and (now - self._rest_cache_time) < self.CACHE_TTL:
            return self._rest_cache

        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/vital-signs",
                timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()

            result = {
                'heart_rate': data.get('heart_rate', data.get('hr_bpm', 0)),
                'breathing_rate': data.get('breathing_rate', data.get('br_bpm', 0)),
                'confidence': data.get('confidence', 0),
                'motion_level': data.get('motion_level', 0),
            }

            self._rest_cache = result
            self._rest_cache_time = now
            return result

        except Exception as e:
            logger.warning(f"[RuViewProvider] Vitals poll failed: {e}")
            return None

    def _averaged_vitals(self) -> dict:
        """Average recent vital sign readings from buffer."""
        if not self._vitals_buffer:
            return None

        readings = list(self._vitals_buffer)
        n = len(readings)

        # Weighted average — more recent readings get higher weight
        weights = [(i + 1) / n for i in range(n)]
        total_weight = sum(weights)

        avg_hr = sum(r['heart_rate'] * w for r, w in zip(readings, weights)) / total_weight
        avg_br = sum(r['breathing_rate'] * w for r, w in zip(readings, weights)) / total_weight
        avg_conf = sum(r['confidence'] * w for r, w in zip(readings, weights)) / total_weight

        motion = self._presence_data.get('motion_level', 0) if self._presence_data else 0

        return {
            'heart_rate': round(avg_hr, 1),
            'breathing_rate': round(avg_br, 1),
            'confidence': round(avg_conf, 2),
            'motion_level': motion,
        }

    def _breathing_to_hrv(self, breathing_rate: float) -> float:
        """
        Approximate HRV from breathing rate.
        Inverse relationship: slower breathing → higher HRV (relaxation).
        Research basis: respiratory sinus arrhythmia (RSA).
        """
        cfg = self.HRV_FROM_BREATHING
        if breathing_rate <= 0:
            return 50.0  # default

        # Clamp to expected range
        br = max(cfg['min_br'], min(cfg['max_br'], breathing_rate))

        # Inverse linear mapping: low BR → high HRV
        ratio = (cfg['max_br'] - br) / (cfg['max_br'] - cfg['min_br'])
        hrv = cfg['min_hrv'] + ratio * (cfg['max_hrv'] - cfg['min_hrv'])

        return round(hrv, 1)

    def _motion_to_eda(self, motion_level: float) -> float:
        """
        Approximate EDA from motion level.
        Direct relationship: more motion → higher sympathetic activation.
        """
        cfg = self.EDA_FROM_MOTION
        motion = max(cfg['min_motion'], min(cfg['max_motion'], motion_level))

        ratio = (motion - cfg['min_motion']) / (cfg['max_motion'] - cfg['min_motion'])
        eda = cfg['min_eda'] + ratio * (cfg['max_eda'] - cfg['min_eda'])

        return round(eda, 1)

    def _update_mood_signals(self):
        """Derive mood-relevant signals from current RuView data."""
        signals = []

        vitals = self._averaged_vitals() if self._vitals_buffer else None
        if vitals:
            hr = vitals.get('heart_rate', 0)
            br = vitals.get('breathing_rate', 0)

            if hr > 100:
                signals.append('elevated_heart_rate')
            if hr < 55:
                signals.append('low_heart_rate')
            if br > 22:
                signals.append('rapid_breathing')
            if br < 8:
                signals.append('slow_breathing')

        presence = self._presence_data or {}
        motion = presence.get('motion_level', 0)
        occupancy = presence.get('occupancy', 0)

        if motion > 0.7:
            signals.append('high_activity')
        if motion < 0.1 and presence.get('detected', False):
            signals.append('very_still')
        if occupancy > 3:
            signals.append('crowded_environment')
        if occupancy == 0:
            signals.append('alone')

        self._mood_signals = signals

    # ─── Insight Generation ────────────────────────────────────────────

    def get_insight(self) -> str | None:
        """
        Generate human-readable insight from RuView data.
        Compatible with ExternalContextProvider pattern.
        """
        biometrics = self.get_biometrics()
        if not biometrics:
            return None

        signals = biometrics.get('mood_signals', [])
        hr = biometrics.get('heart_rate', 0)
        br = biometrics.get('breathing_rate', 0)

        parts = []

        if 'elevated_heart_rate' in signals:
            parts.append(f"WiFi sensing detects elevated heart rate ({hr:.0f} BPM)")
        if 'rapid_breathing' in signals:
            parts.append(f"Breathing rate is elevated ({br:.0f} BPM) — possible stress response")
        if 'slow_breathing' in signals:
            parts.append(f"Deep, slow breathing detected ({br:.0f} BPM) — relaxation state")
        if 'very_still' in signals:
            parts.append("Very still posture detected — could indicate deep focus or low energy")
        if 'high_activity' in signals:
            parts.append("High physical activity detected")
        if 'crowded_environment' in signals:
            parts.append(f"Multiple people detected in environment ({biometrics.get('occupancy', 0)})")
        if 'alone' in signals:
            parts.append("Alone in environment — private session")

        if not parts:
            return f"WiFi sensing active: HR {hr:.0f} BPM, breathing {br:.0f} BPM — within normal ranges."

        return '. '.join(parts) + '.'


# ─── Singleton for shared use ──────────────────────────────────────────────

_provider_instance = None
_provider_lock = threading.Lock()


def get_ruview_provider(**kwargs) -> RuViewProvider:
    """Get or create shared RuViewProvider instance."""
    global _provider_instance
    with _provider_lock:
        if _provider_instance is None:
            _provider_instance = RuViewProvider(**kwargs)
        return _provider_instance
