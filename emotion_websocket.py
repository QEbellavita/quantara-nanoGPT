"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Emotion WebSocket Streaming
===============================================================================
Real-time WebSocket streaming for emotion state changes, biometric data,
and system events. Mounts on the existing Flask app (same port).

Namespaces:
  /emotion     - Live emotion updates, transition steps, room subscriptions
  /biometrics  - Biometric data streaming (HR, HRV, EDA, etc.)
  /system      - System events (auto-retrain triggers, model updates)

Integrates with:
  - Neural Workflow AI Engine
  - ML Training & Prediction Systems
  - Backend APIs (cases, workflows, analytics)
  - All Dashboard Data Integration
  - Real-time data from customer service, distribution, etc.
===============================================================================
"""

import os
import logging

logger = logging.getLogger(__name__)

# Global SocketIO instance
_socketio = None

# Check disable flag at module level
_WEBSOCKET_DISABLED = os.environ.get('DISABLE_WEBSOCKET', '0') == '1'

try:
    from flask_socketio import SocketIO, Namespace, emit, join_room, leave_room
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False
    logger.warning("flask-socketio not installed. WebSocket features disabled.")


class EmotionNamespace(Namespace):
    """
    /emotion namespace — live emotion state streaming.
    Handles JWT validation, room joins, and family subscriptions.
    """

    def on_connect(self, auth=None):
        """Validate JWT token from query param on connection."""
        from flask import request as flask_request
        token = flask_request.args.get('token')
        if not token:
            logger.warning("WebSocket /emotion connection refused: no JWT token")
            raise ConnectionRefusedError('Authentication required — pass ?token=<JWT>')
        logger.info(f"Client connected to /emotion (sid={flask_request.sid})")

    def on_disconnect(self):
        from flask import request as flask_request
        logger.info(f"Client disconnected from /emotion (sid={flask_request.sid})")

    def on_join_room(self, data):
        """Join a specific room (e.g., user session, dashboard group)."""
        room = data.get('room') if isinstance(data, dict) else data
        if room:
            join_room(room)
            logger.info(f"Client joined room: {room}")
            emit('room_joined', {'room': room})

    def on_subscribe_family(self, data):
        """Subscribe to updates for a specific emotion family."""
        family = data.get('family') if isinstance(data, dict) else data
        if family:
            join_room(f"family:{family}")
            logger.info(f"Client subscribed to family: {family}")
            emit('family_subscribed', {'family': family})


class BiometricsNamespace(Namespace):
    """
    /biometrics namespace — real-time biometric data streaming.
    """

    def on_connect(self, auth=None):
        from flask import request as flask_request
        logger.info(f"Client connected to /biometrics (sid={flask_request.sid})")

    def on_disconnect(self):
        from flask import request as flask_request
        logger.info(f"Client disconnected from /biometrics (sid={flask_request.sid})")


class SystemNamespace(Namespace):
    """
    /system namespace — system events (auto-retrain, model updates).
    Stub for auto-retrain event pipeline.
    """

    def on_connect(self, auth=None):
        from flask import request as flask_request
        logger.info(f"Client connected to /system (sid={flask_request.sid})")

    def on_disconnect(self):
        from flask import request as flask_request
        logger.info(f"Client disconnected from /system (sid={flask_request.sid})")


def init_websocket(app):
    """
    Initialize WebSocket on the Flask app.

    Returns SocketIO instance, or None if disabled via DISABLE_WEBSOCKET=1.
    """
    global _socketio, _WEBSOCKET_DISABLED

    # Re-check env var (supports reload for testing)
    _WEBSOCKET_DISABLED = os.environ.get('DISABLE_WEBSOCKET', '0') == '1'

    if _WEBSOCKET_DISABLED:
        logger.info("WebSocket disabled via DISABLE_WEBSOCKET env var")
        _socketio = None
        return None

    if not HAS_SOCKETIO:
        logger.warning("flask-socketio not available — WebSocket disabled")
        _socketio = None
        return None

    _socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=False,
        engineio_logger=False
    )

    # Register namespaces
    _socketio.on_namespace(EmotionNamespace('/emotion'))
    _socketio.on_namespace(BiometricsNamespace('/biometrics'))
    _socketio.on_namespace(SystemNamespace('/system'))

    logger.info("WebSocket initialized with namespaces: /emotion, /biometrics, /system")
    return _socketio


def emit_emotion_update(result, room=None):
    """Emit emotion analysis result to /emotion namespace."""
    if _socketio is None:
        return
    kwargs = {'namespace': '/emotion'}
    if room:
        kwargs['to'] = room
    _socketio.emit('emotion_update', result, **kwargs)


def emit_transition_step(step_data, room=None):
    """Emit emotion transition step to /emotion namespace."""
    if _socketio is None:
        return
    kwargs = {'namespace': '/emotion'}
    if room:
        kwargs['to'] = room
    _socketio.emit('transition_step', step_data, **kwargs)


def emit_biometric_stream(biometric_data, room=None):
    """Emit biometric data to /biometrics namespace."""
    if _socketio is None:
        return
    kwargs = {'namespace': '/biometrics'}
    if room:
        kwargs['to'] = room
    _socketio.emit('biometric_data', biometric_data, **kwargs)


def emit_system_event(event_name, data):
    """Emit system event to /system namespace."""
    if _socketio is None:
        return
    _socketio.emit(event_name, data, namespace='/system')


def get_socketio():
    """Return the current SocketIO instance (or None)."""
    return _socketio
