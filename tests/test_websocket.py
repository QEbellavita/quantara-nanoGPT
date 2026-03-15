import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmotionWebSocket:

    def test_init_websocket_returns_socketio(self):
        from flask import Flask
        from emotion_websocket import init_websocket
        app = Flask(__name__)
        socketio = init_websocket(app)
        assert socketio is not None

    def test_emit_emotion_update_does_not_crash_without_clients(self):
        from flask import Flask
        from emotion_websocket import init_websocket, emit_emotion_update
        app = Flask(__name__)
        init_websocket(app)
        emit_emotion_update({
            'dominant_emotion': 'joy',
            'family': 'Joy',
            'confidence': 0.85,
            'scores': {},
            'modality_weights': {'text': 1.0, 'bio': 0.0, 'pose': 0.0}
        })

    def test_emit_transition_step_does_not_crash(self):
        from flask import Flask
        from emotion_websocket import init_websocket, emit_transition_step
        app = Flask(__name__)
        init_websocket(app)
        emit_transition_step({
            'session_id': 'test-1',
            'step': 1,
            'technique': 'Grounding',
            'from_emotion': 'anxiety',
            'to_emotion': 'grounded'
        })

    def test_disable_websocket_env_var(self):
        """When DISABLE_WEBSOCKET=1, init returns None."""
        import importlib
        import emotion_websocket
        os.environ['DISABLE_WEBSOCKET'] = '1'
        try:
            importlib.reload(emotion_websocket)
            from flask import Flask
            app = Flask(__name__)
            result = emotion_websocket.init_websocket(app)
            assert result is None
            assert emotion_websocket._socketio is None
        finally:
            del os.environ['DISABLE_WEBSOCKET']
            importlib.reload(emotion_websocket)
