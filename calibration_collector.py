"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Calibration Data Collector
===============================================================================
Collects paired (RuView WiFi, wearable) biometric readings for training
and fine-tuning the WiFi calibration model. Runs a timed session that
simultaneously polls RuView and the Quantara Backend HealthKit API.

Integrates with:
- Neural Workflow AI Engine
- Biometric Integration Engine
- RuView WiFi Sensing Provider
- Real-time Data
- Dashboard Data Integration

Usage:
  # Collect 30 minutes of paired data
  python calibration_collector.py --duration 30 --output calibration_data/session_001.json

  # Collect with custom endpoints
  python calibration_collector.py --ruview-url http://localhost:8080 \
      --healthkit-url http://localhost:3000 --duration 15

  # Retrain calibration model from collected data
  python calibration_collector.py --retrain --data-dir calibration_data/
===============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

import requests
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RUVIEW_URL = 'http://localhost:8080'
DEFAULT_HEALTHKIT_URL = 'http://localhost:3000'
DEFAULT_EMOTION_API_URL = 'http://localhost:5050'
POLL_INTERVAL = 5  # seconds between readings


class CalibrationCollector:
    """
    Collects paired biometric readings from RuView (WiFi) and HealthKit (wearable).
    Stores timestamped pairs for offline training or feeds them directly to
    PersonalCalibrationBuffer for online adaptation.

    Connected to:
    - Neural Workflow AI Engine
    - Biometric Integration Engine
    - RuView WiFi Sensing Provider
    - Real-time Data
    """

    def __init__(
        self,
        ruview_url: str = None,
        healthkit_url: str = None,
        emotion_api_url: str = None,
        poll_interval: float = POLL_INTERVAL,
    ):
        self.ruview_url = ruview_url or os.environ.get('RUVIEW_URL', DEFAULT_RUVIEW_URL)
        self.healthkit_url = healthkit_url or os.environ.get('HEALTHKIT_URL', DEFAULT_HEALTHKIT_URL)
        self.emotion_api_url = emotion_api_url or os.environ.get('EMOTION_API_URL', DEFAULT_EMOTION_API_URL)
        self.poll_interval = poll_interval
        self.pairs = []
        self._session_start = None

    def collect_session(self, duration_minutes: float = 30, output_path: str = None) -> list:
        """
        Run a timed collection session.

        Polls RuView and HealthKit simultaneously every poll_interval seconds.
        Stores paired readings with timestamps.

        Args:
            duration_minutes: Session duration in minutes.
            output_path: Path to save collected data as JSON.

        Returns:
            List of paired readings.
        """
        self._session_start = datetime.now()
        duration_secs = duration_minutes * 60
        end_time = time.time() + duration_secs

        print(f"{'=' * 60}")
        print(f"  CALIBRATION DATA COLLECTION SESSION")
        print(f"{'=' * 60}")
        print(f"  Duration: {duration_minutes} minutes")
        print(f"  Poll interval: {self.poll_interval}s")
        print(f"  RuView: {self.ruview_url}")
        print(f"  HealthKit: {self.healthkit_url}")
        print(f"  Started: {self._session_start.isoformat()}")
        print(f"{'=' * 60}")
        print()
        print("  Ensure your Apple Watch / wearable is active.")
        print("  Stay in the room with WiFi sensing coverage.")
        print("  Press Ctrl+C to stop early.")
        print()

        reading_count = 0

        try:
            while time.time() < end_time:
                pair = self._collect_one_pair()
                if pair:
                    self.pairs.append(pair)
                    reading_count += 1
                    elapsed = time.time() - (end_time - duration_secs)
                    remaining = max(0, end_time - time.time())

                    wifi_hr = pair['wifi'].get('heart_rate', '?')
                    wifi_br = pair['wifi'].get('breathing_rate', '?')
                    wear_hr = pair['wearable'].get('heart_rate', '?')
                    wear_hrv = pair['wearable'].get('hrv', '?')

                    print(f"  [{reading_count:4d}] WiFi: HR={wifi_hr}, BR={wifi_br} | "
                          f"Watch: HR={wear_hr}, HRV={wear_hrv} | "
                          f"{remaining/60:.1f}m remaining")
                else:
                    print(f"  [----] Skipped (data unavailable from one or both sources)")

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print(f"\n  Session stopped early by user.")

        session_end = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"  SESSION COMPLETE")
        print(f"  Paired readings: {len(self.pairs)}")
        print(f"  Duration: {(session_end - self._session_start).seconds // 60}m "
              f"{(session_end - self._session_start).seconds % 60}s")
        print(f"{'=' * 60}")

        if output_path:
            self._save_session(output_path)

        return self.pairs

    def _collect_one_pair(self) -> dict | None:
        """Collect one paired reading from RuView + HealthKit."""
        wifi = self._get_ruview_reading()
        wearable = self._get_healthkit_reading()

        if not wifi or not wearable:
            return None

        # Both sources must have at least heart_rate
        if not wifi.get('heart_rate') or not wearable.get('heart_rate'):
            return None

        return {
            'timestamp': datetime.now().isoformat(),
            'wifi': wifi,
            'wearable': wearable,
        }

    def _get_ruview_reading(self) -> dict | None:
        """Get current vital signs from RuView."""
        # Try via Emotion API first (which has the calibration-aware provider)
        try:
            resp = requests.get(
                f"{self.emotion_api_url}/api/ruview/biometrics",
                timeout=3
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get('status') == 'success':
                    return {
                        'heart_rate': data.get('heart_rate'),
                        'breathing_rate': data.get('breathing_rate'),
                        'motion_level': data.get('motion_level'),
                        'confidence': data.get('confidence'),
                    }
        except Exception:
            pass

        # Fall back to direct RuView API
        try:
            resp = requests.get(
                f"{self.ruview_url}/api/v1/vital-signs",
                timeout=3
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'heart_rate': data.get('heart_rate', data.get('hr_bpm')),
                    'breathing_rate': data.get('breathing_rate', data.get('br_bpm')),
                    'motion_level': data.get('motion_level', 0),
                    'confidence': data.get('confidence', 0),
                }
        except Exception as e:
            logger.debug(f"[Collector] RuView unavailable: {e}")

        return None

    def _get_healthkit_reading(self) -> dict | None:
        """
        Get latest wearable biometrics from HealthKit sync API.
        Falls back to direct REST query if sync API unavailable.
        """
        # Try Quantara Backend HealthKit API
        try:
            resp = requests.get(
                f"{self.healthkit_url}/api/healthkit/latest",
                timeout=3
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'heart_rate': data.get('heart_rate', data.get('hr')),
                    'hrv': data.get('hrv', data.get('heart_rate_variability')),
                    'respiratory_rate': data.get('respiratory_rate'),
                    'eda': data.get('eda', data.get('electrodermal_activity')),
                    'source': data.get('source', 'healthkit'),
                }
        except Exception:
            pass

        # Try alternative endpoints
        for endpoint in ['/api/biometric/latest', '/api/health/vitals']:
            try:
                resp = requests.get(
                    f"{self.healthkit_url}{endpoint}",
                    timeout=3
                )
                if resp.status_code == 200:
                    data = resp.json()
                    hr = data.get('heart_rate') or data.get('hr')
                    if hr:
                        return {
                            'heart_rate': hr,
                            'hrv': data.get('hrv') or data.get('heart_rate_variability'),
                            'respiratory_rate': data.get('respiratory_rate'),
                            'eda': data.get('eda'),
                            'source': 'backend_api',
                        }
            except Exception:
                continue

        return None

    def _save_session(self, output_path: str):
        """Save collected pairs to JSON file."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        session_data = {
            'session_start': self._session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'pair_count': len(self.pairs),
            'poll_interval': self.poll_interval,
            'ruview_url': self.ruview_url,
            'healthkit_url': self.healthkit_url,
            'pairs': self.pairs,
        }

        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"  Saved to: {output_path}")

    def feed_to_calibration_buffer(self, profile_id: str = None):
        """
        Feed collected pairs directly into PersonalCalibrationBuffer
        for immediate online adaptation.
        """
        if not self.pairs:
            print("  No pairs to feed.")
            return

        try:
            from wifi_calibration import load_calibration_model, PersonalCalibrationBuffer, WiFiCalibrationModel

            # Load or create calibration model
            model, _ = load_calibration_model()
            if model is None:
                print("  No base calibration model found. Train one first:")
                print("    python wifi_calibration.py --train")
                return

            buffer = PersonalCalibrationBuffer(
                base_model=model,
                profile_id=profile_id,
            )

            fed_count = 0
            for pair in self.pairs:
                wifi = pair['wifi']
                wearable = pair['wearable']

                # Need breathing_rate + motion_level from WiFi
                br = wifi.get('breathing_rate')
                motion = wifi.get('motion_level', 0)

                # Need HRV + EDA from wearable
                hrv = wearable.get('hrv')
                eda = wearable.get('eda')

                if br is not None and hrv is not None:
                    # EDA may not be available from all wearables
                    if eda is None:
                        eda = 2.0  # Default neutral EDA

                    buffer.add_pair(
                        wifi={'breathing_rate': float(br), 'motion_level': float(motion)},
                        wearable={'hrv': float(hrv), 'eda': float(eda)}
                    )
                    fed_count += 1

            print(f"  Fed {fed_count}/{len(self.pairs)} pairs to calibration buffer")
            print(f"  Fine-tune count: {buffer.fine_tune_count}")
            print(f"  Profile: {buffer.profile_id}")

        except ImportError as e:
            print(f"  Error: {e}")
            print("  Make sure wifi_calibration.py is available.")


def retrain_from_collected_data(data_dir: str, output: str = 'checkpoints/ruview_calibration.pt'):
    """
    Retrain calibration model using collected paired data files.
    Combines all JSON session files in data_dir.
    """
    import torch
    from wifi_calibration import WiFiCalibrationModel, HRV_MIN, HRV_MAX, EDA_MIN, EDA_MAX

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"  Data directory not found: {data_dir}")
        return

    # Load all session files
    all_pairs = []
    for json_file in sorted(data_path.glob('*.json')):
        with open(json_file) as f:
            session = json.load(f)
        pairs = session.get('pairs', [])
        all_pairs.extend(pairs)
        print(f"  Loaded {len(pairs)} pairs from {json_file.name}")

    if not all_pairs:
        print("  No paired data found.")
        return

    # Extract training data
    inputs = []
    targets = []

    for pair in all_pairs:
        wifi = pair['wifi']
        wearable = pair['wearable']

        br = wifi.get('breathing_rate')
        motion = wifi.get('motion_level', 0)
        hrv = wearable.get('hrv')
        eda = wearable.get('eda', 2.0)

        if br is not None and hrv is not None:
            inputs.append([float(br), float(motion)])
            targets.append([
                max(HRV_MIN, min(HRV_MAX, float(hrv))),
                max(EDA_MIN, min(EDA_MAX, float(eda))),
            ])

    if len(inputs) < 10:
        print(f"  Only {len(inputs)} valid pairs — need at least 10 for training.")
        return

    print(f"\n  Training on {len(inputs)} real paired readings...")

    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(targets, dtype=torch.float32)

    model = WiFiCalibrationModel()

    # Try loading existing model as starting point
    if os.path.exists(output):
        try:
            model.load_state_dict(torch.load(output, weights_only=True))
            print(f"  Fine-tuning from existing checkpoint: {output}")
        except Exception:
            print(f"  Training from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(200):
        pred = model(X)
        loss = criterion(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/200, loss={loss.item():.4f}")

    model.eval()

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    torch.save(model.state_dict(), output)

    # Validation
    with torch.no_grad():
        preds = model(X)
        hrv_mae = (preds[:, 0] - Y[:, 0]).abs().mean().item()
        eda_mae = (preds[:, 1] - Y[:, 1]).abs().mean().item()

    print(f"\n  Training complete:")
    print(f"    HRV MAE: {hrv_mae:.1f} ms")
    print(f"    EDA MAE: {eda_mae:.1f} µS")
    print(f"    Checkpoint: {output}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Collect paired RuView + wearable data for calibration'
    )
    subparsers = parser.add_subparsers(dest='command')

    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Run a data collection session')
    collect_parser.add_argument('--duration', type=float, default=30,
                                help='Session duration in minutes (default: 30)')
    collect_parser.add_argument('--interval', type=float, default=5,
                                help='Poll interval in seconds (default: 5)')
    collect_parser.add_argument('--output', default=None,
                                help='Output JSON path (default: auto-named)')
    collect_parser.add_argument('--ruview-url', default=None,
                                help=f'RuView URL (default: {DEFAULT_RUVIEW_URL})')
    collect_parser.add_argument('--healthkit-url', default=None,
                                help=f'HealthKit API URL (default: {DEFAULT_HEALTHKIT_URL})')
    collect_parser.add_argument('--emotion-api-url', default=None,
                                help=f'Emotion API URL (default: {DEFAULT_EMOTION_API_URL})')
    collect_parser.add_argument('--feed', action='store_true',
                                help='Feed collected data to calibration buffer after session')
    collect_parser.add_argument('--profile', default=None,
                                help='Calibration profile ID (default: auto)')

    # Retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Retrain model from collected data')
    retrain_parser.add_argument('--data-dir', default='calibration_data',
                                help='Directory with session JSON files')
    retrain_parser.add_argument('--output', default='checkpoints/ruview_calibration.pt',
                                help='Output checkpoint path')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check RuView + wearable connectivity')
    status_parser.add_argument('--ruview-url', default=None)
    status_parser.add_argument('--healthkit-url', default=None)
    status_parser.add_argument('--emotion-api-url', default=None)

    args = parser.parse_args()

    if args.command == 'collect':
        output = args.output or f"calibration_data/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        collector = CalibrationCollector(
            ruview_url=args.ruview_url,
            healthkit_url=args.healthkit_url,
            emotion_api_url=args.emotion_api_url,
            poll_interval=args.interval,
        )

        pairs = collector.collect_session(
            duration_minutes=args.duration,
            output_path=output,
        )

        if args.feed and pairs:
            print("\n  Feeding to calibration buffer...")
            collector.feed_to_calibration_buffer(profile_id=args.profile)

    elif args.command == 'retrain':
        retrain_from_collected_data(
            data_dir=args.data_dir,
            output=args.output,
        )

    elif args.command == 'status':
        collector = CalibrationCollector(
            ruview_url=args.ruview_url,
            healthkit_url=args.healthkit_url,
            emotion_api_url=args.emotion_api_url,
        )

        print(f"\nChecking connectivity...")

        wifi = collector._get_ruview_reading()
        print(f"  RuView: {'CONNECTED' if wifi else 'UNAVAILABLE'}")
        if wifi:
            print(f"    HR={wifi.get('heart_rate')}, BR={wifi.get('breathing_rate')}, "
                  f"Confidence={wifi.get('confidence')}")

        wearable = collector._get_healthkit_reading()
        print(f"  Wearable: {'CONNECTED' if wearable else 'UNAVAILABLE'}")
        if wearable:
            print(f"    HR={wearable.get('heart_rate')}, HRV={wearable.get('hrv')}, "
                  f"Source={wearable.get('source')}")

        if wifi and wearable:
            print(f"\n  Both sources available. Ready to collect paired data.")
        else:
            print(f"\n  Both sources must be available for calibration collection.")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python calibration_collector.py status")
        print("  python calibration_collector.py collect --duration 15")
        print("  python calibration_collector.py collect --duration 30 --feed")
        print("  python calibration_collector.py retrain --data-dir calibration_data/")
