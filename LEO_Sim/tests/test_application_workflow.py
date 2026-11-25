"""Verify all application insight payloads use the shared queue and ground receipt path."""

import json
import sys
from pathlib import Path

LEO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LEO_ROOT.parent
if str(LEO_ROOT) not in sys.path:
    sys.path.insert(0, str(LEO_ROOT))

from onboard_processor.application_runner import run
from src.communication import build_json_payload
from src.transmission_queue import TransmissionQueue


def test_each_application_payload_dispatches_to_ground_station():
    eco_image = PROJECT_ROOT / "Eco_Act" / "Dataset" / "monthly_10yr" / "tumkur_monthly_2024_12.png"
    if not eco_image.exists():
        eco_image = next((PROJECT_ROOT / "Eco_Act" / "Dataset" / "monthly_10yr").glob("*.png"))
    inputs = [
        ("Eco_Act", {"image_path": str(eco_image)}),
        ("UrbanPulse", {"lat": 13.34, "lon": 77.10}),
        ("DisasterRisk", {"preset_id": "odisha-flood-2026"}),
    ]
    queue = TransmissionQueue()
    expected = {}
    for app, kwargs in inputs:
        result = run(app, **kwargs)
        payload = build_json_payload(result)
        expected[app] = json.loads(payload)
        queue.enqueue(payload, "HIGH", label=app, metadata={"application": app})

    dispatched = queue.dispatch(visible=True, bandwidth_mbps=2.0, contact_seconds=60)
    assert not dispatched["deferred"]
    received = {item.label: json.loads(item.payload.decode("utf-8")) for item in dispatched["sent"]}
    assert set(received) == {app for app, _ in inputs}
    for app in received:
        assert received[app] == expected[app]
        assert received[app]["application"] == app

