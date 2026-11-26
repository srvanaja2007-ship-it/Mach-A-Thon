"""Verify genuine insights use the visibility-gated transmission queue."""

import json
import sys
from pathlib import Path

LEO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LEO_ROOT.parent
if str(LEO_ROOT) not in sys.path:
    sys.path.insert(0, str(LEO_ROOT))

from onboard_processor.application_runner import run
from src.orbit import GROUND_STATION, Satellite
from src.transmission_queue import TransmissionQueue


def test_eco_act_payload_is_received_only_after_visible_transmission():
    image = PROJECT_ROOT / "Eco_Act" / "Dataset" / "monthly_10yr" / "tumkur_monthly_2024_12.png"
    result = run("Eco_Act", image_path=str(image))
    payload = json.dumps(
        {"application": "Eco_Act", **{k: v for k, v in result.items() if k not in {"image_path", "status"}}},
        separators=(",", ":"),
    ).encode("utf-8")
    queue = TransmissionQueue()
    queue.enqueue(payload, "MEDIUM", "Eco_Act", {"application": "Eco_Act"})

    no_contact = queue.dispatch(False, 2.0, 60.0)
    assert not no_contact["sent"]
    assert len(queue.pending()) == 1

    satellite = Satellite()
    gs = GROUND_STATION
    contact_end = next(
        second for second in range(1, int(satellite.params.period_seconds))
        if not satellite.visibility_at(second, gs["lat"], gs["lon"])["visible"]
    )
    visible = queue.dispatch(True, 2.0, contact_end - 1)
    assert not visible["deferred"]
    assert len(visible["sent"]) == 1
    assert not queue.pending()
    assert visible["sent"][0].metadata["simulated_receive_offset_s"] > 0
    assert json.loads(visible["sent"][0].payload)["application"] == "Eco_Act"
