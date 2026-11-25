import json
import math
from pathlib import Path

from LEO_Sim.src.communication import compare_transmission
from LEO_Sim.src.orbit import GROUND_STATION, OrbitalParams, Satellite
from LEO_Sim.src.transmission_queue import TransmissionQueue
from LEO_Sim.onboard_processor.processor import run


ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "Eco_Act" / "Dataset" / "monthly_10yr" / "tumkur_monthly_2024_12.png"


def test_orbit_mechanics_and_look_angles():
    sat = Satellite(OrbitalParams())
    assert 94.0 < sat.params.period_minutes < 95.0
    assert math.isclose(float(sat.position_eci_km(0.0) @ sat.position_eci_km(0.0)) ** 0.5,
                        sat.params.semi_major_axis_km, rel_tol=1e-9)
    view = sat.visibility_at(0.0, GROUND_STATION["lat"], GROUND_STATION["lon"])
    assert view["distance_km"] > 0
    assert 0 <= view["azimuth_deg"] < 360


def test_priority_queue_defers_when_not_visible():
    queue = TransmissionQueue()
    low = queue.enqueue(b"low", "LOW", "routine")
    high = queue.enqueue(b"high", "HIGH", "alert")
    result = queue.dispatch(False, 2.0, 10.0)
    assert result["sent"] == []
    assert [item.label for item in result["deferred"]] == ["alert", "routine"]
    result = queue.dispatch(True, 2.0, 0.00002)
    assert len(result["sent"]) == 1
    assert result["sent"][0] is high


def test_real_image_changes_payload_metrics_and_sizes():
    metrics = run(str(IMAGE))
    comm = compare_transmission(str(IMAGE), metrics, 2.0)
    assert metrics["image_filename"] == IMAGE.name
    assert comm["raw_image_size_bytes"] == IMAGE.stat().st_size
    assert comm["json_payload_size_bytes"] == len(json.dumps(
        {k: v for k, v in metrics.items() if k not in ("image_path", "status")},
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))
    assert comm["data_reduction_pct"] == round(
        (comm["raw_image_size_bytes"] - comm["json_payload_size_bytes"])
        / comm["raw_image_size_bytes"] * 100, 2
    )
