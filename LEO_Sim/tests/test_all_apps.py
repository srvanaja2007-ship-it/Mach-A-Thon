"""Source-gated integration checks for the three existing Part A applications."""

import json
import sys
import unittest
from pathlib import Path

LEO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LEO_ROOT.parent
if str(LEO_ROOT) not in sys.path:
    sys.path.insert(0, str(LEO_ROOT))

from onboard_processor.application_runner import run
from src.orbit import GROUND_STATION, OrbitalParams, Satellite
from src.transmission_queue import TransmissionQueue


class TestApplicationWorkflow(unittest.TestCase):
  def test_eco_act_runs_original_pipeline_and_transmits_actual_payload(self):
    image = PROJECT_ROOT / "Eco_Act" / "Dataset" / "monthly_10yr" / "tumkur_monthly_2024_12.png"
    result = run("Eco_Act", image_path=str(image))
    payload = json.dumps(
        {"application": "Eco_Act", **{k: v for k, v in result.items() if k not in {"image_path", "status"}}},
        separators=(",", ":"),
    ).encode("utf-8")

    self.assertEqual(result["status"], "SUCCESS")
    self.assertEqual(result["image_size_bytes"], image.stat().st_size)
    self.assertIsNotNone(result["economic_activity_index"])
    self.assertGreater(len(payload), 0)

    sat = Satellite(OrbitalParams())
    gs = GROUND_STATION
    queue = TransmissionQueue()
    queue.enqueue(payload, "MEDIUM", "Eco_Act", {"application": "Eco_Act"})
    self.assertEqual(queue.dispatch(False, 2.0, 120.0)["sent"], [])

    contact = 0.0
    while contact < sat.params.period_seconds and sat.visibility_at(contact, gs["lat"], gs["lon"])["visible"]:
        contact += 1.0
    contact = max(0.0, contact - 1.0)
    sent = queue.dispatch(True, 2.0, contact)["sent"]
    self.assertEqual(len(sent), 1)
    self.assertAlmostEqual(sent[0].metadata["simulated_transmit_duration_s"], len(payload) * 8 / 2_000_000)
    self.assertEqual(json.loads(sent[0].payload)["application"], "Eco_Act")


  def test_urbanpulse_refuses_synthetic_cached_layers(self):
    with self.assertRaisesRegex(RuntimeError, "synthetic layers.*No model was run"):
        run("UrbanPulse", lat=13.34, lon=77.10)


  def test_disasterrisk_refuses_demo_preset_data(self):
    with self.assertRaisesRegex(RuntimeError, "is_demo=True.*No hazard"):
        run("DisasterRisk", preset_id="odisha-flood-2026")


  def test_application_inputs_are_validated(self):
    for args in ({}, {"lat": 100, "lon": 77}, {"lat": float("nan"), "lon": 77}):
        with self.assertRaises(ValueError):
            run("UrbanPulse", **args)

  def test_priority_order_and_contact_capacity_retain_deferred_item(self):
    queue = TransmissionQueue()
    queue.enqueue(b"L", "LOW", "low")
    queue.enqueue(b"M", "MEDIUM", "medium")
    queue.enqueue(b"H", "HIGH", "high")
    result = queue.dispatch(True, bandwidth_mbps=2.0, contact_seconds=0.000008)
    self.assertEqual([item.label for item in result["sent"]], ["high", "medium"])
    self.assertEqual([item.label for item in result["deferred"]], ["low"])
    self.assertEqual([item.label for item in queue.pending()], ["low"])


if __name__ == "__main__":
    unittest.main()
