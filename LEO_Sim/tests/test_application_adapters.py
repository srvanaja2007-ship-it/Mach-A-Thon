"""Independent integration checks for each onboard application's real pipeline."""

import sys
from pathlib import Path

LEO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LEO_ROOT.parent
if str(LEO_ROOT) not in sys.path:
    sys.path.insert(0, str(LEO_ROOT))

from onboard_processor.application_runner import run


def test_eco_act_adapter_runs_existing_processor():
    image = PROJECT_ROOT / "Eco_Act" / "Dataset" / "monthly_10yr" / "tumkur_monthly_2024_12.png"
    if not image.exists():
        image = next((PROJECT_ROOT / "Eco_Act" / "Dataset" / "monthly_10yr").glob("*.png"))
    result = run("Eco_Act", image_path=str(image))
    assert result["status"] == "SUCCESS"
    assert "brightness" in result and "gddp" in result


def test_urbanpulse_adapter_runs_existing_model_on_real_cached_layers():
    result = run("UrbanPulse", lat=13.34, lon=77.10)
    assert result["status"] == "SUCCESS"
    assert result["data_sources"]
    assert not any("synthetic" in str(source).lower()
                   for source in result["data_sources"].values())
    assert "zone_counts" in result["insights"]
    assert "sprawl_risk_mean" in result["insights"]


def test_disasterrisk_adapter_runs_existing_scenario_pipeline():
    result = run("DisasterRisk", preset_id="odisha-flood-2026")
    assert result["status"] == "SUCCESS"
    assert result["data_status"] == "DEMO_SCENARIO"
    assert result["hazard"]["affected_area_km2"] > 0
    assert result["loss"]["total_expected_loss_inr"] >= 0
    assert "recommended_action_headline" in result["decisions"]

