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


def test_urbanpulse_adapter_rejects_synthetic_cached_layers():
    try:
        run("UrbanPulse", lat=13.34, lon=77.10)
    except RuntimeError as exc:
        assert "synthetic layers" in str(exc)
        assert "No model was run" in str(exc)
    else:
        raise AssertionError("Synthetic cache must not produce an UrbanPulse inference")


def test_disasterrisk_adapter_rejects_demo_scenario():
    try:
        run("DisasterRisk", preset_id="odisha-flood-2026")
    except RuntimeError as exc:
        assert "is_demo=True" in str(exc)
        assert "No hazard" in str(exc)
    else:
        raise AssertionError("Demo preset must not produce a DisasterRisk inference")

