# LEO_Sim technical audit

## Current verified state

- `onboard_processor/application_runner.py` validates each application's
  required arguments and finite WGS84 coordinates.
- Eco_Act executes its original analysis pipeline against the selected VIIRS
  image and reports measured file bytes and processing time.
- UrbanPulse's existing fetch pipeline silently falls back to synthetic layers.
  The onboard adapter now checks source metadata before calling its existing
  ML pipeline and rejects missing or synthetic layers.
- DisasterRisk's preset connector calls `build_scenario_raw_data`, whose
  presets are marked `is_demo=True`. The onboard adapter rejects that input
  before running hazard, exposure, loss, or decision calculations.
- JSON payloads include the application identifier. The queue orders HIGH,
  MEDIUM, LOW; it retains items when the link is unavailable or a complete
  payload does not fit the contact window.
- Link visibility uses ECEF position and ground-station look angles with a
  five-degree elevation mask. The UI dispatch path uses the remaining pass
  duration sampled against that mask, not a fixed contact allowance.

## Orbital/link model limits

The simulator uses circular two-body propagation with an approximate RAAN
drift and spherical Earth. Contact duration is sampled at one-second steps.
The model excludes J2, drag, attitude, link budget losses, coding overhead,
weather, and actual modem behavior. Transfer duration is a calculated
simulation value, not a measured radio transmission.

## Remaining source blockers

1. **UrbanPulse:** all checked-in location caches use `synthetic_physics` for
   all required layers. An isolated live refresh fetched Sentinel-2 scenes
   from 2021 and 2026, Sentinel-1 RTC, and a WorldCover electricity proxy. The
   required OSM road/water query to Overpass did not complete. The full model
   was not run; no partial-source predictions are emitted.
2. **DisasterRisk:** the adapter only exposes the scenario connector. The
   connector returns demo-generated event data, so there is no compatible
   observation acquisition path from which to generate a real risk result.

See [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) for measured Eco_Act bytes,
processing time, and simulated transfer calculations.
