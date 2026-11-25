# LEO simulator technical audit

## Scope

The existing entry point is [app.py](./app.py). It calls the circular-orbit
model in `src/orbit.py`, the Eco_Act adapters through
`onboard_processor/processor.py`, and byte-based transmission calculations in
`src/communication.py`.

## Findings and priority

1. **High — visibility data was incomplete.** The app showed only a boolean
   line-of-sight result. Distance, elevation, azimuth, elevation-mask control,
   and next contact were not exposed. These are now calculated from ECEF
   coordinates and the ground-station ENU frame.
2. **High — adaptive transmission was absent.** Payloads had no queue or
   priority scheduling. `src/transmission_queue.py` now orders HIGH, MEDIUM,
   and LOW payloads and defers them when no contact or insufficient contact
   budget exists.
3. **Medium — Urbanization was a separate application.** Its existing
   `run_ml_pipeline(data)` contract consumes aligned geospatial feature layers,
   not a standalone image. `urbanization_adapter` reports that exact boundary
   and requirements instead of fabricating a prediction.
4. **Medium — orbit API was presentation-oriented.** The existing model used
   a spherical ground-track projection and had no reusable ECI/ECEF/look-angle
   API. The original UI remains, with the physically meaningful APIs added.
5. **Low — documentation drift.** The README listed directories that are not
   present; this audit records the actual architecture and assumptions.

## Verified existing workflow

Dataset selection or upload sets `session_state["image_path"]`; pressing Run
invokes the Eco_Act adapter pipeline on that exact path. Payload bytes are
serialized with UTF-8 JSON, and raw size comes from `os.path.getsize`.

## Equations and assumptions

- `a = R_e + h`
- `T = 2π √(a³ / μ)`
- `v = √(μ / a)`
- ECI is rotated into ECEF by Earth’s sidereal rotation rate.
- Ground-station coordinates use a spherical Earth radius of 6371 km.
- Contact is available when calculated elevation is at least the configured
  elevation mask.
- This remains a simulated two-body circular orbit. It omits J2 perturbations,
  drag, ephemeris errors, attitude, link losses, coding overhead, and weather.

## Metric validation

Raw and JSON sizes are actual serialized byte counts. Transmission time is
calculated as `bytes * 8 / (Mbps * 1,000,000)`. Reduction is
`(raw - JSON) / raw * 100`; no 97.3% or 36× value is hardcoded.
