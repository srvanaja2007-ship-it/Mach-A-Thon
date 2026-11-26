# Verified LEO_Sim measurements

Run with the project `.venv` (Python 3.14, NumPy 2.5.5) against
`Eco_Act/Dataset/monthly_10yr/tumkur_monthly_2024_12.png`.

| Measurement | Result |
|---|---:|
| VIIRS image file | 172,711 bytes |
| Compact UTF-8 JSON payload, including application metadata | 5,145 bytes |
| Data reduction | 97.02% |
| Eco_Act processing time | 1.1761 s (measured laptop wall time) |
| Raw-image transfer at 2 Mbps | 0.690844 s |
| Insight transfer at 2 Mbps | 0.020580 s |
| Simulated orbit elevation at t=0 | 11.227 degrees |
| Sampled remaining contact window at t=0 | 474 s at a 1 s step |
| Queue result | 1 payload transmitted and removed from pending |

The queue check used the serialized result above. Transmission duration is
computed from payload bits and configured bandwidth. Orbit and link results
are simulated; the processing time is a laptop measurement, not flight
hardware performance. Contact-window duration is sampled at one-second
intervals and is therefore approximate.

## Other applications

- **UrbanPulse:** no model inference is reported for the checked-in caches;
  all four identify Sentinel-2, Sentinel-1, electricity, and OSM layers as
  `synthetic_physics`. A live refresh fetched real Sentinel-2 2021/2026,
  Sentinel-1 RTC, and WorldCover electricity inputs. The Overpass/OSM request
  did not complete, so the required road/water layer was unavailable and the
  model was not run on partial or fallback data.
- **DisasterRisk:** no model inference is reported. Its preset connector
  returns `is_demo=True` and scenario-built zones, not acquired observations.

There are no valid raw-input sizes, JSON insight sizes, reduction percentages,
or transmission durations to report for these two applications because no
eligible observation reached inference or payload generation.
