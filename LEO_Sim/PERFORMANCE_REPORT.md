# Phase 2 performance report

Measured with `tumkur_monthly_2024_12.png` using the existing Eco_Act
processor and a 2.0 Mbps simulated link.

| Metric | Traditional transmission | Onboard AI transmission |
|---|---:|---:|
| Data transmitted | 172,711 bytes image | 4,597 bytes UTF-8 JSON |
| Processing location | Ground station scenario | Satellite simulator |
| Transmission time | 0.690844 s | 0.018388 s |
| Bandwidth requirement | 2.0 Mbps | 2.0 Mbps |

Derived results:

- Data reduction: **97.34%**
- Size ratio: **37.6×**
- Link time saved: **0.6725 s**
- Measured Eco_Act processing time: **1.0909 s**

Transmission times are deterministic calculations from actual serialized byte
counts; processing time is a wall-clock measurement on the development
machine. End-to-end latency additionally depends on contact availability,
queue delay, radio overhead, and model execution hardware, so those values are
not claimed as flight results.

The adaptive queue now accounts for contact visibility and a finite contact
budget. High-priority items are selected before medium and low priority items;
items that do not fit remain queued for a later pass.
