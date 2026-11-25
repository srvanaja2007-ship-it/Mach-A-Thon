"""
Communication simulation module.

Calculates transmission times for:
  A. Raw satellite image (full file)
  B. Compact JSON insight payload

Uses ACTUAL file sizes (os.path.getsize) and actual JSON byte length.
No values are fabricated – sizes come from real files and real serialisation.
"""

import json
import os
import time
from typing import Any, Dict, Generator

DEFAULT_BANDWIDTH_MBPS = 2.0   # Mbps  (configurable from dashboard)


# ── Payload serialisation ─────────────────────────────────────────────────────

def build_json_payload(metrics: Dict[str, Any]) -> bytes:
    """
    Serialise the onboard-processor metrics dict into a compact JSON payload.
    Strips the raw 'image_path' key (not transmitted).
    Returns UTF-8 encoded bytes (actual transmitted content).
    """
    payload = {k: v for k, v in metrics.items()
               if k not in ("image_path", "status")}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ── Size helpers ──────────────────────────────────────────────────────────────

def image_size_bytes(image_path: str) -> int:
    """Return actual on-disk file size of the VIIRS image in bytes."""
    return os.path.getsize(image_path)


def payload_size_bytes(payload: bytes) -> int:
    """Return byte length of the JSON payload."""
    return len(payload)


# ── Transmission time ─────────────────────────────────────────────────────────

def transmission_time_seconds(size_bytes: int, bandwidth_mbps: float) -> float:
    """
    Compute transmission time:
      t = (size_bytes * 8) / (bandwidth_mbps * 1e6)
    """
    if bandwidth_mbps <= 0:
        raise ValueError("Bandwidth must be > 0 Mbps")
    bits = size_bytes * 8
    return bits / (bandwidth_mbps * 1_000_000)


# ── Main comparison function ──────────────────────────────────────────────────

def compare_transmission(image_path: str,
                          metrics: Dict[str, Any],
                          bandwidth_mbps: float = DEFAULT_BANDWIDTH_MBPS) -> Dict[str, Any]:
    """
    Compare raw-image transmission vs JSON-insight transmission.

    Returns a dict with:
      - raw_image_size_bytes
      - json_payload_size_bytes
      - json_payload_str           (pretty-printed for display)
      - bandwidth_mbps
      - raw_transmission_time_s
      - json_transmission_time_s
      - time_saved_s
      - data_reduction_pct
      - size_reduction_ratio       (raw / json)
      - note                       ('[SIMULATED] Communication scenarios only')
    """
    raw_bytes  = image_size_bytes(image_path)
    payload    = build_json_payload(metrics)
    json_bytes = payload_size_bytes(payload)

    raw_time  = transmission_time_seconds(raw_bytes,  bandwidth_mbps)
    json_time = transmission_time_seconds(json_bytes, bandwidth_mbps)
    time_saved = raw_time - json_time

    reduction_pct = ((raw_bytes - json_bytes) / raw_bytes) * 100.0 if raw_bytes > 0 else 0.0
    size_ratio    = raw_bytes / json_bytes if json_bytes > 0 else float("inf")

    # Pretty JSON for dashboard display
    payload_dict = json.loads(payload.decode("utf-8"))
    pretty_json  = json.dumps(payload_dict, indent=2, ensure_ascii=False)

    return {
        "raw_image_size_bytes":      raw_bytes,
        "raw_image_size_kb":         round(raw_bytes / 1024, 2),
        "json_payload_size_bytes":   json_bytes,
        "json_payload_size_kb":      round(json_bytes / 1024, 3),
        "json_payload_str":          pretty_json,
        "bandwidth_mbps":            bandwidth_mbps,
        "raw_transmission_time_s":   round(raw_time, 4),
        "json_transmission_time_s":  round(json_time, 6),
        "time_saved_s":              round(time_saved, 4),
        "data_reduction_pct":        round(reduction_pct, 2),
        "size_reduction_ratio":      round(size_ratio, 1),
        "note": "[SIMULATED] Transmission scenarios – communication simulation only",
    }


# ── Progress generator ────────────────────────────────────────────────────────

def transmission_progress(total_bytes: int,
                           bandwidth_mbps: float,
                           label: str = "Transmitting",
                           steps: int = 20) -> Generator[dict, None, None]:
    """
    Yields progress dicts {pct, bytes_sent, label} simulating transmission.
    Does NOT actually send data – purely a UI simulation.
    Total simulated wall-clock time matches the calculated transmission time.
    """
    total_time_s = transmission_time_seconds(total_bytes, bandwidth_mbps)
    sleep_per_step = total_time_s / steps

    for i in range(1, steps + 1):
        time.sleep(sleep_per_step)
        pct = int((i / steps) * 100)
        bytes_sent = int(total_bytes * i / steps)
        yield {"pct": pct, "bytes_sent": bytes_sent, "label": label}
