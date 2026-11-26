"""Executive End-to-End Demo: Track B (Orbit Edge AI + 3-Tier Aerial) -> Track A (ClaimPulse)."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from rich.console import Console
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from track_b.aerial_dispatch import run_aerial_verification
from track_b.bridge import ingest_track_b_into_track_a
from track_b.edge_inference import run_edge_satellite_inference
from track_b.schema import TrackBMergedTelemetry


def execute_scenario(
    *,
    scenario_name: str,
    satellite_id: str,
    orbit_pass_type: str,
    bounding_box: list[float],
    detected_peril: str,
    peak_hazard_intensity: float,
    cloud_obstruction_pct: float,
    district: str,
) -> TrackBMergedTelemetry:
    """Run full Track B (Edge AI -> 3-Tier Aerial) + Track A (Financial & Decision) pipeline."""
    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    edge_res = run_edge_satellite_inference(
        satellite_id=satellite_id,
        orbit_pass_type=orbit_pass_type,  # type: ignore[arg-type]
        scene_timestamp_utc=ts_utc,
        bounding_box=bounding_box,
        detected_peril=detected_peril,  # type: ignore[arg-type]
        peak_hazard_intensity=peak_hazard_intensity,
        cloud_obstruction_pct=cloud_obstruction_pct,
    )
    aerial_rep = run_aerial_verification(edge_res.alert)
    track_a_res = ingest_track_b_into_track_a(
        aerial_rep.verified_alert,
        aerial_report=aerial_rep,
        title=f"{scenario_name} ({satellite_id} {detected_peril.title()})",
        country="India",
        state="Odisha",
        district=district,
    )

    return TrackBMergedTelemetry(
        scenario_name=scenario_name,
        edge_alert=aerial_rep.verified_alert,
        benchmark=edge_res.benchmark,
        aerial_verification=aerial_rep.model_dump(),
        track_a_integration=track_a_res.model_dump(),
    )


def _render_ascii_summary(results: list[TrackBMergedTelemetry], out_path: Path) -> None:
    line = "=" * 96
    sub = "-" * 96
    print(f"\n{line}")
    print("  CLAIMPULSE MERGED PIPELINE | TRACK B (ORBIT EDGE AI + HAPS/DRONE) -> TRACK A (FINANCE)")
    print(f"{line}")

    headers = [
        ("Scenario", 26),
        ("Sat / Peril", 16),
        ("Peak", 8),
        ("Payload", 10),
        ("Priority", 12),
        ("HAPS/Drone", 11),
        ("Action", 14),
    ]
    hdr_row = " | ".join(f"{h:<{w}}" for h, w in headers)
    print(hdr_row)
    print(sub)

    for res in results:
        a = res.edge_alert
        av = res.aerial_verification
        haps_on = av["tier2_haps"]["dispatched"]
        drone_on = av["tier3_drone"]["dispatched"]
        aerial_str = "ACTIVE" if (haps_on and drone_on) else "IDLE"
        row = [
            (res.scenario_name[:25], 26),
            (f"{a.satellite_id}/{a.detected_peril}", 16),
            (f"{a.peak_hazard_intensity:.2f}", 8),
            (f"{a.insight_payload_bytes} B", 10),
            (a.downlink_priority, 12),
            (aerial_str, 11),
            (a.recommended_action_code, 14),
        ]
        print(" | ".join(f"{val:<{w}}" for val, w in row))

    print(sub)
    for res in results:
        a = res.edge_alert
        b = res.benchmark
        av = res.aerial_verification
        ta = res.track_a_integration
        t1 = av["tier1_satellite_gnss"]
        t2 = av["tier2_haps"]
        t3 = av["tier3_drone"]
        print(f"\n[{res.scenario_name}]")
        print(
            f"  Edge Inference : {b.raw_scene_mb:.0f} MB raw -> {a.insight_payload_bytes} B vector JSON "
            f"({b.bandwidth_reduction_pct:.5f}% compression) in {b.inference_latency_ms:.2f} ms | Model: {a.model_version_hash}"
        )
        print(
            f"  Tier 1 LEO GNSS: {t1['standard_gps_error_m']}m -> {t1['leo_augmented_error_m']}m "
            f"({t1['precision_gain_factor']}x precision gain)"
        )
        print(
            f"  Tier 2 HAPS    : dispatched={t2['dispatched']} | alt={t2['altitude_km']} km | "
            f"ETA={t2['eta_minutes']} min | cloud_lock={t2['all_weather_stratospheric_lock']} | {t2['status']}"
        )
        print(
            f"  Tier 3 Drone   : dispatched={t3['dispatched']} | ETA={t3['eta_minutes']} min | "
            f"conf {t3['confidence_before']:.2f} -> {t3['confidence_after']:.2f} | "
            f"assets={a.critical_assets_intercepted}, corridors={a.transport_corridors_severed}"
        )
        print(
            f"  Track A Output : Expected Loss = ${ta['total_expected_loss_usd']:,.2f} "
            f"(INR {ta['total_expected_loss_inr']:,.2f}) | "
            f"Provisional Payout = ${ta['total_provisional_payout_usd']:,.2f} | "
            f"Fast-Tracked={ta['claims_fast_tracked_count']} | Action={ta['executed_action_code']}"
        )

    print(f"\n{line}")
    print(f"  Exported validated merged telemetry JSON -> {out_path}")
    print(f"{line}\n")


def _render_rich_summary(results: list[TrackBMergedTelemetry], out_path: Path) -> None:
    console = Console()
    table = Table(title="ClaimPulse Merged Pipeline (Track B Orbit -> Track A Decision Engine)")
    table.add_column("Scenario", style="bold cyan")
    table.add_column("Satellite / Peril")
    table.add_column("Peak", justify="right")
    table.add_column("Payload", justify="right")
    table.add_column("BW Saved", justify="right")
    table.add_column("Priority")
    table.add_column("HAPS/Drone")
    table.add_column("Action", style="bold green")
    table.add_column("Track A Loss (USD)", justify="right")

    for res in results:
        a = res.edge_alert
        b = res.benchmark
        av = res.aerial_verification
        ta = res.track_a_integration
        aerial_str = "ACTIVE" if av["escalation_triggered"] else "IDLE"
        table.add_row(
            res.scenario_name,
            f"{a.satellite_id} / {a.detected_peril}",
            f"{a.peak_hazard_intensity:.2f}",
            f"{a.insight_payload_bytes} B",
            f"{b.bandwidth_reduction_pct:.4f}%",
            a.downlink_priority,
            aerial_str,
            a.recommended_action_code,
            f"${ta['total_expected_loss_usd']:,.2f}",
        )
    console.print(table)
    console.print(f"[bold]Exported validated merged telemetry JSON:[/bold] {out_path}")


def run_demo_scenarios(output_path: Path | None = None) -> tuple[list[TrackBMergedTelemetry], Path]:
    """Execute Scenario A (Critical EOS-04 Flood) and Scenario B (Routine RISAT-1A Waterlogging)."""
    scenario_a = execute_scenario(
        scenario_name="Scenario A: Critical SAR Flood",
        satellite_id="EOS-04",
        orbit_pass_type="descending",
        bounding_box=[85.72, 19.76, 86.38, 20.24],
        detected_peril="flood",
        peak_hazard_intensity=2.15,
        cloud_obstruction_pct=0.0,
        district="Puri & Kendrapara",
    )

    scenario_b = execute_scenario(
        scenario_name="Scenario B: Routine Waterlogging",
        satellite_id="RISAT-1A",
        orbit_pass_type="ascending",
        bounding_box=[85.94, 19.93, 86.08, 20.03],
        detected_peril="flood",
        peak_hazard_intensity=0.40,
        cloud_obstruction_pct=0.0,
        district="Satyabadi Basin",
    )

    results = [scenario_a, scenario_b]
    target_out = output_path or (_ROOT / "outputs" / "merged_disaster_alert.json")
    target_out.parent.mkdir(parents=True, exist_ok=True)

    export_doc: dict[str, Any] = {
        "schema_version": "track-b-v1.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "primary_scenario": scenario_a.model_dump(),
        "scenarios": [r.model_dump() for r in results],
    }
    target_out.write_text(json.dumps(export_doc, indent=2), encoding="utf-8")
    return results, target_out


def main() -> None:
    results, out_path = run_demo_scenarios()
    if _RICH_AVAILABLE:
        _render_rich_summary(results, out_path)
    else:
        _render_ascii_summary(results, out_path)


if __name__ == "__main__":
    main()
