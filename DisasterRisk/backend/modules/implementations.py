from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from backend.core.interfaces import (
    DataConnector,
    DecisionWorkflowModule,
    EconomicModel,
    ExposureModule,
    HazardModule,
)
from backend.core.schemas import (
    AuditRecord,
    ClaimRecommendation,
    DatasetProvenance,
    DecisionBundle,
    DisasterType,
    EventRequest,
    ExposureSummary,
    FormulaAssumption,
    HazardMetrics,
    InsuranceClaimDecision,
    LenderCollateralAlert,
    LossEstimate,
    PriorityZone,
    SectorLoss,
    SeverityLevel,
    StakeholderParameter,
    StakeholderPrediction,
    TimeSeriesLossPoint,
)
from backend.scenarios.data_builder import build_scenario_raw_data, build_zone_polygon_coords


INR_SCALE = 84.0


def fmt_inr(amount: float, ascii_only: bool = False) -> str:
    prefix = "Rs. " if ascii_only else "₹"
    if abs(amount) >= 1e7:
        return f"{prefix}{amount / 1e7:,.2f} Cr"
    if abs(amount) >= 1e5:
        return f"{prefix}{amount / 1e5:,.2f} Lakh"
    return f"{prefix}{amount:,.0f}"


def _classify_severity(disaster_type: DisasterType, intensity: float) -> SeverityLevel:
    if disaster_type == "wildfire":
        if intensity >= 0.72:
            return "extreme"
        if intensity >= 0.50:
            return "high"
        if intensity >= 0.28:
            return "medium"
        if intensity >= 0.12:
            return "low"
        return "safe"
    if disaster_type == "cyclone":
        if intensity >= 1.60:
            return "extreme"
        if intensity >= 1.15:
            return "high"
        if intensity >= 0.70:
            return "medium"
        if intensity >= 0.30:
            return "low"
        return "safe"
    if intensity >= 2.20:
        return "extreme"
    if intensity >= 1.40:
        return "high"
    if intensity >= 0.70:
        return "medium"
    if intensity >= 0.20:
        return "low"
    return "safe"


def _normalize_intensity(disaster_type: DisasterType, raw_intensity: float) -> float:
    if disaster_type == "wildfire":
        return round(min(1.0, max(0.02, raw_intensity / 1.0)), 4)
    if disaster_type == "cyclone":
        return round(min(1.0, max(0.02, raw_intensity / 2.2)), 4)
    return round(min(1.0, max(0.02, raw_intensity / 3.6)), 4)


def _damage_multiplier(disaster_type: DisasterType, severity: SeverityLevel, sector: str) -> float:
    base_map = {
        "safe": 0.03,
        "low": 0.14,
        "medium": 0.28,
        "high": 0.44,
        "extreme": 0.62,
    }
    sector_adj = {
        "housing": 1.00,
        "agriculture": 1.12,
        "infrastructure": 0.92,
        "business": 1.08,
    }
    hazard_adj = {
        "flood": 1.00,
        "wildfire": 1.15,
        "cyclone": 1.06,
    }
    return round(base_map[severity] * sector_adj.get(sector, 1.0) * hazard_adj[disaster_type], 4)


class DemoSatelliteConnector(DataConnector):
    connector_id = "nasa-copernicus-demo-connector"
    provider = "NASA Disasters Portal / Copernicus Sentinel / OSM / WorldPop"

    def fetch(self, event_req: EventRequest) -> dict[str, Any]:
        return build_scenario_raw_data(event_req)


def build_raw_data_from_track_b(
    event_req: EventRequest,
    track_b_alert: dict[str, Any],
    aerial_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a verified 19-field Track B edge alert into Track A's connector payload."""
    base = build_scenario_raw_data(event_req)
    peril = track_b_alert["detected_peril"]
    peak_raw = float(track_b_alert["peak_hazard_intensity"])
    peak_for_track_a = round(peak_raw / 100.0, 3) if peril == "cyclone" and peak_raw > 10.0 else peak_raw

    zones = [dict(z) for z in base["zones"]]
    existing_max = max((z["base_intensity"] for z in zones), default=1.0) or 1.0
    intensity_ratio = peak_for_track_a / existing_max
    existing_area = sum(z["area_km2"] for z in zones) or 1.0
    target_area = max(0.5, float(track_b_alert["affected_area_km2"]))
    area_ratio = target_area / existing_area

    verified_conf = float(track_b_alert["inference_confidence"])
    for z in zones:
        z["base_intensity"] = round(max(0.05, z["base_intensity"] * intensity_ratio), 3)
        z["area_km2"] = round(max(0.1, z["area_km2"] * area_ratio), 2)
        z["confidence"] = round(min(0.99, max(0.50, (z["confidence"] * 0.3) + (verified_conf * 0.7))), 2)

    sat_id = track_b_alert["satellite_id"]
    ts_utc = track_b_alert["scene_timestamp_utc"]
    model_hash = track_b_alert["model_version_hash"]
    track_b_prov = DatasetProvenance(
        dataset_name=f"{sat_id} Onboard Edge AI ({model_hash})",
        provider="Track B LEO Edge Inference + 3-Tier Aerial Verification",
        spatial_resolution="10 m (INT8 Vector + 0.04m LEO-GNSS)",
        timestamp=ts_utc,
        license="Track B Telemetry Contract v1",
        confidence=verified_conf,
        is_demo=False,
        layer_id="hazard_extent",
    )
    provenance = {
        "hazard": [track_b_prov, *base["provenance"]["hazard"]],
        "exposure": list(base["provenance"]["exposure"]),
    }

    unit = "km/h (wind)" if peril == "cyclone" else ("dNBR" if peril == "wildfire" else "m (flood depth)")
    base.update(
        {
            "id": f"trackb-{sat_id.lower()}-{peril}-{ts_utc.replace(':', '').replace('-', '')}",
            "zones": zones,
            "max_intensity_label": f"{peak_raw:.2f} {unit} (Track B {sat_id})",
            "provenance": provenance,
            "is_demo": False,
            "track_b_alert": track_b_alert,
            "aerial_verification": aerial_verification or {},
        }
    )
    return base


class TrackBEdgeDataConnector(DataConnector):
    """Connector that ingests verified Track B satellite/HAPS/drone alerts into Track A."""

    connector_id = "track-b-orbit-edge-connector"
    provider = "ISRO EOS-04 / RISAT-1A Edge AI + Stratospheric HAPS + IP65 Drone"

    def __init__(
        self,
        track_b_alert: dict[str, Any] | None = None,
        aerial_verification: dict[str, Any] | None = None,
    ) -> None:
        self.track_b_alert = track_b_alert
        self.aerial_verification = aerial_verification

    def fetch(self, event_req: EventRequest) -> dict[str, Any]:
        if self.track_b_alert is None:
            return build_scenario_raw_data(event_req)
        return build_raw_data_from_track_b(
            event_req,
            self.track_b_alert,
            aerial_verification=self.aerial_verification,
        )


class BaseHazardModule(HazardModule):
    def __init__(self, disaster_type: DisasterType, module_name: str) -> None:
        self.disaster_type = disaster_type
        self.module_name = module_name

    def compute_damage(
        self, event_req: EventRequest, connector_payload: dict[str, Any]
    ) -> dict[str, Any]:
        mult = event_req.intensity_multiplier
        features: list[dict[str, Any]] = []
        severity_dist: dict[SeverityLevel, float] = {
            "safe": 0.0,
            "low": 0.0,
            "medium": 0.0,
            "high": 0.0,
            "extreme": 0.0,
        }
        intensities: list[float] = []
        total_affected_area = 0.0

        for idx, z in enumerate(connector_payload["zones"]):
            scaled_intensity = round(z["base_intensity"] * mult, 3)
            severity = _classify_severity(self.disaster_type, scaled_intensity)
            area_km2 = round(z["area_km2"] * (1.0 + (mult - 1.0) * 0.25 if severity != "safe" else 1.0), 2)
            severity_dist[severity] = round(severity_dist[severity] + area_km2, 2)
            if severity != "safe":
                total_affected_area += area_km2
            intensities.append(scaled_intensity)

            coords = build_zone_polygon_coords(z, idx)
            if self.disaster_type == "wildfire":
                pre_state = "Intact forest & canopy (NDVI 0.74)"
                post_state = f"Burn scar dNBR {scaled_intensity:.2f} ({severity.upper()})"
            elif self.disaster_type == "cyclone":
                pre_state = "Normal coastal wind (< 25 km/h) & dry shore"
                post_state = f"Surge/Wind index {scaled_intensity:.2f} ({severity.upper()})"
            else:
                pre_state = f"Dry floodplain (Elev {z['elevation_m']}m, 0.00m water)"
                post_state = f"Inundated {scaled_intensity:.2f}m depth ({severity.upper()})"

            features.append(
                {
                    "type": "Feature",
                    "id": z["id"],
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "zone_id": z["id"],
                        "name": z["name"],
                        "district": z["district"],
                        "severity": severity,
                        "hazard_intensity": scaled_intensity,
                        "normalized_intensity": _normalize_intensity(self.disaster_type, scaled_intensity),
                        "intensity_unit": connector_payload["intensity_unit"],
                        "elevation_m": z["elevation_m"],
                        "area_km2": area_km2,
                        "confidence": z["confidence"],
                        "vulnerability_score": z["vulnerability"],
                        "infrastructure_criticality": z["infra_crit"],
                        "center": z["center"],
                        "pre_state": pre_state,
                        "post_state": post_state,
                    },
                }
            )

        mean_int = round(sum(intensities) / max(1, len(intensities)), 2)
        metrics = HazardMetrics(
            affected_area_km2=round(total_affected_area, 2),
            max_intensity_label=connector_payload["max_intensity_label"]
            if abs(mult - 1.0) < 0.01
            else f"{round(max(intensities), 2)} {connector_payload['intensity_unit']} ({int(mult*100)}% scenario)",
            mean_intensity_value=mean_int,
            intensity_unit=connector_payload["intensity_unit"],
            severity_distribution_km2=severity_dist,
            datasets=connector_payload["provenance"]["hazard"],
            timeline_dates=connector_payload["timeline_dates"],
            imagery_comparison=connector_payload["imagery_comparison"],
        )
        return {
            "geojson": {"type": "FeatureCollection", "features": features},
            "metrics": metrics.model_dump(),
        }


class StandardExposureModule(ExposureModule):
    module_name = "osm-worldpop-fao-exposure-v1"

    def overlay_exposure(
        self,
        event_req: EventRequest,
        damage_result: dict[str, Any],
        connector_payload: dict[str, Any],
    ) -> dict[str, Any]:
        zone_lookup = {
            f["properties"]["zone_id"]: f["properties"]
            for f in damage_result["geojson"]["features"]
        }
        raw_zones = {z["id"]: z for z in connector_payload["zones"]}
        features: list[dict[str, Any]] = []

        total_buildings = 0
        total_population = 0
        total_roads_km = 0.0
        total_farmland_ha = 0.0
        total_exposed_val_inr = 0.0

        for idx, z in enumerate(connector_payload["zones"]):
            z_props = zone_lookup[z["id"]]
            sev: SeverityLevel = z_props["severity"]
            factor = 0.15 if sev == "safe" else 1.0
            b_count = int(round(z["buildings"] * factor))
            p_count = int(round(z["population"] * factor))
            f_ha = round(z["farmland_ha"] * factor, 1)
            r_km = round(z["roads_km"] * factor, 1)
            z_val_inr = (z["housing_val"] + z["agri_val"] + z["infra_val"] + z["biz_val"]) * INR_SCALE

            if sev != "safe":
                total_buildings += b_count
                total_population += p_count
                total_farmland_ha += f_ha
                total_roads_km += r_km
                total_exposed_val_inr += z_val_inr

            c_lon, c_lat = z["center"]
            for b_i in range(3):
                angle = (b_i * 2.1) + idx * 0.7
                blon = round(c_lon + math.cos(angle) * z["rx"] * 0.38, 5)
                blat = round(c_lat + math.sin(angle) * z["ry"] * 0.38, 5)
                d_lon, d_lat = z["rx"] * 0.09, z["ry"] * 0.09
                features.append(
                    {
                        "type": "Feature",
                        "id": f"BLD-{z['id']}-{b_i+1}",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [blon - d_lon, blat - d_lat],
                                    [blon + d_lon, blat - d_lat],
                                    [blon + d_lon, blat + d_lat],
                                    [blon - d_lon, blat + d_lat],
                                    [blon - d_lon, blat - d_lat],
                                ]
                            ],
                        },
                        "properties": {
                            "asset_id": f"BLD-{z['id']}-{b_i+1}",
                            "asset_type": "building",
                            "name": f"{z['name']} Building Cluster #{b_i+1}",
                            "zone_id": z["id"],
                            "zone_name": z["name"],
                            "severity": sev,
                            "sector": "housing" if b_i != 1 else "business",
                            "building_count": max(12, b_count // 3),
                            "exposure_value_inr": round(((z["housing_val"] + z["biz_val"]) * INR_SCALE) / 3.0, 2),
                            "hazard_intensity": z_props["hazard_intensity"],
                            "normalized_intensity": z_props["normalized_intensity"],
                            "vulnerability_score": z_props["vulnerability_score"],
                            "confidence": z_props["confidence"],
                        },
                    }
                )

            if z["farmland_ha"] > 20:
                flon = round(c_lon - z["rx"] * 0.35, 5)
                flat = round(c_lat - z["ry"] * 0.32, 5)
                f_dx, f_dy = z["rx"] * 0.22, z["ry"] * 0.18
                features.append(
                    {
                        "type": "Feature",
                        "id": f"FRM-{z['id']}",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [flon - f_dx, flat - f_dy],
                                    [flon + f_dx, flat - f_dy],
                                    [flon + f_dx, flat + f_dy],
                                    [flon - f_dx, flat + f_dy],
                                    [flon - f_dx, flat - f_dy],
                                ]
                            ],
                        },
                        "properties": {
                            "asset_id": f"FRM-{z['id']}",
                            "asset_type": "farmland",
                            "name": f"{z['name']} Cropland Belt",
                            "zone_id": z["id"],
                            "zone_name": z["name"],
                            "severity": sev,
                            "sector": "agriculture",
                            "area_ha": f_ha,
                            "exposure_value_inr": float(z["agri_val"] * INR_SCALE),
                            "hazard_intensity": z_props["hazard_intensity"],
                            "normalized_intensity": z_props["normalized_intensity"],
                            "vulnerability_score": z_props["vulnerability_score"],
                            "confidence": z_props["confidence"],
                        },
                    }
                )

            features.append(
                {
                    "type": "Feature",
                    "id": f"POP-{z['id']}",
                    "geometry": {"type": "Point", "coordinates": [c_lon, c_lat]},
                    "properties": {
                        "asset_id": f"POP-{z['id']}",
                        "asset_type": "population_cell",
                        "name": f"{z['name']} Population Cluster",
                        "zone_id": z["id"],
                        "zone_name": z["name"],
                        "severity": sev,
                        "population_exposed": p_count,
                        "hazard_intensity": z_props["hazard_intensity"],
                        "confidence": z_props["confidence"],
                    },
                }
            )

        for rd in connector_payload["roads"]:
            z_props = zone_lookup[rd["zone_id"]]
            features.append(
                {
                    "type": "Feature",
                    "id": rd["id"],
                    "geometry": {"type": "LineString", "coordinates": rd["coords"]},
                    "properties": {
                        "asset_id": rd["id"],
                        "asset_type": "road",
                        "name": rd["name"],
                        "zone_id": rd["zone_id"],
                        "zone_name": z_props["name"],
                        "severity": z_props["severity"],
                        "sector": "infrastructure",
                        "length_km": rd["length_km"],
                        "exposure_value_inr": float(rd["value_usd"] * INR_SCALE),
                        "hazard_intensity": z_props["hazard_intensity"],
                        "normalized_intensity": z_props["normalized_intensity"],
                        "vulnerability_score": z_props["vulnerability_score"],
                        "confidence": z_props["confidence"],
                    },
                }
            )

        crit_at_risk = 0
        for cf in connector_payload["critical_facilities"]:
            z_props = zone_lookup[cf["zone_id"]]
            if z_props["severity"] != "safe":
                crit_at_risk += 1
            features.append(
                {
                    "type": "Feature",
                    "id": cf["id"],
                    "geometry": {"type": "Point", "coordinates": [cf["lon"], cf["lat"]]},
                    "properties": {
                        "asset_id": cf["id"],
                        "asset_type": "critical_infrastructure",
                        "critical_category": cf["category"],
                        "name": cf["name"],
                        "zone_id": cf["zone_id"],
                        "zone_name": z_props["name"],
                        "severity": z_props["severity"],
                        "sector": "infrastructure",
                        "exposure_value_inr": float(cf["value_usd"] * INR_SCALE),
                        "criticality": cf["criticality"],
                        "hazard_intensity": z_props["hazard_intensity"],
                        "normalized_intensity": z_props["normalized_intensity"],
                        "vulnerability_score": z_props["vulnerability_score"],
                        "confidence": z_props["confidence"],
                    },
                }
            )

        sec_val_map = {
            "housing": "housing_val",
            "agriculture": "agri_val",
            "infrastructure": "infra_val",
            "business": "biz_val",
        }
        for z in connector_payload["zones"]:
            z_props = zone_lookup[z["id"]]
            rz = raw_zones[z["id"]]
            pol_lon = round(z["center"][0] + z["rx"] * 0.25, 5)
            pol_lat = round(z["center"][1] - z["ry"] * 0.15, 5)
            pol_val_inr = rz[sec_val_map[rz["sector"]]] * INR_SCALE
            features.append(
                {
                    "type": "Feature",
                    "id": rz["policy_id"],
                    "geometry": {"type": "Point", "coordinates": [pol_lon, pol_lat]},
                    "properties": {
                        "asset_id": rz["policy_id"],
                        "asset_type": "policy_zone",
                        "policy_id": rz["policy_id"],
                        "policyholder": rz["policyholder"],
                        "name": rz["asset_name"],
                        "zone_id": z["id"],
                        "zone_name": z["name"],
                        "severity": z_props["severity"],
                        "sector": rz["sector"],
                        "exposure_value_inr": float(pol_val_inr),
                        "hazard_intensity": z_props["hazard_intensity"],
                        "normalized_intensity": z_props["normalized_intensity"],
                        "vulnerability_score": z_props["vulnerability_score"],
                        "confidence": z_props["confidence"],
                    },
                }
            )

        summary = ExposureSummary(
            buildings_affected=total_buildings,
            population_exposed=total_population,
            roads_affected_km=round(total_roads_km, 1),
            farmland_affected_ha=round(total_farmland_ha, 1),
            critical_facilities_at_risk=crit_at_risk,
            total_exposed_value_inr=round(total_exposed_val_inr, 2),
            datasets=connector_payload["provenance"]["exposure"],
        )
        return {
            "geojson": {"type": "FeatureCollection", "features": features},
            "summary": summary.model_dump(),
        }


class TransparentEconomicModel(EconomicModel):
    model_name = "claimpulse-transparent-loss-v1"

    def estimate_loss(
        self,
        event_req: EventRequest,
        damage_result: dict[str, Any],
        exposure_result: dict[str, Any],
        connector_payload: dict[str, Any] | None = None,
    ) -> LossEstimate:
        payload = connector_payload or build_scenario_raw_data(event_req)
        zone_props_map = {
            f["properties"]["zone_id"]: f["properties"]
            for f in damage_result["geojson"]["features"]
        }

        sector_totals: dict[str, float] = {
            "housing": 0.0,
            "agriculture": 0.0,
            "infrastructure": 0.0,
            "business": 0.0,
        }
        sector_counts: dict[str, int] = {
            "housing": 0,
            "agriculture": 0,
            "infrastructure": 0,
            "business": 0,
        }
        severity_totals: dict[SeverityLevel, float] = {
            "safe": 0.0,
            "low": 0.0,
            "medium": 0.0,
            "high": 0.0,
            "extreme": 0.0,
        }
        biz_interruption_total = 0.0
        priority_zones: list[PriorityZone] = []

        for z in payload["zones"]:
            zp = zone_props_map[z["id"]]
            sev: SeverityLevel = zp["severity"]
            norm_int = zp["normalized_intensity"]
            vuln = zp["vulnerability_score"]

            zone_loss = 0.0
            for sec, val_key in [
                ("housing", "housing_val"),
                ("agriculture", "agri_val"),
                ("infrastructure", "infra_val"),
                ("business", "biz_val"),
            ]:
                exp_val_inr = float(z[val_key]) * INR_SCALE
                dmg_mult = _damage_multiplier(event_req.disaster_type, sev, sec)
                sec_loss = round(norm_int * exp_val_inr * vuln * dmg_mult, 2)
                sector_totals[sec] += sec_loss
                if exp_val_inr > 0 and sev != "safe":
                    sector_counts[sec] += max(1, z["buildings"] // 4) if sec == "housing" else 1
                zone_loss += sec_loss
                if sec == "business":
                    biz_interruption_total += round(sec_loss * 0.32, 2)

            zone_loss = round(zone_loss, 2)
            severity_totals[sev] = round(severity_totals[sev] + zone_loss, 2)

            relief_amt = (
                round(zone_loss * 0.14 + z["population"] * 15540.0 * vuln * norm_int, 2)
                if sev != "safe"
                else 0.0
            )

            priority_action = (
                "Immediate emergency shelter, dewatering & medical airdrop"
                if sev == "extreme"
                else "Deploy rapid relief kits, mobile power & road clearance"
                if sev == "high"
                else "Standby relief staging & targeted utility inspection"
                if sev == "medium"
                else "Routine monitoring"
            )
            priority_zones.append(
                PriorityZone(
                    rank=0,
                    zone_id=z["id"],
                    zone_name=z["name"],
                    district_or_ward=z["district"],
                    severity=sev,
                    expected_loss_inr=zone_loss,
                    population_exposed=z["population"] if sev != "safe" else 0,
                    buildings_affected=z["buildings"] if sev != "safe" else 0,
                    vulnerability_score=vuln,
                    infrastructure_criticality=z["infra_crit"],
                    recommended_relief_inr=relief_amt,
                    priority_action=priority_action,
                )
            )

        priority_zones.sort(
            key=lambda p: (
                p.expected_loss_inr * 0.45
                + p.population_exposed * 100800.0 * p.vulnerability_score
                + p.infrastructure_criticality * 672000000.0
            ),
            reverse=True,
        )
        ranked_top_10: list[PriorityZone] = [
            pz.model_copy(update={"rank": idx + 1}) for idx, pz in enumerate(priority_zones[:10])
        ]

        total_loss = round(sum(sector_totals.values()), 2)
        uncertainty_low = round(total_loss * 0.82, 2)
        uncertainty_high = round(total_loss * 1.21, 2)

        labels = {
            "housing": "Housing & Residential",
            "agriculture": "Agriculture & Cropland",
            "infrastructure": "Infrastructure & Utilities",
            "business": "Commercial & Business Interruption",
        }
        by_sector = [
            SectorLoss(
                sector=sec,  # type: ignore[arg-type]
                label=labels[sec],
                expected_loss_inr=round(sector_totals[sec], 2),
                share_pct=round((sector_totals[sec] / max(1.0, total_loss)) * 100.0, 1),
                asset_count=sector_counts[sec],
                business_interruption_inr=round(biz_interruption_total, 2) if sec == "business" else 0.0,
            )
            for sec in ["housing", "agriculture", "infrastructure", "business"]
        ]

        dates = payload["timeline_dates"]
        weights = [0.08, 0.34, 0.78, 0.95, 1.0][: len(dates)]
        if len(weights) < len(dates):
            weights = [round((i + 1) / len(dates), 2) for i in range(len(dates))]
        total_area = damage_result["metrics"]["affected_area_km2"]
        time_series = [
            TimeSeriesLossPoint(
                date=d,
                label="Baseline" if i == 0 else ("Peak Event" if i == len(dates) - 2 else f"T+{i*2}d"),
                cumulative_loss_inr=round(total_loss * weights[i], 2),
                affected_area_km2=round(total_area * weights[i], 2),
                confidence=round(0.84 + 0.02 * i, 2),
            )
            for i, d in enumerate(dates)
        ]

        assumptions = [
            FormulaAssumption(
                parameter="Normalized Hazard Intensity",
                value="Flood: depth(m)/3.6m | Wildfire: dNBR/1.0 | Cyclone: idx/2.2",
                rationale="Maps physical satellite intensity onto a 0.0–1.0 bounded damage driver while preserving linear sensitivity to depth/severity scaling.",
                source="Copernicus DEM + Sentinel-1 SAR / FIRMS",
            ),
            FormulaAssumption(
                parameter="Depth-Damage Function Multiplier",
                value="Safe: 0.03 | Low: 0.14 | Medium: 0.28 | High: 0.44 | Extreme: 0.62",
                rationale="Calibrated from JRC / NDMA flood depth-damage curves adjusted by sector (Agriculture ×1.12, Business ×1.08, Infrastructure ×0.92).",
                source="JRC Global Flood Depth-Damage Functions",
            ),
            FormulaAssumption(
                parameter="Business Interruption Add-on",
                value="32% of direct commercial physical loss (in INR)",
                rationale="Captures downtime from power/road disruption and supply-chain halt during 5–14 day recovery window.",
                source="Regional MSME Recovery Benchmark (Demo)",
            ),
            FormulaAssumption(
                parameter="Uncertainty Envelope",
                value="-18% to +21% around point estimate",
                rationale="Accounts for SAR speckle under tree canopy, 30m DEM vertical RMSE (±0.5m), and synthetic replacement-cost variance.",
                source="ClaimPulse Error Propagation Model",
            ),
        ]
        limitations = [
            "Bundled demo scenario uses curated satellite-derived geometries paired with synthetic Indian Rupee (₹) insurance policy ledgers.",
            "Sentinel-1 SAR can underestimate standing water beneath dense urban high-rises or mangrove canopies due to double-bounce backscatter.",
            "All provisional payouts > ₹8.82 Cr or confidence < 86% are automatically routed to Mandatory Human Review—never auto-denied.",
        ]

        # Build distinct domain-specific economic calculations & parameters for every stakeholder option
        stakeholder_models = _build_stakeholder_models(
            event_req=event_req,
            payload=payload,
            zone_props_map=zone_props_map,
            sector_totals=sector_totals,
            biz_interruption_total=biz_interruption_total,
            priority_zones=priority_zones,
            total_loss=total_loss,
        )

        return LossEstimate(
            total_expected_loss_inr=total_loss,
            business_interruption_inr=round(biz_interruption_total, 2),
            uncertainty_low_inr=uncertainty_low,
            uncertainty_high_inr=uncertainty_high,
            formula_text="Expected loss (₹) = Hazard intensity (norm) × Exposure value (₹) × Vulnerability × Damage-function multiplier",
            by_sector=by_sector,
            by_severity=severity_totals,
            time_series=time_series,
            top_10_zones=ranked_top_10,
            assumptions=assumptions,
            limitations=limitations,
            stakeholder_models=stakeholder_models,
        )


def _build_stakeholder_models(
    event_req: EventRequest,
    payload: dict[str, Any],
    zone_props_map: dict[str, dict[str, Any]],
    sector_totals: dict[str, float],
    biz_interruption_total: float,
    priority_zones: list[PriorityZone],
    total_loss: float,
) -> dict[str, StakeholderPrediction]:
    """Compute distinct calculations, parameter breakdowns, and zone impacts per stakeholder mode."""
    pz_map = {p.zone_id: p for p in priority_zones}

    # 1. Insurer model (Claims & Underwriting Liquidity)
    ins_zone_map: dict[str, float] = {}
    ins_by_sec = {"housing": 0.0, "agriculture": 0.0, "infrastructure": 0.0, "business": 0.0}
    for idx, z in enumerate(payload["zones"]):
        zp = zone_props_map[z["id"]]
        sev: SeverityLevel = zp["severity"]
        norm_int = zp["normalized_intensity"]
        vuln = zp["vulnerability_score"]
        sec = z["sector"]
        val_key = "housing_val" if sec == "housing" else ("agri_val" if sec == "agriculture" else ("infra_val" if sec == "infrastructure" else "biz_val"))
        pol_exp = float(z[val_key]) * INR_SCALE * (0.14 + 0.02 * (idx % 4))
        dmg_mult = _damage_multiplier(event_req.disaster_type, sev, sec)
        claim_amt = round(norm_int * pol_exp * vuln * dmg_mult, 2) if sev != "safe" else 0.0
        ins_zone_map[z["id"]] = claim_amt
        ins_by_sec[sec] += claim_amt
    ins_total = round(sum(ins_zone_map.values()), 2)

    # 2. Government Relief & Recovery Funds (SDRF / NDRF + Infrastructure Restoration + Ex-gratia)
    gov_zone_map: dict[str, float] = {}
    gov_infra = 0.0
    gov_housing_exgratia = 0.0
    gov_crop_subsidy = 0.0
    gov_shelter_dewater = 0.0
    for z in payload["zones"]:
        zp = zone_props_map[z["id"]]
        sev: SeverityLevel = zp["severity"]
        norm_int = zp["normalized_intensity"]
        vuln = zp["vulnerability_score"]
        if sev == "safe":
            gov_zone_map[z["id"]] = 0.0
            continue
        p_infra = round(z["infra_val"] * INR_SCALE * norm_int * 0.16 * z["infra_crit"], 2)
        p_house = round(z["buildings"] * 38000.0 * norm_int * vuln, 2)
        p_crop = round(z["farmland_ha"] * 22500.0 * norm_int, 2)
        p_shelter = round(z["population"] * 6800.0 * vuln * norm_int, 2)
        gov_infra += p_infra
        gov_housing_exgratia += p_house
        gov_crop_subsidy += p_crop
        gov_shelter_dewater += p_shelter
        gov_zone_map[z["id"]] = round(p_infra + p_house + p_crop + p_shelter, 2)
    gov_total = round(sum(gov_zone_map.values()), 2)

    # 3. Humanitarian Aid & Emergency Response (Food/WASH, Medical, Shelter Kits, Cash Assistance)
    hum_zone_map: dict[str, float] = {}
    hum_food_wash = 0.0
    hum_medical = 0.0
    hum_shelter_kits = 0.0
    hum_cash_transfer = 0.0
    total_exposed_pop = 0
    for z in payload["zones"]:
        zp = zone_props_map[z["id"]]
        sev: SeverityLevel = zp["severity"]
        norm_int = zp["normalized_intensity"]
        vuln = zp["vulnerability_score"]
        if sev == "safe":
            hum_zone_map[z["id"]] = 0.0
            continue
        pop = z["population"]
        total_exposed_pop += pop
        fw = round(pop * 3100.0 * norm_int * vuln, 2)
        med = round(pop * 2400.0 * norm_int * z["infra_crit"], 2)
        sh = round((pop / 4.2) * 11500.0 * norm_int, 2)
        cash = round((pop / 4.2) * 14000.0 * vuln * norm_int, 2)
        hum_food_wash += fw
        hum_medical += med
        hum_shelter_kits += sh
        hum_cash_transfer += cash
        hum_zone_map[z["id"]] = round(fw + med + sh + cash, 2)
    hum_total = round(sum(hum_zone_map.values()), 2)

    # 4. Lender / Bank Collateral Risk & Capital Buffer
    lnd_zone_map: dict[str, float] = {}
    lnd_agri_loan = 0.0
    lnd_msme_loan = 0.0
    lnd_housing_mort = 0.0
    lnd_infra_proj = 0.0
    for z in payload["zones"]:
        zp = zone_props_map[z["id"]]
        sev: SeverityLevel = zp["severity"]
        norm_int = zp["normalized_intensity"]
        vuln = zp["vulnerability_score"]
        if sev == "safe":
            lnd_zone_map[z["id"]] = 0.0
            continue
        ag = round(z["agri_val"] * INR_SCALE * 0.24 * norm_int * vuln, 2)
        ms = round(z["biz_val"] * INR_SCALE * 0.28 * norm_int * vuln, 2)
        hm = round(z["housing_val"] * INR_SCALE * 0.19 * norm_int * vuln, 2)
        ip = round(z["infra_val"] * INR_SCALE * 0.22 * norm_int * vuln, 2)
        lnd_agri_loan += ag
        lnd_msme_loan += ms
        lnd_housing_mort += hm
        lnd_infra_proj += ip
        lnd_zone_map[z["id"]] = round(ag + ms + hm + ip, 2)
    lnd_total = round(sum(lnd_zone_map.values()), 2)

    # 5. Business & Supply-Chain Interruption
    biz_zone_map: dict[str, float] = {}
    biz_direct = 0.0
    biz_downtime = 0.0
    biz_logistics = 0.0
    biz_utility_outage = 0.0
    for z in payload["zones"]:
        zp = zone_props_map[z["id"]]
        sev: SeverityLevel = zp["severity"]
        norm_int = zp["normalized_intensity"]
        vuln = zp["vulnerability_score"]
        if sev == "safe":
            biz_zone_map[z["id"]] = 0.0
            continue
        pz = pz_map[z["id"]]
        dmg_mult = _damage_multiplier(event_req.disaster_type, sev, "business")
        dir_dmg = round(norm_int * (z["biz_val"] * INR_SCALE) * vuln * dmg_mult, 2)
        down = round(dir_dmg * 0.34, 2)
        logi = round((z["roads_km"] * 2400000.0) * norm_int * z["infra_crit"], 2)
        util = round(pz.expected_loss_inr * 0.09 * z["infra_crit"], 2)
        biz_direct += dir_dmg
        biz_downtime += down
        biz_logistics += logi
        biz_utility_outage += util
        biz_zone_map[z["id"]] = round(dir_dmg + down + logi + util, 2)
    biz_total = round(sum(biz_zone_map.values()), 2)

    def _mk_params(items: list[tuple[str, float, str]], denom: float) -> list[StakeholderParameter]:
        d = max(1.0, denom)
        return [
            StakeholderParameter(
                name=n,
                amount_inr=round(v, 2),
                share_pct=round((v / d) * 100.0, 1),
                driver_note=note,
            )
            for n, v, note in items
        ]

    return {
        "insurer": StakeholderPrediction(
            stakeholder="insurer",
            label="Insurance Claims & Liquidity",
            primary_metric_title="Predicted Claims Payout",
            primary_amount_inr=ins_total,
            uncertainty_low_inr=round(ins_total * 0.82, 2),
            uncertainty_high_inr=round(ins_total * 1.21, 2),
            secondary_metric_label="Portfolio Loss Share",
            secondary_metric_value=f"{round((ins_total / max(1.0, total_loss)) * 100.0, 1)}% insured",
            headline=f"Reserve {fmt_inr(ins_total)} for provisional claims payout across {len(payload['zones'])} policy zones.",
            formula="Claim Payout = norm_intensity × Insured Exposure × Vulnerability × Sector Damage Curve",
            parameters=_mk_params(
                [
                    ("Commercial & MSME", ins_by_sec["business"], "Property + 32% BI cover"),
                    ("Infrastructure Policies", ins_by_sec["infrastructure"], "Substations, ports & tollways"),
                    ("Residential Housing", ins_by_sec["housing"], "Dwelling structural damage"),
                    ("Agri & Crop Pools", ins_by_sec["agriculture"], "Parametric crop triggers"),
                ],
                ins_total,
            ),
            zone_amounts_inr=ins_zone_map,
        ),
        "government": StakeholderPrediction(
            stakeholder="government",
            label="Government Relief & Recovery Funds",
            primary_metric_title="Predicted Relief Budget",
            primary_amount_inr=gov_total,
            uncertainty_low_inr=round(gov_total * 0.85, 2),
            uncertainty_high_inr=round(gov_total * 1.18, 2),
            secondary_metric_label="Per-Capita Relief",
            secondary_metric_value=fmt_inr(gov_total / max(1, total_exposed_pop)),
            headline=f"Sanction {fmt_inr(gov_total)} in SDRF/NDRF relief funds covering public infrastructure repair, housing ex-gratia, and crop subsidies.",
            formula="Gov Relief = Public Infra Repair (16%×Crit) + Ex-gratia Housing (₹38k/bldg) + Crop Subsidy (₹22.5k/ha) + Dewatering",
            parameters=_mk_params(
                [
                    ("Public Infra Restoration", gov_infra, "Roads, bridges, grid & water plants"),
                    ("Housing Ex-Gratia", gov_housing_exgratia, "₹38,000 × intensity × vulnerability"),
                    ("Farmer Input Subsidy", gov_crop_subsidy, "₹22,500/ha × hazard intensity"),
                    ("Shelter & Dewatering", gov_shelter_dewater, "₹6,800/resident emergency ops"),
                ],
                gov_total,
            ),
            zone_amounts_inr=gov_zone_map,
        ),
        "humanitarian": StakeholderPrediction(
            stakeholder="humanitarian",
            label="Humanitarian Aid & Emergency Response",
            primary_metric_title="Predicted Aid Requirement",
            primary_amount_inr=hum_total,
            uncertainty_low_inr=round(hum_total * 0.86, 2),
            uncertainty_high_inr=round(hum_total * 1.22, 2),
            secondary_metric_label="Families Requiring Aid",
            secondary_metric_value=f"{int(round(total_exposed_pop / 4.2)):,} households",
            headline=f"Mobilize {fmt_inr(hum_total)} in immediate humanitarian aid for {int(round(total_exposed_pop / 4.2)):,} vulnerable households.",
            formula="Aid Need = Food & WASH (₹3.1k/pop) + Medical (₹2.4k/pop) + Shelter Kits (₹11.5k/HH) + Direct Cash (₹14k/HH)",
            parameters=_mk_params(
                [
                    ("Direct Household Cash", hum_cash_transfer, "₹14,000/HH × vulnerability"),
                    ("Temporary Shelter Kits", hum_shelter_kits, "Tarpaulins, solar & bedding"),
                    ("Food & Clean Water (WASH)", hum_food_wash, "14-day ration + chlorination"),
                    ("Emergency Medical & Trauma", hum_medical, "Mobile clinics & cold-chain"),
                ],
                hum_total,
            ),
            zone_amounts_inr=hum_zone_map,
        ),
        "lender": StakeholderPrediction(
            stakeholder="lender",
            label="Bank / Lender Collateral & Credit Risk",
            primary_metric_title="Loan Portfolio at Risk",
            primary_amount_inr=lnd_total,
            uncertainty_low_inr=round(lnd_total * 0.80, 2),
            uncertainty_high_inr=round(lnd_total * 1.24, 2),
            secondary_metric_label="Avg Risk-Premium Adj.",
            secondary_metric_value="+95 bps",
            headline=f"Flag {fmt_inr(lnd_total)} in collateral exposure at risk; enforce moratoriums and +35 to +145 bps risk-pricing.",
            formula="Collateral VaR = Sector Credit Penetration (19–28%) × Asset Value × Hazard Intensity × Vulnerability",
            parameters=_mk_params(
                [
                    ("MSME & Working Capital", lnd_msme_loan, "28% credit exposure factor"),
                    ("Infra Project Finance", lnd_infra_proj, "22% project debt exposure"),
                    ("Housing Mortgages", lnd_housing_mort, "19% residential LTV stress"),
                    ("Agri KCC & Crop Loans", lnd_agri_loan, "24% seasonal crop credit"),
                ],
                lnd_total,
            ),
            zone_amounts_inr=lnd_zone_map,
        ),
        "business": StakeholderPrediction(
            stakeholder="business",
            label="Business & Supply-Chain Interruption",
            primary_metric_title="Predicted Business Impact",
            primary_amount_inr=biz_total,
            uncertainty_low_inr=round(biz_total * 0.81, 2),
            uncertainty_high_inr=round(biz_total * 1.25, 2),
            secondary_metric_label="Est. Recovery Downtime",
            secondary_metric_value="9–14 days",
            headline=f"Mitigate {fmt_inr(biz_total)} in combined commercial asset damage, operational downtime, and corridor disruption.",
            formula="Business Impact = Direct Commercial Loss + 34% Downtime + Road Corridor Disruption + Utility Outage Penalty",
            parameters=_mk_params(
                [
                    ("Direct Facility Damage", biz_direct, "Warehouses, plants & retail"),
                    ("Utility & Power Outage", biz_utility_outage, "Substation criticality penalty"),
                    ("Operational Downtime", biz_downtime, "9–14 day revenue halt"),
                    ("Freight Corridor Disruption", biz_logistics, "₹24 Lakh/km road rerouting"),
                ],
                biz_total,
            ),
            zone_amounts_inr=biz_zone_map,
        ),
    }


class MultiStakeholderDecisionModule(DecisionWorkflowModule):
    workflow_name = "claimpulse-explainable-decisions-v1"

    def evaluate_decisions(
        self,
        event_req: EventRequest,
        damage_result: dict[str, Any],
        exposure_result: dict[str, Any],
        loss_estimate: LossEstimate,
        connector_payload: dict[str, Any] | None = None,
    ) -> DecisionBundle:
        payload = connector_payload or build_scenario_raw_data(event_req)
        zone_props_map = {
            f["properties"]["zone_id"]: f["properties"]
            for f in damage_result["geojson"]["features"]
        }
        ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        haz_ts = payload["provenance"]["hazard"][0].timestamp
        exp_ts = payload["provenance"]["exposure"][0].timestamp

        claims: list[InsuranceClaimDecision] = []
        lender_alerts: list[LenderCollateralAlert] = []
        audit_trail: list[AuditRecord] = []

        fast_tracked = 0
        manual_review = 0
        total_payout = 0.0
        fast_track_cap_inr = 88200000.0  # ₹8.82 Cr

        for idx, z in enumerate(payload["zones"]):
            zp = zone_props_map[z["id"]]
            sev: SeverityLevel = zp["severity"]
            conf: float = zp["confidence"]
            norm_int: float = zp["normalized_intensity"]
            vuln: float = zp["vulnerability_score"]
            sec = z["sector"]

            val_key = (
                "housing_val"
                if sec == "housing"
                else "agri_val"
                if sec == "agriculture"
                else "infra_val"
                if sec == "infrastructure"
                else "biz_val"
            )
            pol_exposure_inr = round(float(z[val_key]) * INR_SCALE * (0.14 + 0.02 * (idx % 4)), 2)
            dmg_mult = _damage_multiplier(event_req.disaster_type, sev, sec)
            est_claim_inr = round(norm_int * pol_exposure_inr * vuln * dmg_mult, 2)
            dmg_prob = round(min(0.98, max(0.05, norm_int * 0.85 + vuln * 0.15)), 2)

            rec: ClaimRecommendation
            human_req = True
            if conf < 0.55:
                rec = "Insufficient evidence"
            elif conf < 0.72:
                rec = "Investigate"
            elif est_claim_inr > fast_track_cap_inr or conf < 0.86:
                rec = "Manual review"
            elif sev in ("high", "extreme") and conf >= 0.86:
                rec = "Auto-fast-track"
                human_req = False
            else:
                rec = "Manual review"

            if rec == "Auto-fast-track":
                fast_tracked += 1
            else:
                manual_review += 1
            if sev != "safe":
                total_payout += est_claim_inr

            drivers = [
                f"{zp['hazard_intensity']} {zp['intensity_unit']} ({sev.upper()} tier)",
                f"Vulnerability index {vuln:.2f} in {z['district']}",
                f"DEM elevation {z['elevation_m']} m (Copernicus GLO-30)",
            ]
            if conf < 0.78:
                drivers.append(f"Reduced SAR/optical confidence ({int(conf*100)}%) — manual verification safeguard")

            calc_str = (
                f"norm_intensity({norm_int:.3f}) × exposure({fmt_inr(pol_exposure_inr)}) "
                f"× vulnerability({vuln:.2f}) × dmg_fn({dmg_mult:.3f}) = {fmt_inr(est_claim_inr)}"
            )
            dec_id = f"CLM-{z['policy_id']}"
            claims.append(
                InsuranceClaimDecision(
                    decision_id=dec_id,
                    policy_id=z["policy_id"],
                    policyholder=z["policyholder"],
                    asset_name=z["asset_name"],
                    zone_name=z["name"],
                    sector=sec,
                    severity=sev,
                    damage_probability=dmg_prob,
                    exposure_value_inr=pol_exposure_inr,
                    estimated_claim_inr=est_claim_inr,
                    uncertainty_low_inr=round(est_claim_inr * 0.82, 2),
                    uncertainty_high_inr=round(est_claim_inr * 1.21, 2),
                    confidence=conf,
                    recommendation=rec,
                    review_status="Approved" if rec == "Auto-fast-track" else "Pending",
                    human_review_required=human_req,
                    key_drivers=drivers,
                    calculation_breakdown=calc_str,
                    evidence_links=[
                        f"Sentinel-1/VIIRS Layer ({z['id']})",
                        "Copernicus DEM Elevation Profile",
                        f"Policy Ledger {z['policy_id']}",
                    ],
                    data_timestamps=[haz_ts, exp_ts],
                )
            )

            if idx < 5 and sev != "safe":
                port_exp_inr = round(pol_exposure_inr * 1.45, 2)
                col_loss_inr = round(est_claim_inr * 1.35, 2)
                ltv_stress = round(min(48.0, (col_loss_inr / max(1.0, port_exp_inr)) * 100.0), 1)
                flag = "Critical" if sev == "extreme" else ("Elevated" if sev == "high" else "Watch")
                bps = 145 if flag == "Critical" else (85 if flag == "Elevated" else 35)
                lender_alerts.append(
                    LenderCollateralAlert(
                        loan_id=f"LN-{2026}-{idx+101}",
                        borrower=z["policyholder"],
                        sector=sec,
                        collateral_name=z["asset_name"],
                        portfolio_exposure_inr=port_exp_inr,
                        expected_collateral_loss_inr=col_loss_inr,
                        ltv_stress_pct=ltv_stress,
                        risk_flag=flag,  # type: ignore[arg-type]
                        recommended_action=(
                            "Require parametric flood endorsement & 14-day collateral inspection"
                            if flag == "Critical"
                            else "Increase satellite monitoring cadence & review covenant buffer"
                        ),
                        risk_premium_bps_adj=bps,
                        confidence=conf,
                    )
                )

        total_relief = round(sum(p.recommended_relief_inr for p in loss_estimate.top_10_zones), 2)
        total_lender_risk = round(sum(a.expected_collateral_loss_inr for a in lender_alerts), 2)
        avg_conf = round(sum(c.confidence for c in claims) / max(1, len(claims)), 2)

        audit_trail.append(
            AuditRecord(
                audit_id=f"AUD-{payload['id'].upper()}-001",
                timestamp=ts_now,
                actor="ClaimPulse Decision Engine v1",
                action="Initial Satellite-to-Economic Triage Completed",
                target_id=payload["id"],
                rationale=(
                    f"Processed {len(claims)} policy zones across {damage_result['metrics']['affected_area_km2']} km²; "
                    f"{fast_tracked} fast-tracked, {manual_review} routed to human review (zero auto-denials)."
                ),
                inputs_snapshot={
                    "intensity_multiplier": event_req.intensity_multiplier,
                    "total_expected_loss_inr": loss_estimate.total_expected_loss_inr,
                    "hazard_timestamp": haz_ts,
                },
            )
        )

        track_b_alert = payload.get("track_b_alert")
        if isinstance(track_b_alert, dict):
            action_code = track_b_alert.get("recommended_action_code", "manual_review")
            trigger_st = track_b_alert.get("parametric_trigger_status", "none")
            priority = track_b_alert.get("downlink_priority", "P2_ROUTINE")
            audit_trail.insert(
                0,
                AuditRecord(
                    audit_id=f"AUD-TRKB-{payload['id'].upper()[:16]}",
                    timestamp=ts_now,
                    actor=f"Track B Orbit Bridge ({track_b_alert.get('satellite_id', 'LEO-SAT')})",
                    action=f"Track B Edge-Verified Action Executed: {action_code}",
                    target_id=payload["id"],
                    rationale=(
                        f"Ingested {track_b_alert.get('insight_payload_bytes', 0)}-byte edge telemetry "
                        f"[{priority} | trigger={trigger_st} | model={track_b_alert.get('model_version_hash', '')}] "
                        f"with confidence {track_b_alert.get('inference_confidence', 0.0):.2f}."
                    ),
                    inputs_snapshot={
                        "satellite_id": track_b_alert.get("satellite_id"),
                        "recommended_action_code": action_code,
                        "parametric_trigger_status": trigger_st,
                        "downlink_priority": priority,
                        "insight_payload_bytes": track_b_alert.get("insight_payload_bytes"),
                    },
                ),
            )

        headline = (
            f"Release {fmt_inr(total_payout)} provisional insurance liquidity ({fast_tracked} fast-track) "
            f"& allocate {fmt_inr(total_relief)} emergency relief across top {len(loss_estimate.top_10_zones)} priority zones."
        )

        return DecisionBundle(
            recommended_action_headline=headline,
            total_provisional_payout_inr=round(total_payout, 2),
            total_recommended_relief_inr=total_relief,
            total_lender_exposure_at_risk_inr=total_lender_risk,
            overall_confidence=avg_conf,
            uncertainty_range_inr=[
                round(total_payout * 0.82, 2),
                round(total_payout * 1.21, 2),
            ],
            claims_fast_tracked_count=fast_tracked,
            claims_manual_review_count=manual_review,
            relief_priority_zones_count=len([p for p in loss_estimate.top_10_zones if p.severity != "safe"]),
            claims=claims,
            relief_allocations=loss_estimate.top_10_zones,
            lender_alerts=lender_alerts,
            audit_trail=audit_trail,
        )
