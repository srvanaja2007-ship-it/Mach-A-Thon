from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from backend.core.interfaces import PlatformRegistry
from backend.core.schemas import (
    AskRequest,
    AuditRecord,
    BoundingBox,
    ConnectedModule,
    EventFullResponse,
    EventRequest,
    EventSummary,
    ReviewDecisionRequest,
    WebhookDelivery,
    WebhookSubscription,
)
from backend.exporters.export_service import export_csv, export_geojson, export_pdf
from backend.modules.implementations import (
    BaseHazardModule,
    DemoSatelliteConnector,
    MultiStakeholderDecisionModule,
    StandardExposureModule,
    TrackBEdgeDataConnector,
    TransparentEconomicModel,
    build_raw_data_from_track_b,
)
from backend.scenarios.data_builder import PRESET_EVENTS
from backend.services.analyst import answer_question


registry = PlatformRegistry()
registry.register_connector(DemoSatelliteConnector())
registry.register_connector(TrackBEdgeDataConnector())
registry.register_hazard(BaseHazardModule("flood", "sentinel1-viirs-flood-v1"))
registry.register_hazard(BaseHazardModule("wildfire", "firms-dnbr-wildfire-v1"))
registry.register_hazard(BaseHazardModule("cyclone", "wind-surge-blackmarble-cyclone-v1"))
registry.register_exposure(StandardExposureModule())
registry.register_economic_model(TransparentEconomicModel())
registry.register_decision_workflow(MultiStakeholderDecisionModule())

for mod in [
    ConnectedModule(
        module_id="mod-dronescout",
        name="DroneScout Field Verify",
        partner_team="Team AeroInspect",
        category="data_connector",
        description="Dispatches UAV orthomosaic verification for claims flagged Manual Review or Insufficient Evidence.",
        status="active",
        layer_id="critical_infrastructure",
        action_label="Dispatch UAV Verification",
        kpis={"Queue": "4 flights ready", "Resolution": "3 cm/px"},
        webhook_url="https://partner.claimpulse.demo/webhooks/dronescout",
    ),
    ConnectedModule(
        module_id="mod-agroyield",
        name="AgroYield Crop Index",
        partner_team="Team HarvestAI",
        category="economic_model",
        description="Refines Kharif/orchard crop loss curves using NDVI pre-event biomass baselines.",
        status="active",
        layer_id="farmland",
        action_label="Sync Crop Index",
        kpis={"Crop Accuracy": "91.4%", "Coverage": "FAO + Sentinel-2"},
    ),
    ConnectedModule(
        module_id="mod-gridguard",
        name="GridGuard Utility Telemetry",
        partner_team="Team PowerPulse",
        category="exposure_module",
        description="Streams substation load-drop & outage telemetry to weight infrastructure criticality scores.",
        status="active",
        layer_id="insurance_portfolio",
        action_label="Check Substation Status",
        kpis={"Substations": "6 monitored", "Latency": "Real-time stub"},
    ),
    ConnectedModule(
        module_id="mod-trackb-orbit",
        name="Track B Orbit-to-Edge Telemetry",
        partner_team="Track B Aerospace Systems",
        category="data_connector",
        description="Ingests INT8 onboard satellite edge alerts + HAPS/Drone 0.04m LEO-GNSS verification.",
        status="active",
        layer_id="hazard_extent",
        action_label="Ingest Track B Edge Alert",
        kpis={"Compression": "99.99%", "GNSS Precision": "0.04 m"},
    ),
]:
    registry.register_ui_module(mod)


EVENT_STORE: dict[str, EventFullResponse] = {}
EVENT_REQUESTS: dict[str, EventRequest] = {}
WEBHOOKS: list[WebhookSubscription] = [
    WebhookSubscription(
        webhook_id="WH-PARTNER-01",
        partner_name="Insurer Core Claims Bus",
        url="https://claims.partner-insurer.demo/hooks/claimpulse",
        events=["event.processed", "decision.updated"],
        active=True,
    ),
    WebhookSubscription(
        webhook_id="WH-NDMA-02",
        partner_name="State Disaster Relief Operations",
        url="https://sdma.gov.demo/api/relief-dispatch",
        events=["event.processed"],
        active=True,
    ),
]
WEBHOOK_DELIVERIES: list[WebhookDelivery] = []


def _emit_webhook(event_type: str, payload: dict[str, Any]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for wh in WEBHOOKS:
        if wh.active and event_type in wh.events:
            WEBHOOK_DELIVERIES.insert(
                0,
                WebhookDelivery(
                    delivery_id=f"DLV-{uuid.uuid4().hex[:8].upper()}",
                    webhook_id=wh.webhook_id,
                    event_type=event_type,
                    timestamp=ts,
                    status="202 Accepted (Stub)",
                    payload_preview=payload,
                ),
            )
    if len(WEBHOOK_DELIVERIES) > 30:
        del WEBHOOK_DELIVERIES[30:]


def _run_pipeline(
    req: EventRequest, raw_override: dict[str, Any] | None = None
) -> EventFullResponse:
    connector = registry.connectors["nasa-copernicus-demo-connector"]
    raw = raw_override if raw_override is not None else connector.fetch(req)
    hazard_mod = registry.hazard_modules[req.disaster_type]
    exposure_mod = registry.exposure_modules["osm-worldpop-fao-exposure-v1"]
    econ_mod: TransparentEconomicModel = registry.economic_models["claimpulse-transparent-loss-v1"]  # type: ignore[assignment]
    decision_mod: MultiStakeholderDecisionModule = registry.decision_workflows["claimpulse-explainable-decisions-v1"]  # type: ignore[assignment]

    dmg = hazard_mod.compute_damage(req, raw)
    exp = exposure_mod.overlay_exposure(req, dmg, raw)
    loss = econ_mod.estimate_loss(req, dmg, exp, raw)
    decisions = decision_mod.evaluate_decisions(req, dmg, exp, loss, raw)

    # Enrich damage polygons with per-stakeholder predicted amounts (INR) for live map switching
    for feat in dmg["geojson"]["features"]:
        zid = feat["properties"].get("zone_id")
        for sk_key, sk_model in loss.stakeholder_models.items():
            feat["properties"][f"{sk_key}_inr"] = sk_model.zone_amounts_inr.get(zid, 0.0)

    ev_id = raw["id"]
    if abs(req.intensity_multiplier - 1.0) > 0.01 and req.preset_id:
        ev_id = f"{raw['id']}-x{int(round(req.intensity_multiplier*100))}"

    summary = EventSummary(
        id=ev_id,
        preset_id=raw.get("preset_id"),
        title=raw["title"],
        disaster_type=raw["disaster_type"],
        country=raw["country"],
        state=raw["state"],
        district=raw["district"],
        start_date=raw["start_date"],
        end_date=raw["end_date"],
        center=raw["center"],
        zoom=raw["zoom"],
        bbox=BoundingBox(**raw["bbox"]),
        is_demo=raw.get("is_demo", True),
        status="processed",
        overall_confidence=decisions.overall_confidence,
        data_freshness=dmg["metrics"]["datasets"][0]["timestamp"],
        intensity_multiplier=req.intensity_multiplier,
    )
    bundle = EventFullResponse(
        event=summary,
        damage=dmg,
        exposure=exp,
        loss=loss,
        decisions=decisions,
    )
    EVENT_STORE[ev_id] = bundle
    EVENT_REQUESTS[ev_id] = req
    return bundle


def _seed_presets() -> None:
    for pid, pdata in PRESET_EVENTS.items():
        req = EventRequest(
            preset_id=pid,
            disaster_type=pdata["disaster_type"],
            title=pdata["title"],
            country=pdata["country"],
            state=pdata["state"],
            district=pdata["district"],
            start_date=pdata["start_date"],
            end_date=pdata["end_date"],
            bbox=BoundingBox(**pdata["bbox"]),
            intensity_multiplier=1.0,
        )
        _run_pipeline(req)


_seed_presets()


def _get_bundle(event_id: str) -> EventFullResponse | None:
    return EVENT_STORE.get(event_id)


async def health(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "ClaimPulse Decision Engine",
            "events_cached": len(EVENT_STORE),
            "modules_active": len(registry.list_ui_modules()),
        }
    )


async def list_events(request: Request) -> JSONResponse:
    return JSONResponse([b.event.model_dump() for b in EVENT_STORE.values()])


async def create_or_run_event(request: Request) -> JSONResponse:
    body = await request.json()
    bbox_raw = body.get("bbox")
    bbox = BoundingBox(**bbox_raw) if isinstance(bbox_raw, dict) else None
    req = EventRequest(
        preset_id=body.get("preset_id"),
        disaster_type=body.get("disaster_type", "flood"),
        title=body.get("title"),
        country=body.get("country", "India"),
        state=body.get("state", "Odisha"),
        district=body.get("district", "Puri & Kendrapara"),
        start_date=body.get("start_date", "2026-10-18"),
        end_date=body.get("end_date", "2026-10-26"),
        bbox=bbox,
        intensity_multiplier=float(body.get("intensity_multiplier", 1.0)),
    )
    bundle = _run_pipeline(req)
    _emit_webhook(
        "event.processed",
        {
            "event_id": bundle.event.id,
            "disaster_type": bundle.event.disaster_type,
            "total_expected_loss_inr": bundle.loss.total_expected_loss_inr,
            "provisional_payout_inr": bundle.decisions.total_provisional_payout_inr,
        },
    )
    return JSONResponse(bundle.model_dump())


async def get_event(request: Request) -> JSONResponse:
    bundle = _get_bundle(request.path_params["event_id"])
    if not bundle:
        return JSONResponse({"detail": "Event not found"}, status_code=404)
    return JSONResponse(bundle.model_dump())


async def get_event_damage(request: Request) -> JSONResponse:
    bundle = _get_bundle(request.path_params["event_id"])
    if not bundle:
        return JSONResponse({"detail": "Event not found"}, status_code=404)
    return JSONResponse(bundle.damage)


async def get_event_exposure(request: Request) -> JSONResponse:
    bundle = _get_bundle(request.path_params["event_id"])
    if not bundle:
        return JSONResponse({"detail": "Event not found"}, status_code=404)
    return JSONResponse(bundle.exposure)


async def get_event_loss(request: Request) -> JSONResponse:
    bundle = _get_bundle(request.path_params["event_id"])
    if not bundle:
        return JSONResponse({"detail": "Event not found"}, status_code=404)
    return JSONResponse(bundle.loss.model_dump())


async def get_event_decisions(request: Request) -> JSONResponse:
    bundle = _get_bundle(request.path_params["event_id"])
    if not bundle:
        return JSONResponse({"detail": "Event not found"}, status_code=404)
    return JSONResponse(bundle.decisions.model_dump())


async def review_decision(request: Request) -> JSONResponse:
    event_id = request.path_params["event_id"]
    decision_id = request.path_params["decision_id"]
    bundle = _get_bundle(event_id)
    if not bundle:
        return JSONResponse({"detail": "Event not found"}, status_code=404)

    target = next((c for c in bundle.decisions.claims if c.decision_id == decision_id), None)
    if not target:
        return JSONResponse({"detail": "Decision not found"}, status_code=404)

    raw = await request.json()
    body = ReviewDecisionRequest(
        action=raw.get("action", "Approve"),
        reviewer=raw.get("reviewer", "Senior Claims Analyst"),
        notes=raw.get("notes", ""),
    )
    status_map = {
        "Approve": "Approved",
        "Escalate": "Escalated",
        "Request Field Verification": "Field Verification Requested",
    }
    target.review_status = status_map[body.action]  # type: ignore[assignment]
    if body.action == "Approve":
        target.recommendation = "Auto-fast-track"
        target.human_review_required = False
    elif body.action == "Request Field Verification":
        target.recommendation = "Investigate"
        target.human_review_required = True
    else:
        target.recommendation = "Manual review"
        target.human_review_required = True

    if body.notes:
        target.reviewer_notes = body.notes

    bundle.decisions.claims_fast_tracked_count = sum(
        1 for c in bundle.decisions.claims if c.recommendation == "Auto-fast-track"
    )
    bundle.decisions.claims_manual_review_count = sum(
        1 for c in bundle.decisions.claims if c.recommendation != "Auto-fast-track"
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bundle.decisions.audit_trail.insert(
        0,
        AuditRecord(
            audit_id=f"AUD-{uuid.uuid4().hex[:6].upper()}",
            timestamp=ts,
            actor=body.reviewer,
            action=f"Human Review: {body.action} ({target.policy_id})",
            target_id=decision_id,
            rationale=body.notes or f"Analyst executed {body.action} on {target.asset_name}.",
            inputs_snapshot={
                "policy_id": target.policy_id,
                "estimated_claim_inr": target.estimated_claim_inr,
                "new_status": target.review_status,
            },
        ),
    )
    _emit_webhook(
        "decision.updated",
        {
            "event_id": event_id,
            "decision_id": decision_id,
            "policy_id": target.policy_id,
            "review_status": target.review_status,
            "reviewer": body.reviewer,
        },
    )
    return JSONResponse(bundle.model_dump())


async def export_event(request: Request) -> Response:
    event_id = request.path_params["event_id"]
    bundle = _get_bundle(event_id)
    if not bundle:
        return JSONResponse({"detail": "Event not found"}, status_code=404)
    fmt = request.query_params.get("format", "csv").lower()
    if fmt == "csv":
        return Response(
            content=export_csv(bundle),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="claimpulse_{event_id}.csv"'},
        )
    if fmt == "geojson":
        return Response(
            content=export_geojson(bundle),
            media_type="application/geo+json",
            headers={"Content-Disposition": f'attachment; filename="claimpulse_{event_id}.geojson"'},
        )
    if fmt == "pdf":
        return Response(
            content=export_pdf(bundle),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="claimpulse_{event_id}_brief.pdf"'},
        )
    return JSONResponse({"detail": "Unsupported export format"}, status_code=400)


async def ask_claimpulse(request: Request) -> JSONResponse:
    event_id = request.path_params["event_id"]
    bundle = _get_bundle(event_id)
    if not bundle:
        return JSONResponse({"detail": "Event not found"}, status_code=404)
    raw = await request.json()
    body = AskRequest(question=raw.get("question", ""))
    req = EVENT_REQUESTS.get(event_id, EventRequest(preset_id=bundle.event.preset_id))
    resp = answer_question(bundle, body.question, req)
    return JSONResponse(resp.model_dump())


async def get_modules(request: Request) -> JSONResponse:
    return JSONResponse([m.model_dump() for m in registry.list_ui_modules()])


async def register_module(request: Request) -> JSONResponse:
    raw = await request.json()
    mod = ConnectedModule(
        module_id=raw.get("module_id") or f"mod-{uuid.uuid4().hex[:6]}",
        name=raw["name"],
        partner_team=raw.get("partner_team", "Partner Team"),
        category=raw.get("category", "data_connector"),
        description=raw.get("description", ""),
        status=raw.get("status", "active"),
        layer_id=raw.get("layer_id"),
        action_label=raw.get("action_label"),
        kpis=raw.get("kpis", {}),
        webhook_url=raw.get("webhook_url"),
    )
    registry.register_ui_module(mod)
    if mod.webhook_url:
        WEBHOOKS.append(
            WebhookSubscription(
                webhook_id=f"WH-{mod.module_id.upper()}",
                partner_name=mod.partner_team,
                url=mod.webhook_url,
                events=["event.processed", "decision.updated"],
                active=True,
            )
        )
    return JSONResponse(mod.model_dump())


async def get_webhooks(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "subscriptions": [w.model_dump() for w in WEBHOOKS],
            "recent_deliveries": [d.model_dump() for d in WEBHOOK_DELIVERIES],
        }
    )


def ingest_track_b_alert(
    alert_payload: dict[str, Any],
    aerial_verification: dict[str, Any] | None = None,
    title: str | None = None,
    country: str = "India",
    state: str = "Odisha",
    district: str = "Coastal Flood Corridor",
) -> EventFullResponse:
    """Ingest a verified Track B 19-field edge alert directly into Track A's economic/decision pipeline."""
    bbox_vals = alert_payload["bounding_box"]
    bbox = BoundingBox(
        min_lon=float(bbox_vals[0]),
        min_lat=float(bbox_vals[1]),
        max_lon=float(bbox_vals[2]),
        max_lat=float(bbox_vals[3]),
    )
    scene_date = str(alert_payload["scene_timestamp_utc"])[:10]
    sat_id = str(alert_payload["satellite_id"])
    peril = alert_payload["detected_peril"]
    req = EventRequest(
        preset_id=None,
        disaster_type=peril,
        title=title or f"Track B {sat_id} {peril.title()} Alert ? {district}",
        country=country,
        state=state,
        district=district,
        start_date=scene_date,
        end_date=scene_date,
        bbox=bbox,
        intensity_multiplier=1.0,
    )
    raw = build_raw_data_from_track_b(req, alert_payload, aerial_verification=aerial_verification)
    bundle = _run_pipeline(req, raw_override=raw)
    _emit_webhook(
        "event.processed",
        {
            "event_id": bundle.event.id,
            "source": "track_b_edge_telemetry",
            "satellite_id": sat_id,
            "disaster_type": bundle.event.disaster_type,
            "recommended_action_code": alert_payload.get("recommended_action_code"),
            "total_expected_loss_inr": bundle.loss.total_expected_loss_inr,
            "provisional_payout_inr": bundle.decisions.total_provisional_payout_inr,
        },
    )
    return bundle


async def ingest_track_b_endpoint(request: Request) -> JSONResponse:
    body = await request.json()
    alert_payload = body.get("edge_alert", body)
    aerial_verification = body.get("aerial_verification")
    bundle = ingest_track_b_alert(
        alert_payload,
        aerial_verification=aerial_verification,
        title=body.get("title"),
        country=body.get("country", "India"),
        state=body.get("state", "Odisha"),
        district=body.get("district", "Coastal Flood Corridor"),
    )
    return JSONResponse(bundle.model_dump())


async def register_webhook(request: Request) -> JSONResponse:
    raw = await request.json()
    sub = WebhookSubscription(
        webhook_id=raw.get("webhook_id") or f"WH-{uuid.uuid4().hex[:6].upper()}",
        partner_name=raw["partner_name"],
        url=raw["url"],
        events=raw.get("events", ["event.processed", "decision.updated"]),
        active=raw.get("active", True),
    )
    WEBHOOKS.append(sub)
    return JSONResponse(sub.model_dump())


def _build_api_routes(prefix: str = "") -> list[Route]:
    return [
        Route(f"{prefix}/health", health, methods=["GET"]),
        Route(f"{prefix}/events", list_events, methods=["GET"]),
        Route(f"{prefix}/events", create_or_run_event, methods=["POST"]),
        Route(f"{prefix}/events/{{event_id}}", get_event, methods=["GET"]),
        Route(f"{prefix}/events/{{event_id}}/damage", get_event_damage, methods=["GET"]),
        Route(f"{prefix}/events/{{event_id}}/exposure", get_event_exposure, methods=["GET"]),
        Route(f"{prefix}/events/{{event_id}}/loss", get_event_loss, methods=["GET"]),
        Route(f"{prefix}/events/{{event_id}}/decisions", get_event_decisions, methods=["GET"]),
        Route(
            f"{prefix}/events/{{event_id}}/decisions/{{decision_id}}/review",
            review_decision,
            methods=["POST"],
        ),
        Route(f"{prefix}/events/{{event_id}}/export", export_event, methods=["GET"]),
        Route(f"{prefix}/events/{{event_id}}/ask", ask_claimpulse, methods=["POST"]),
        Route(f"{prefix}/modules", get_modules, methods=["GET"]),
        Route(f"{prefix}/modules/register", register_module, methods=["POST"]),
        Route(f"{prefix}/webhooks", get_webhooks, methods=["GET"]),
        Route(f"{prefix}/webhooks", register_webhook, methods=["POST"]),
        Route(f"{prefix}/track-b/ingest", ingest_track_b_endpoint, methods=["POST"]),
    ]


routes: list[Any] = _build_api_routes("/api") + _build_api_routes("")

DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST_DIR.exists():
    routes.append(Mount("/", app=StaticFiles(directory=str(DIST_DIR), html=True), name="static"))

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)
