from __future__ import annotations

import csv
import io
import json
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.core.schemas import EventFullResponse
from backend.modules.implementations import fmt_inr


def export_csv(bundle: EventFullResponse) -> str:
    """Export affected zones, claims, and asset exposure in INR as UTF-8 CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Record Type",
            "ID",
            "Name",
            "Zone",
            "Sector / Category",
            "Severity",
            "Hazard Intensity",
            "Confidence",
            "Exposure Value (INR)",
            "Expected Loss / Claim (INR)",
            "Uncertainty Low (INR)",
            "Uncertainty High (INR)",
            "Recommendation / Action",
            "Review Status",
        ]
    )

    for clm in bundle.decisions.claims:
        writer.writerow(
            [
                "Insurance Claim",
                clm.policy_id,
                clm.asset_name,
                clm.zone_name,
                clm.sector,
                clm.severity,
                "",
                f"{clm.confidence:.2f}",
                f"{clm.exposure_value_inr:.2f}",
                f"{clm.estimated_claim_inr:.2f}",
                f"{clm.uncertainty_low_inr:.2f}",
                f"{clm.uncertainty_high_inr:.2f}",
                clm.recommendation,
                clm.review_status,
            ]
        )

    for zone in bundle.loss.top_10_zones:
        writer.writerow(
            [
                "Priority Relief Zone",
                zone.zone_id,
                zone.zone_name,
                zone.district_or_ward,
                "Multi-sector",
                zone.severity,
                f"Vuln {zone.vulnerability_score:.2f}",
                f"{bundle.decisions.overall_confidence:.2f}",
                "",
                f"{zone.expected_loss_inr:.2f}",
                f"{zone.expected_loss_inr * 0.82:.2f}",
                f"{zone.expected_loss_inr * 1.21:.2f}",
                f"Relief {fmt_inr(zone.recommended_relief_inr, ascii_only=True)} - {zone.priority_action}",
                "Active",
            ]
        )

    for feat in bundle.exposure["geojson"]["features"]:
        props: dict[str, Any] = feat["properties"]
        if props.get("asset_type") in ("critical_infrastructure", "road", "farmland"):
            writer.writerow(
                [
                    props.get("asset_type"),
                    props.get("asset_id"),
                    props.get("name"),
                    props.get("zone_name"),
                    props.get("sector") or props.get("critical_category"),
                    props.get("severity"),
                    props.get("hazard_intensity"),
                    props.get("confidence"),
                    props.get("exposure_value_inr", ""),
                    "",
                    "",
                    "",
                    "Exposed Asset",
                    "Tracked",
                ]
            )

    return output.getvalue()


def export_geojson(bundle: EventFullResponse) -> str:
    """Export combined damage polygons and exposure features with INR economic metadata."""
    damage_features = bundle.damage["geojson"]["features"]
    zone_loss_map = {z.zone_id: z for z in bundle.loss.top_10_zones}
    enriched_features = []

    for feat in damage_features:
        props = dict(feat["properties"])
        zid = props.get("zone_id")
        if zid in zone_loss_map:
            pz = zone_loss_map[zid]
            props["expected_loss_inr"] = pz.expected_loss_inr
            props["recommended_relief_inr"] = pz.recommended_relief_inr
            props["priority_rank"] = pz.rank
        enriched_features.append({**feat, "properties": props})

    payload = {
        "type": "FeatureCollection",
        "name": f"ClaimPulse_{bundle.event.id}",
        "metadata": {
            "event_id": bundle.event.id,
            "title": bundle.event.title,
            "disaster_type": bundle.event.disaster_type,
            "currency": "INR",
            "total_expected_loss_inr": bundle.loss.total_expected_loss_inr,
            "overall_confidence": bundle.decisions.overall_confidence,
            "is_demo": bundle.event.is_demo,
        },
        "features": enriched_features + bundle.exposure["geojson"]["features"],
    }
    return json.dumps(payload, indent=2)


def export_pdf(bundle: EventFullResponse) -> bytes:
    """Generate a clean one-page executive Decision Brief PDF in INR (Rs.)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=28,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CPTitle",
        parent=styles["Heading1"],
        fontSize=15,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "CPSub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    sec_style = ParagraphStyle(
        "CPSec",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#0369A1"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "CPBody",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
    )

    ev = bundle.event
    dmg = bundle.damage["metrics"]
    exp = bundle.exposure["summary"]
    loss = bundle.loss
    dec = bundle.decisions

    headline_ascii = dec.recommended_action_headline.replace("₹", "Rs. ")
    formula_ascii = loss.formula_text.replace("₹", "INR")

    story: list[Any] = []
    story.append(Paragraph(f"ClaimPulse - Executive Decision Brief: {ev.title}", title_style))
    story.append(
        Paragraph(
            f"Event ID: {ev.id} | Type: {ev.disaster_type.upper()} | Window: {ev.start_date} to {ev.end_date} | "
            f"Currency: INR | Mode: {'DEMO SCENARIO' if ev.is_demo else 'LIVE'} | Confidence: {int(dec.overall_confidence*100)}%",
            sub_style,
        )
    )

    story.append(Paragraph("1. Recommended Economic Action (INR)", sec_style))
    story.append(
        Paragraph(
            f"<b>Headline:</b> {headline_ascii}<br/>"
            f"<b>Formula:</b> {formula_ascii}<br/>"
            f"<b>Uncertainty Envelope:</b> {fmt_inr(loss.uncertainty_low_inr, ascii_only=True)} - {fmt_inr(loss.uncertainty_high_inr, ascii_only=True)}",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    kpi_data = [
        ["Affected Area", "Buildings", "Pop. Exposed", "Expected Loss", "Provisional Payout", "Relief Allocation"],
        [
            f"{dmg['affected_area_km2']} km2",
            f"{exp['buildings_affected']:,}",
            f"{exp['population_exposed']:,}",
            fmt_inr(loss.total_expected_loss_inr, ascii_only=True),
            fmt_inr(dec.total_provisional_payout_inr, ascii_only=True),
            fmt_inr(dec.total_recommended_relief_inr, ascii_only=True),
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[78, 70, 82, 105, 102, 103])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(kpi_table)

    story.append(Paragraph("2. Loss Breakdown by Sector (INR)", sec_style))
    sec_rows = [["Sector", "Expected Loss (INR)", "Share", "Notes"]]
    for s in loss.by_sector:
        note = (
            f"Incl. {fmt_inr(s.business_interruption_inr, ascii_only=True)} interruption"
            if s.business_interruption_inr > 0
            else f"{s.asset_count:,} assets"
        )
        sec_rows.append([s.label, fmt_inr(s.expected_loss_inr, ascii_only=True), f"{s.share_pct:.1f}%", note])
    sec_table = Table(sec_rows, colWidths=[180, 125, 65, 170])
    sec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(sec_table)

    story.append(Paragraph("3. Top 5 Priority Zones & Claims Triage", sec_style))
    zone_rows = [["Rank", "Zone", "Severity", "Pop. Exposed", "Expected Loss", "Relief Allocation"]]
    for z in loss.top_10_zones[:5]:
        zone_rows.append(
            [
                f"#{z.rank}",
                z.zone_name,
                z.severity.upper(),
                f"{z.population_exposed:,}",
                fmt_inr(z.expected_loss_inr, ascii_only=True),
                fmt_inr(z.recommended_relief_inr, ascii_only=True),
            ]
        )
    z_table = Table(zone_rows, colWidths=[35, 190, 62, 78, 88, 87])
    z_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0369A1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    story.append(z_table)

    story.append(Paragraph("4. Data Provenance & Governance", sec_style))
    prov_lines = " | ".join(f"{d['dataset_name']} ({d['spatial_resolution']})" for d in dmg["datasets"])
    story.append(
        Paragraph(
            f"<b>Satellite Sources:</b> {prov_lines}<br/>"
            f"<b>Human-in-the-loop Policy:</b> Zero claims are auto-denied. "
            f"{dec.claims_fast_tracked_count} claims fast-tracked; {dec.claims_manual_review_count} flagged for human review.",
            body_style,
        )
    )

    doc.build(story)
    return buf.getvalue()
