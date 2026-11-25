from __future__ import annotations

import re

from backend.core.schemas import AskCitation, AskResponse, EventFullResponse, EventRequest
from backend.modules.implementations import (
    BaseHazardModule,
    DemoSatelliteConnector,
    MultiStakeholderDecisionModule,
    StandardExposureModule,
    TransparentEconomicModel,
    fmt_inr,
)


def answer_question(
    bundle: EventFullResponse,
    question: str,
    event_req: EventRequest,
) -> AskResponse:
    """Strictly grounded retrieval & counterfactual engine over computed event results (in INR ₹)."""
    q = question.strip().lower()
    ev = bundle.event
    dmg_metrics = bundle.damage["metrics"]
    exp_summary = bundle.exposure["summary"]
    loss = bundle.loss
    dec = bundle.decisions

    haz_ds = dmg_metrics["datasets"][0]
    exp_ds = exp_summary["datasets"][0]

    # 1. Counterfactual depth / intensity sensitivity ("20% increase in flood depth", etc.)
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if any(k in q for k in ["increase", "depth", "what if", "sensitivity", "higher", "surge"]):
        pct = float(pct_match.group(1)) if pct_match else 20.0
        if "decrease" in q or "lower" in q or "drop" in q:
            new_mult = max(0.25, round(event_req.intensity_multiplier * (1.0 - pct / 100.0), 3))
            direction = f"-{pct:.0f}%"
        else:
            new_mult = min(3.0, round(event_req.intensity_multiplier * (1.0 + pct / 100.0), 3))
            direction = f"+{pct:.0f}%"

        connector = DemoSatelliteConnector()
        raw = connector.fetch(event_req)
        cf_req = event_req.model_copy(update={"intensity_multiplier": new_mult})
        haz_mod = BaseHazardModule(cf_req.disaster_type, "counterfactual-hazard")
        cf_dmg = haz_mod.compute_damage(cf_req, raw)
        cf_exp = StandardExposureModule().overlay_exposure(cf_req, cf_dmg, raw)
        cf_loss = TransparentEconomicModel().estimate_loss(cf_req, cf_dmg, cf_exp, raw)
        cf_dec = MultiStakeholderDecisionModule().evaluate_decisions(cf_req, cf_dmg, cf_exp, cf_loss, raw)

        delta_loss = cf_loss.total_expected_loss_inr - loss.total_expected_loss_inr
        delta_payout = cf_dec.total_provisional_payout_inr - dec.total_provisional_payout_inr
        pct_payout_change = (delta_payout / max(1.0, dec.total_provisional_payout_inr)) * 100.0

        return AskResponse(
            answer=(
                f"Simulating a {direction} hazard intensity shift (multiplier {event_req.intensity_multiplier:.2f}x → {new_mult:.2f}x) "
                f"increases total expected economic loss from {fmt_inr(loss.total_expected_loss_inr)} to "
                f"{fmt_inr(cf_loss.total_expected_loss_inr)} (+{fmt_inr(delta_loss)}). "
                f"Recommended provisional insurance payouts rise from {fmt_inr(dec.total_provisional_payout_inr)} to "
                f"{fmt_inr(cf_dec.total_provisional_payout_inr)} (+{fmt_inr(delta_payout)}, {pct_payout_change:+.1f}%), "
                f"with {cf_dec.claims_fast_tracked_count} fast-track claims and {cf_dec.claims_manual_review_count} human-review claims."
            ),
            confidence=round(dec.overall_confidence * 0.96, 2),
            formula_trace=(
                f"Re-evaluated Expected loss (₹) = min(1.0, intensity×{new_mult:.2f}/ref) × Exposure (₹) × Vulnerability × Damage_fn across {len(raw['zones'])} zones"
            ),
            citations=[
                AskCitation(
                    layer=haz_ds["dataset_name"],
                    timestamp=haz_ds["timestamp"],
                    metric="Counterfactual Total Loss",
                    value=fmt_inr(cf_loss.total_expected_loss_inr),
                ),
                AskCitation(
                    layer="ClaimPulse Decision Engine",
                    timestamp=dec.audit_trail[0].timestamp,
                    metric="Provisional Payout Delta",
                    value=f"+{fmt_inr(delta_payout)} ({pct_payout_change:+.1f}%)",
                ),
            ],
        )

    # 2. Highest expected loss areas / priority zones
    if any(k in q for k in ["highest", "top", "worst", "most loss", "priority zone"]):
        top3 = loss.top_10_zones[:3]
        summary_lines = "; ".join(
            f"#{z.rank} {z.zone_name} ({fmt_inr(z.expected_loss_inr)} loss, {z.population_exposed:,} exposed, {z.severity.upper()})"
            for z in top3
        )
        return AskResponse(
            answer=(
                f"The highest expected loss zones in {ev.title} are: {summary_lines}. "
                f"Together, the top 3 zones account for "
                f"{fmt_inr(sum(z.expected_loss_inr for z in top3))} of the {fmt_inr(loss.total_expected_loss_inr)} total loss "
                f"and receive {fmt_inr(sum(z.recommended_relief_inr for z in top3))} in recommended emergency relief."
            ),
            confidence=dec.overall_confidence,
            formula_trace=loss.formula_text,
            citations=[
                AskCitation(
                    layer=haz_ds["dataset_name"],
                    timestamp=haz_ds["timestamp"],
                    metric="Top Zone Expected Loss",
                    value=f"{top3[0].zone_name}: {fmt_inr(top3[0].expected_loss_inr)}",
                ),
                AskCitation(
                    layer=exp_ds["dataset_name"],
                    timestamp=exp_ds["timestamp"],
                    metric="Top Zone Population Exposed",
                    value=f"{top3[0].population_exposed:,} residents",
                ),
            ],
        )

    # 3. Fast-tracked claims
    if any(k in q for k in ["fast-track", "fast track", "fasttracked", "auto", "payout"]):
        ft = [c for c in dec.claims if c.recommendation == "Auto-fast-track"]
        if not ft:
            return AskResponse(
                answer=(
                    f"No claims currently meet the Auto-fast-track threshold (confidence ≥ 86% and claim ≤ ₹8.82 Cr) "
                    f"in {ev.title}; all {len(dec.claims)} claims are routed to Manual review / Investigate to prevent false approvals."
                ),
                confidence=dec.overall_confidence,
                formula_trace="Gate: severity in {high, extreme} AND confidence >= 0.86 AND estimated_claim <= ₹8.82 Cr",
                citations=[
                    AskCitation(
                        layer="ClaimPulse Decision Engine",
                        timestamp=dec.audit_trail[0].timestamp,
                        metric="Fast-tracked Count",
                        value="0 (Human review safeguard active)",
                    )
                ],
            )
        ft_str = "; ".join(
            f"{c.policy_id} ({c.asset_name}: {fmt_inr(c.estimated_claim_inr)}, {int(c.confidence*100)}% conf)"
            for c in ft[:4]
        )
        return AskResponse(
            answer=(
                f"{len(ft)} claims qualify for Auto-fast-track provisional payout totaling "
                f"{fmt_inr(sum(c.estimated_claim_inr for c in ft))}: {ft_str}."
            ),
            confidence=round(sum(c.confidence for c in ft) / len(ft), 2),
            formula_trace="Gate: severity in {high, extreme} AND confidence >= 0.86 AND estimated_claim <= ₹8.82 Cr",
            citations=[
                AskCitation(
                    layer=haz_ds["dataset_name"],
                    timestamp=haz_ds["timestamp"],
                    metric="Fast-track Claims",
                    value=f"{len(ft)} policies ({fmt_inr(sum(c.estimated_claim_inr for c in ft))})",
                ),
                AskCitation(
                    layer="ClaimPulse Underwriting Ledger",
                    timestamp=exp_ds["timestamp"],
                    metric="Lead Fast-track Policy",
                    value=f"{ft[0].policy_id}: {fmt_inr(ft[0].estimated_claim_inr)}",
                ),
            ],
        )

    # 4. Agricultural loss
    if any(k in q for k in ["agricult", "crop", "farm", "paddy", "timber"]):
        agri = next((s for s in loss.by_sector if s.sector == "agriculture"), None)
        if not agri:
            return AskResponse(
                answer="Insufficient evidence; recommend field verification.",
                confidence=0.35,
                formula_trace="No agriculture sector records found.",
                citations=[],
                insufficient_evidence=True,
            )
        return AskResponse(
            answer=(
                f"Estimated agricultural and cropland loss is {fmt_inr(agri.expected_loss_inr)} "
                f"({agri.share_pct:.1f}% of total loss), spanning {exp_summary['farmland_affected_ha']:,.0f} hectares "
                f"detected via FAO WorldCover overlaid with {haz_ds['dataset_name']}. "
                f"Uncertainty range is {fmt_inr(agri.expected_loss_inr*0.82)} – {fmt_inr(agri.expected_loss_inr*1.21)}."
            ),
            confidence=0.87,
            formula_trace="Expected loss (₹) = norm_intensity × farmland_exposure_inr × vulnerability × damage_fn(sector=agriculture, ×1.12)",
            citations=[
                AskCitation(
                    layer="FAO Global Land Cover & Cropland Mask",
                    timestamp=exp_summary["datasets"][2]["timestamp"],
                    metric="Farmland Affected",
                    value=f"{exp_summary['farmland_affected_ha']:,.0f} ha",
                ),
                AskCitation(
                    layer=haz_ds["dataset_name"],
                    timestamp=haz_ds["timestamp"],
                    metric="Agricultural Expected Loss",
                    value=f"{fmt_inr(agri.expected_loss_inr)} ({agri.share_pct:.1f}%)",
                ),
            ],
        )

    # 5. Critical infrastructure at risk
    if any(k in q for k in ["critical", "infrastructure", "hospital", "substation", "power", "port", "road"]):
        crit_feats = [
            f["properties"]
            for f in bundle.exposure["geojson"]["features"]
            if f["properties"].get("asset_type") == "critical_infrastructure"
            and f["properties"].get("severity") != "safe"
        ]
        infra_sec = next((s for s in loss.by_sector if s.sector == "infrastructure"), None)
        names = ", ".join(
            f"{c['name']} ({c['critical_category']}, {c['severity'].upper()})"
            for c in crit_feats[:4]
        )
        return AskResponse(
            answer=(
                f"{len(crit_feats)} critical infrastructure facilities and {exp_summary['roads_affected_km']} km of roads "
                f"are exposed, with total infrastructure loss estimated at "
                f"{fmt_inr(infra_sec.expected_loss_inr if infra_sec else 0)}. "
                f"Key at-risk facilities: {names}."
            ),
            confidence=0.90,
            formula_trace="Exposure intersection of OSM critical facility coordinates with hazard severity polygons",
            citations=[
                AskCitation(
                    layer="OpenStreetMap Critical Infrastructure",
                    timestamp=exp_ds["timestamp"],
                    metric="Facilities at Risk",
                    value=f"{len(crit_feats)} facilities, {exp_summary['roads_affected_km']} km roads",
                ),
                AskCitation(
                    layer=haz_ds["dataset_name"],
                    timestamp=haz_ds["timestamp"],
                    metric="Infrastructure Sector Loss",
                    value=fmt_inr(infra_sec.expected_loss_inr if infra_sec else 0),
                ),
            ],
        )

    # 6. Why flagged for manual review
    if any(k in q for k in ["flag", "review", "why", "investigate", "insufficient", "manual"]):
        flagged = [c for c in dec.claims if c.recommendation != "Auto-fast-track"]
        if not flagged:
            return AskResponse(
                answer="All active claims met automated fast-track thresholds; no manual review flags are pending.",
                confidence=dec.overall_confidence,
                formula_trace="All policies >= 86% confidence and <= ₹8.82 Cr.",
                citations=[],
            )
        examples = "; ".join(
            f"{c.policy_id} [{c.recommendation}]: {c.key_drivers[-1]} (est. {fmt_inr(c.estimated_claim_inr)}, {int(c.confidence*100)}% conf)"
            for c in flagged[:3]
        )
        return AskResponse(
            answer=(
                f"{len(flagged)} claims are flagged for human review, investigation, or additional evidence "
                f"(ClaimPulse never auto-denies claims). Triggers include claim values exceeding ₹8.82 Cr or SAR/optical confidence below 86%: {examples}."
            ),
            confidence=round(sum(c.confidence for c in flagged) / len(flagged), 2),
            formula_trace="Review policy: conf < 0.55 → Insufficient evidence; conf < 0.72 → Investigate; claim > ₹8.82 Cr or conf < 0.86 → Manual review",
            citations=[
                AskCitation(
                    layer="ClaimPulse Governance Policy",
                    timestamp=dec.audit_trail[0].timestamp,
                    metric="Flagged Claims",
                    value=f"{len(flagged)} of {len(dec.claims)} policies",
                ),
                AskCitation(
                    layer=haz_ds["dataset_name"],
                    timestamp=haz_ds["timestamp"],
                    metric="Sample Flagged Policy",
                    value=f"{flagged[0].policy_id} ({flagged[0].recommendation})",
                ),
            ],
        )

    # 7. Guardrail against ungrounded questions
    out_of_scope_terms = ["interior", "mold", "basement", "soil ph", "stock price", "ceo", "forecast next year", "2030", "exact roof"]
    if any(t in q for t in out_of_scope_terms):
        return AskResponse(
            answer="Insufficient evidence; recommend field verification.",
            confidence=0.25,
            formula_trace="Query references sub-surface, interior, or non-satellite attributes absent from loaded Earth observation and exposure layers.",
            citations=[
                AskCitation(
                    layer=haz_ds["dataset_name"],
                    timestamp=haz_ds["timestamp"],
                    metric="Coverage Scope",
                    value="Surface hazard & macro exposure only",
                )
            ],
            insufficient_evidence=True,
        )

    # 8. General grounded summary
    return AskResponse(
        answer=(
            f"For {ev.title} ({ev.start_date} to {ev.end_date}), satellite analysis detects {dmg_metrics['affected_area_km2']} km² affected, "
            f"{exp_summary['buildings_affected']:,} buildings, {exp_summary['population_exposed']:,} residents, and "
            f"{fmt_inr(loss.total_expected_loss_inr)} expected economic loss (uncertainty {fmt_inr(loss.uncertainty_low_inr)}–{fmt_inr(loss.uncertainty_high_inr)}). "
            f"Recommended action: {dec.recommended_action_headline}"
        ),
        confidence=dec.overall_confidence,
        formula_trace=loss.formula_text,
        citations=[
            AskCitation(
                layer=haz_ds["dataset_name"],
                timestamp=haz_ds["timestamp"],
                metric="Affected Area",
                value=f"{dmg_metrics['affected_area_km2']} km²",
            ),
            AskCitation(
                layer=exp_ds["dataset_name"],
                timestamp=exp_ds["timestamp"],
                metric="Total Expected Loss",
                value=fmt_inr(loss.total_expected_loss_inr),
            ),
        ],
    )
