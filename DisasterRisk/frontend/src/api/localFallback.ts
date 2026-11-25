import bundledRaw from '../data/bundledPresets.json';
import type {
  AskResponse,
  BoundingBox,
  ConnectedModule,
  DisasterType,
  EventFullResponse,
  EventSummary,
} from '../types/claimpulse';
import { formatInr, formatNumber } from '../utils/format';

interface BundledData {
  presets: Record<string, EventFullResponse>;
  modules: ConnectedModule[];
}

const bundled = bundledRaw as unknown as BundledData;

const localEvents: Record<string, EventFullResponse> = structuredClone(bundled.presets);
const localModules: ConnectedModule[] = structuredClone(bundled.modules);

export function fallbackListEvents(): EventSummary[] {
  return Object.values(localEvents).map((b) => b.event);
}

export function fallbackGetEvent(eventId: string): EventFullResponse {
  return localEvents[eventId] || localEvents['odisha-flood-2026'];
}

export function fallbackRunEvent(params: {
  preset_id?: string | null;
  disaster_type: DisasterType;
  title?: string;
  country?: string;
  state?: string;
  district?: string;
  start_date: string;
  end_date: string;
  bbox?: BoundingBox;
  intensity_multiplier?: number;
}): EventFullResponse {
  const mult = params.intensity_multiplier ?? 1.0;
  const baseKey =
    params.preset_id && localEvents[params.preset_id]
      ? params.preset_id
      : Object.keys(localEvents).find(
          (k) => localEvents[k].event.disaster_type === params.disaster_type
        ) || 'odisha-flood-2026';

  const base = structuredClone(localEvents[baseKey]);
  const scaleFactor = Math.min(1.85, Math.max(0.55, mult));

  base.event.id =
    Math.abs(mult - 1.0) > 0.01 ? `${base.event.id}-x${Math.round(mult * 100)}` : base.event.id;
  base.event.intensity_multiplier = mult;
  if (params.district) base.event.district = params.district;
  if (params.start_date) base.event.start_date = params.start_date;
  if (params.end_date) base.event.end_date = params.end_date;

  // Scale losses & payouts smoothly in fallback mode
  base.loss.total_expected_loss_inr = Math.round(base.loss.total_expected_loss_inr * scaleFactor);
  base.loss.business_interruption_inr = Math.round(
    base.loss.business_interruption_inr * scaleFactor
  );
  base.loss.uncertainty_low_inr = Math.round(base.loss.total_expected_loss_inr * 0.82);
  base.loss.uncertainty_high_inr = Math.round(base.loss.total_expected_loss_inr * 1.21);
  base.loss.by_sector = base.loss.by_sector.map((s) => ({
    ...s,
    expected_loss_inr: Math.round(s.expected_loss_inr * scaleFactor),
    business_interruption_inr: Math.round(s.business_interruption_inr * scaleFactor),
  }));
  base.loss.top_10_zones = base.loss.top_10_zones.map((z) => ({
    ...z,
    expected_loss_inr: Math.round(z.expected_loss_inr * scaleFactor),
    recommended_relief_inr: Math.round(z.recommended_relief_inr * scaleFactor),
  }));
  if (base.loss.stakeholder_models) {
    Object.values(base.loss.stakeholder_models).forEach((sm) => {
      sm.primary_amount_inr = Math.round(sm.primary_amount_inr * scaleFactor);
      sm.uncertainty_low_inr = Math.round(sm.uncertainty_low_inr * scaleFactor);
      sm.uncertainty_high_inr = Math.round(sm.uncertainty_high_inr * scaleFactor);
      sm.parameters = sm.parameters.map((p) => ({
        ...p,
        amount_inr: Math.round(p.amount_inr * scaleFactor),
      }));
      Object.keys(sm.zone_amounts_inr).forEach((zid) => {
        sm.zone_amounts_inr[zid] = Math.round(sm.zone_amounts_inr[zid] * scaleFactor);
      });
    });
  }
  base.decisions.claims = base.decisions.claims.map((c) => ({
    ...c,
    estimated_claim_inr: Math.round(c.estimated_claim_inr * scaleFactor),
    uncertainty_low_inr: Math.round(c.uncertainty_low_inr * scaleFactor),
    uncertainty_high_inr: Math.round(c.uncertainty_high_inr * scaleFactor),
  }));
  base.decisions.total_provisional_payout_inr = Math.round(
    base.decisions.total_provisional_payout_inr * scaleFactor
  );
  base.decisions.total_recommended_relief_inr = Math.round(
    base.decisions.total_recommended_relief_inr * scaleFactor
  );
  base.decisions.recommended_action_headline = `Release ${formatInr(
    base.decisions.total_provisional_payout_inr
  )} provisional insurance liquidity (${base.decisions.claims_fast_tracked_count} fast-track) & allocate ${formatInr(
    base.decisions.total_recommended_relief_inr
  )} emergency relief across top ${base.loss.top_10_zones.length} priority zones.`;

  localEvents[base.event.id] = base;
  return base;
}

export function fallbackReviewDecision(
  eventId: string,
  decisionId: string,
  action: 'Approve' | 'Escalate' | 'Request Field Verification',
  reviewer: string,
  notes: string
): EventFullResponse {
  const bundle = fallbackGetEvent(eventId);
  const claim = bundle.decisions.claims.find((c) => c.decision_id === decisionId);
  if (claim) {
    claim.review_status =
      action === 'Approve'
        ? 'Approved'
        : action === 'Escalate'
        ? 'Escalated'
        : 'Field Verification Requested';
    if (action === 'Approve') {
      claim.recommendation = 'Auto-fast-track';
      claim.human_review_required = false;
    } else if (action === 'Request Field Verification') {
      claim.recommendation = 'Investigate';
    }
    if (notes) claim.reviewer_notes = notes;
    bundle.decisions.claims_fast_tracked_count = bundle.decisions.claims.filter(
      (c) => c.recommendation === 'Auto-fast-track'
    ).length;
    bundle.decisions.claims_manual_review_count =
      bundle.decisions.claims.length - bundle.decisions.claims_fast_tracked_count;
    bundle.decisions.audit_trail.unshift({
      audit_id: `AUD-LOCAL-${Date.now().toString().slice(-4)}`,
      timestamp: new Date().toISOString(),
      actor: reviewer,
      action: `Human Review: ${action} (${claim.policy_id})`,
      target_id: decisionId,
      rationale: notes || `Analyst executed ${action} on ${claim.asset_name}.`,
      inputs_snapshot: { estimated_claim_inr: claim.estimated_claim_inr },
    });
  }
  return structuredClone(bundle);
}

export function fallbackAskClaimPulse(eventId: string, question: string): AskResponse {
  const bundle = fallbackGetEvent(eventId);
  const q = question.toLowerCase();
  const haz = bundle.damage.metrics.datasets[0];
  const expDs = bundle.exposure.summary.datasets[0];

  if (q.includes('increase') || q.includes('depth') || q.includes('20%')) {
    const cfLoss = bundle.loss.total_expected_loss_inr * 1.26;
    const cfPayout = bundle.decisions.total_provisional_payout_inr * 1.24;
    return {
      answer: `Simulating a +20% hazard intensity increase raises total expected loss from ${formatInr(
        bundle.loss.total_expected_loss_inr
      )} to ${formatInr(cfLoss)} and increases provisional payouts from ${formatInr(
        bundle.decisions.total_provisional_payout_inr
      )} to ${formatInr(cfPayout)}.`,
      confidence: 0.81,
      formula_trace: bundle.loss.formula_text,
      citations: [
        {
          layer: haz.dataset_name,
          timestamp: haz.timestamp,
          metric: 'Projected Total Loss',
          value: formatInr(cfLoss),
        },
      ],
      insufficient_evidence: false,
    };
  }

  if (q.includes('highest') || q.includes('top') || q.includes('worst')) {
    const top3 = bundle.loss.top_10_zones.slice(0, 3);
    return {
      answer: `Top highest-loss zones: ${top3
        .map((z) => `#${z.rank} ${z.zone_name} (${formatInr(z.expected_loss_inr)})`)
        .join('; ')}.`,
      confidence: bundle.decisions.overall_confidence,
      formula_trace: bundle.loss.formula_text,
      citations: [
        {
          layer: haz.dataset_name,
          timestamp: haz.timestamp,
          metric: 'Top Priority Zone',
          value: `${top3[0].zone_name}: ${formatInr(top3[0].expected_loss_inr)}`,
        },
      ],
      insufficient_evidence: false,
    };
  }

  if (q.includes('fast-track') || q.includes('fast track')) {
    const ft = bundle.decisions.claims.filter((c) => c.recommendation === 'Auto-fast-track');
    return {
      answer: `${ft.length} claims qualify for Auto-fast-track totaling ${formatInr(
        ft.reduce((s, c) => s + c.estimated_claim_inr, 0)
      )}: ${ft.map((c) => `${c.policy_id} (${formatInr(c.estimated_claim_inr)})`).join(', ')}.`,
      confidence: 0.9,
      formula_trace: 'Gate: severity in {high, extreme} AND confidence >= 86% AND claim <= ₹8.82 Cr',
      citations: [
        {
          layer: 'ClaimPulse Decision Engine',
          timestamp: bundle.decisions.audit_trail[0]?.timestamp || haz.timestamp,
          metric: 'Fast-tracked Policies',
          value: `${ft.length} policies`,
        },
      ],
      insufficient_evidence: false,
    };
  }

  if (q.includes('agricult') || q.includes('crop') || q.includes('farm')) {
    const agri = bundle.loss.by_sector.find((s) => s.sector === 'agriculture');
    return {
      answer: `Estimated agricultural loss is ${formatInr(
        agri?.expected_loss_inr || 0
      )} (${agri?.share_pct || 0}% of total loss) across ${formatNumber(
        bundle.exposure.summary.farmland_affected_ha
      )} ha of farmland.`,
      confidence: 0.87,
      formula_trace: bundle.loss.formula_text,
      citations: [
        {
          layer: 'FAO Global Land Cover & Cropland Mask',
          timestamp: expDs.timestamp,
          metric: 'Farmland Affected',
          value: `${formatNumber(bundle.exposure.summary.farmland_affected_ha)} ha`,
        },
      ],
      insufficient_evidence: false,
    };
  }

  if (q.includes('critical') || q.includes('infrastructure') || q.includes('hospital')) {
    const infra = bundle.loss.by_sector.find((s) => s.sector === 'infrastructure');
    return {
      answer: `${bundle.exposure.summary.critical_facilities_at_risk} critical facilities and ${bundle.exposure.summary.roads_affected_km} km of roads are exposed, with infrastructure loss estimated at ${formatInr(
        infra?.expected_loss_inr || 0
      )}.`,
      confidence: 0.9,
      formula_trace: 'OSM critical infrastructure intersection with hazard extent polygons',
      citations: [
        {
          layer: expDs.dataset_name,
          timestamp: expDs.timestamp,
          metric: 'Critical Facilities',
          value: `${bundle.exposure.summary.critical_facilities_at_risk} at risk`,
        },
      ],
      insufficient_evidence: false,
    };
  }

  if (q.includes('flag') || q.includes('review') || q.includes('why')) {
    const flagged = bundle.decisions.claims.filter((c) => c.recommendation !== 'Auto-fast-track');
    return {
      answer: `${flagged.length} claims are flagged for human review, investigation, or extra evidence (zero auto-denials) because claim value exceeds ₹8.82 Cr or satellite confidence is below 86%.`,
      confidence: 0.84,
      formula_trace: 'conf < 0.55 → Insufficient evidence; conf < 0.72 → Investigate; claim > ₹8.82 Cr or conf < 0.86 → Manual review',
      citations: [
        {
          layer: 'ClaimPulse Governance Policy',
          timestamp: haz.timestamp,
          metric: 'Flagged Policies',
          value: `${flagged.length} of ${bundle.decisions.claims.length}`,
        },
      ],
      insufficient_evidence: false,
    };
  }

  return {
    answer: `For ${bundle.event.title}, total expected loss is ${formatInr(
      bundle.loss.total_expected_loss_inr
    )} across ${bundle.damage.metrics.affected_area_km2} km², ${formatNumber(
      bundle.exposure.summary.buildings_affected
    )} buildings, and ${formatNumber(bundle.exposure.summary.population_exposed)} residents.`,
    confidence: bundle.decisions.overall_confidence,
    formula_trace: bundle.loss.formula_text,
    citations: [
      {
        layer: haz.dataset_name,
        timestamp: haz.timestamp,
        metric: 'Total Expected Loss',
        value: formatInr(bundle.loss.total_expected_loss_inr),
      },
    ],
    insufficient_evidence: false,
  };
}

export function fallbackGetModules(): ConnectedModule[] {
  return [...localModules];
}

export function fallbackRegisterModule(
  mod: Omit<ConnectedModule, 'module_id' | 'status'>
): ConnectedModule {
  const created: ConnectedModule = {
    ...mod,
    module_id: `mod-${Date.now().toString().slice(-5)}`,
    status: 'active',
  };
  localModules.push(created);
  return created;
}

export function triggerClientSideDownload(
  bundle: EventFullResponse,
  format: 'csv' | 'geojson'
): void {
  let content = '';
  let mime = 'text/csv;charset=utf-8;';
  let filename = `claimpulse_${bundle.event.id}.csv`;

  if (format === 'geojson') {
    mime = 'application/geo+json;charset=utf-8;';
    filename = `claimpulse_${bundle.event.id}.geojson`;
    content = JSON.stringify(
      {
        type: 'FeatureCollection',
        metadata: {
          event_id: bundle.event.id,
          currency: 'INR',
          total_expected_loss_inr: bundle.loss.total_expected_loss_inr,
        },
        features: [
          ...bundle.damage.geojson.features,
          ...bundle.exposure.geojson.features,
        ],
      },
      null,
      2
    );
  } else {
    const rows = [
      ['Type', 'ID', 'Name', 'Zone', 'Severity', 'Expected Loss / Claim (INR)', 'Recommendation'],
      ...bundle.decisions.claims.map((c) => [
        'Insurance Claim',
        c.policy_id,
        c.asset_name,
        c.zone_name,
        c.severity,
        String(c.estimated_claim_inr),
        c.recommendation,
      ]),
    ];
    content = rows.map((r) => r.map((v) => `"${v.replace(/"/g, '""')}"`).join(',')).join('\n');
  }

  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
