export type DisasterType = 'flood' | 'wildfire' | 'cyclone';
export type SeverityLevel = 'safe' | 'low' | 'medium' | 'high' | 'extreme';
export type SectorType = 'housing' | 'agriculture' | 'infrastructure' | 'business';
export type UserPersona = 'insurer' | 'government' | 'lender' | 'business' | 'humanitarian';

export interface BoundingBox {
  min_lon: number;
  min_lat: number;
  max_lon: number;
  max_lat: number;
}

export interface DatasetProvenance {
  dataset_name: string;
  provider: string;
  spatial_resolution: string;
  timestamp: string;
  license: string;
  confidence: number;
  is_demo: boolean;
  layer_id: string;
}

export interface EventSummary {
  id: string;
  preset_id: string | null;
  title: string;
  disaster_type: DisasterType;
  country: string;
  state: string;
  district: string;
  start_date: string;
  end_date: string;
  center: [number, number];
  zoom: number;
  bbox: BoundingBox;
  is_demo: boolean;
  status: string;
  overall_confidence: number;
  data_freshness: string;
  intensity_multiplier: number;
}

export interface HazardMetrics {
  affected_area_km2: number;
  max_intensity_label: string;
  mean_intensity_value: number;
  intensity_unit: string;
  severity_distribution_km2: Record<SeverityLevel, number>;
  datasets: DatasetProvenance[];
  timeline_dates: string[];
  imagery_comparison: {
    pre_event_date: string;
    post_event_date: string;
    pre_sensor: string;
    post_sensor: string;
    pre_tile_url: string;
    post_tile_url: string;
  };
}

export interface ExposureSummary {
  buildings_affected: number;
  population_exposed: number;
  roads_affected_km: number;
  farmland_affected_ha: number;
  critical_facilities_at_risk: number;
  total_exposed_value_inr: number;
  datasets: DatasetProvenance[];
}

export interface SectorLoss {
  sector: SectorType;
  label: string;
  expected_loss_inr: number;
  share_pct: number;
  asset_count: number;
  business_interruption_inr: number;
}

export interface TimeSeriesLossPoint {
  date: string;
  label: string;
  cumulative_loss_inr: number;
  affected_area_km2: number;
  confidence: number;
}

export interface PriorityZone {
  rank: number;
  zone_id: string;
  zone_name: string;
  district_or_ward: string;
  severity: SeverityLevel;
  expected_loss_inr: number;
  population_exposed: number;
  buildings_affected: number;
  vulnerability_score: number;
  infrastructure_criticality: number;
  recommended_relief_inr: number;
  priority_action: string;
}

export interface FormulaAssumption {
  parameter: string;
  value: string;
  rationale: string;
  source: string;
}

export interface StakeholderParameter {
  name: string;
  amount_inr: number;
  share_pct: number;
  driver_note: string;
}

export interface StakeholderPrediction {
  stakeholder: UserPersona;
  label: string;
  primary_metric_title: string;
  primary_amount_inr: number;
  uncertainty_low_inr: number;
  uncertainty_high_inr: number;
  secondary_metric_label: string;
  secondary_metric_value: string;
  headline: string;
  formula: string;
  parameters: StakeholderParameter[];
  zone_amounts_inr: Record<string, number>;
}

export interface LossEstimate {
  total_expected_loss_inr: number;
  business_interruption_inr: number;
  uncertainty_low_inr: number;
  uncertainty_high_inr: number;
  formula_text: string;
  by_sector: SectorLoss[];
  by_severity: Record<SeverityLevel, number>;
  time_series: TimeSeriesLossPoint[];
  top_10_zones: PriorityZone[];
  assumptions: FormulaAssumption[];
  limitations: string[];
  stakeholder_models: Record<string, StakeholderPrediction>;
}

export interface InsuranceClaimDecision {
  decision_id: string;
  policy_id: string;
  policyholder: string;
  asset_name: string;
  zone_name: string;
  sector: SectorType;
  severity: SeverityLevel;
  damage_probability: number;
  exposure_value_inr: number;
  estimated_claim_inr: number;
  uncertainty_low_inr: number;
  uncertainty_high_inr: number;
  confidence: number;
  recommendation: 'Auto-fast-track' | 'Manual review' | 'Investigate' | 'Insufficient evidence';
  review_status: 'Pending' | 'Approved' | 'Escalated' | 'Field Verification Requested';
  human_review_required: boolean;
  key_drivers: string[];
  calculation_breakdown: string;
  evidence_links: string[];
  data_timestamps: string[];
  reviewer_notes?: string | null;
}

export interface LenderCollateralAlert {
  loan_id: string;
  borrower: string;
  sector: SectorType;
  collateral_name: string;
  portfolio_exposure_inr: number;
  expected_collateral_loss_inr: number;
  ltv_stress_pct: number;
  risk_flag: 'Critical' | 'Elevated' | 'Watch';
  recommended_action: string;
  risk_premium_bps_adj: number;
  confidence: number;
}

export interface AuditRecord {
  audit_id: string;
  timestamp: string;
  actor: string;
  action: string;
  target_id: string;
  rationale: string;
  inputs_snapshot: Record<string, unknown>;
}

export interface DecisionBundle {
  recommended_action_headline: string;
  total_provisional_payout_inr: number;
  total_recommended_relief_inr: number;
  total_lender_exposure_at_risk_inr: number;
  overall_confidence: number;
  uncertainty_range_inr: [number, number];
  claims_fast_tracked_count: number;
  claims_manual_review_count: number;
  relief_priority_zones_count: number;
  claims: InsuranceClaimDecision[];
  relief_allocations: PriorityZone[];
  lender_alerts: LenderCollateralAlert[];
  audit_trail: AuditRecord[];
}

export interface GeoJsonFeatureCollection {
  type: 'FeatureCollection';
  features: Array<{
    type: 'Feature';
    id?: string;
    geometry: {
      type: 'Polygon' | 'Point' | 'LineString';
      coordinates: unknown;
    };
    properties: Record<string, unknown>;
  }>;
}

export interface EventFullResponse {
  event: EventSummary;
  damage: {
    geojson: GeoJsonFeatureCollection;
    metrics: HazardMetrics;
  };
  exposure: {
    geojson: GeoJsonFeatureCollection;
    summary: ExposureSummary;
  };
  loss: LossEstimate;
  decisions: DecisionBundle;
}

export interface AskCitation {
  layer: string;
  timestamp: string;
  metric: string;
  value: string;
}

export interface AskResponse {
  answer: string;
  confidence: number;
  formula_trace: string;
  citations: AskCitation[];
  insufficient_evidence: boolean;
}

export interface ConnectedModule {
  module_id: string;
  name: string;
  partner_team: string;
  category:
    | 'data_connector'
    | 'hazard_module'
    | 'exposure_module'
    | 'economic_model'
    | 'decision_workflow';
  description: string;
  status: 'active' | 'standby' | 'error';
  layer_id?: string | null;
  action_label?: string | null;
  kpis: Record<string, string>;
  webhook_url?: string | null;
}

export interface LayerVisibility {
  satelliteImagery: boolean;
  hazardExtent: boolean;
  buildings: boolean;
  population: boolean;
  roads: boolean;
  farmland: boolean;
  criticalInfra: boolean;
  insurancePortfolio: boolean;
  lossHeatmap: boolean;
}
