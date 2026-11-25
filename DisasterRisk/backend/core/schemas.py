from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Optional


DisasterType = Literal["flood", "wildfire", "cyclone"]
SeverityLevel = Literal["safe", "low", "medium", "high", "extreme"]
AssetType = Literal[
    "building",
    "road",
    "farmland",
    "critical_infrastructure",
    "population_cell",
    "policy_zone",
]
SectorType = Literal["housing", "agriculture", "infrastructure", "business"]
ClaimRecommendation = Literal[
    "Auto-fast-track",
    "Manual review",
    "Investigate",
    "Insufficient evidence",
]
ReviewStatus = Literal[
    "Pending",
    "Approved",
    "Escalated",
    "Field Verification Requested",
]


class SchemaModel:
    """Pure-Python dataclass base providing JSON-ready serialization helpers."""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def model_copy(self, update: dict[str, Any] | None = None) -> Any:
        if not update:
            return replace(self)  # type: ignore[arg-type]
        return replace(self, **update)  # type: ignore[arg-type]


@dataclass
class BoundingBox(SchemaModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


@dataclass
class DatasetProvenance(SchemaModel):
    dataset_name: str
    provider: str
    spatial_resolution: str
    timestamp: str
    license: str
    confidence: float
    is_demo: bool = True
    layer_id: str = ""


@dataclass
class EventRequest(SchemaModel):
    preset_id: Optional[str] = None
    disaster_type: DisasterType = "flood"
    title: Optional[str] = None
    country: str = "India"
    state: str = "Odisha"
    district: str = "Puri & Kendrapara"
    start_date: str = "2026-10-18"
    end_date: str = "2026-10-26"
    bbox: Optional[BoundingBox] = None
    intensity_multiplier: float = 1.0


@dataclass
class EventSummary(SchemaModel):
    id: str
    preset_id: Optional[str]
    title: str
    disaster_type: DisasterType
    country: str
    state: str
    district: str
    start_date: str
    end_date: str
    center: list[float]
    zoom: float
    bbox: BoundingBox
    is_demo: bool = True
    status: str = "processed"
    overall_confidence: float = 0.9
    data_freshness: str = ""
    intensity_multiplier: float = 1.0


@dataclass
class HazardMetrics(SchemaModel):
    affected_area_km2: float
    max_intensity_label: str
    mean_intensity_value: float
    intensity_unit: str
    severity_distribution_km2: dict[SeverityLevel, float]
    datasets: list[DatasetProvenance]
    timeline_dates: list[str]
    imagery_comparison: dict[str, str]


@dataclass
class ExposureSummary(SchemaModel):
    buildings_affected: int
    population_exposed: int
    roads_affected_km: float
    farmland_affected_ha: float
    critical_facilities_at_risk: int
    total_exposed_value_inr: float
    datasets: list[DatasetProvenance]


@dataclass
class FormulaAssumption(SchemaModel):
    parameter: str
    value: str
    rationale: str
    source: str


@dataclass
class SectorLoss(SchemaModel):
    sector: SectorType
    label: str
    expected_loss_inr: float
    share_pct: float
    asset_count: int
    business_interruption_inr: float = 0.0


@dataclass
class TimeSeriesLossPoint(SchemaModel):
    date: str
    label: str
    cumulative_loss_inr: float
    affected_area_km2: float
    confidence: float


@dataclass
class PriorityZone(SchemaModel):
    rank: int
    zone_id: str
    zone_name: str
    district_or_ward: str
    severity: SeverityLevel
    expected_loss_inr: float
    population_exposed: int
    buildings_affected: int
    vulnerability_score: float
    infrastructure_criticality: float
    recommended_relief_inr: float
    priority_action: str


@dataclass
class StakeholderParameter(SchemaModel):
    name: str
    amount_inr: float
    share_pct: float
    driver_note: str


@dataclass
class StakeholderPrediction(SchemaModel):
    stakeholder: Literal["insurer", "government", "humanitarian", "lender", "business"]
    label: str
    primary_metric_title: str
    primary_amount_inr: float
    uncertainty_low_inr: float
    uncertainty_high_inr: float
    secondary_metric_label: str
    secondary_metric_value: str
    headline: str
    formula: str
    parameters: list[StakeholderParameter]
    zone_amounts_inr: dict[str, float]


@dataclass
class LossEstimate(SchemaModel):
    total_expected_loss_inr: float
    business_interruption_inr: float
    uncertainty_low_inr: float
    uncertainty_high_inr: float
    formula_text: str
    by_sector: list[SectorLoss]
    by_severity: dict[SeverityLevel, float]
    time_series: list[TimeSeriesLossPoint]
    top_10_zones: list[PriorityZone]
    assumptions: list[FormulaAssumption]
    limitations: list[str]
    stakeholder_models: dict[str, StakeholderPrediction] = field(default_factory=dict)


@dataclass
class InsuranceClaimDecision(SchemaModel):
    decision_id: str
    policy_id: str
    policyholder: str
    asset_name: str
    zone_name: str
    sector: SectorType
    severity: SeverityLevel
    damage_probability: float
    exposure_value_inr: float
    estimated_claim_inr: float
    uncertainty_low_inr: float
    uncertainty_high_inr: float
    confidence: float
    recommendation: ClaimRecommendation
    review_status: ReviewStatus = "Pending"
    human_review_required: bool = True
    key_drivers: list[str] = field(default_factory=list)
    calculation_breakdown: str = ""
    evidence_links: list[str] = field(default_factory=list)
    data_timestamps: list[str] = field(default_factory=list)
    reviewer_notes: Optional[str] = None


@dataclass
class LenderCollateralAlert(SchemaModel):
    loan_id: str
    borrower: str
    sector: SectorType
    collateral_name: str
    portfolio_exposure_inr: float
    expected_collateral_loss_inr: float
    ltv_stress_pct: float
    risk_flag: Literal["Critical", "Elevated", "Watch"]
    recommended_action: str
    risk_premium_bps_adj: int
    confidence: float


@dataclass
class AuditRecord(SchemaModel):
    audit_id: str
    timestamp: str
    actor: str
    action: str
    target_id: str
    rationale: str
    inputs_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionBundle(SchemaModel):
    recommended_action_headline: str
    total_provisional_payout_inr: float
    total_recommended_relief_inr: float
    total_lender_exposure_at_risk_inr: float
    overall_confidence: float
    uncertainty_range_inr: list[float]
    claims_fast_tracked_count: int
    claims_manual_review_count: int
    relief_priority_zones_count: int
    claims: list[InsuranceClaimDecision]
    relief_allocations: list[PriorityZone]
    lender_alerts: list[LenderCollateralAlert]
    audit_trail: list[AuditRecord]


@dataclass
class ReviewDecisionRequest(SchemaModel):
    action: Literal["Approve", "Escalate", "Request Field Verification"]
    reviewer: str = "Senior Claims Analyst"
    notes: str = ""


@dataclass
class AskRequest(SchemaModel):
    question: str


@dataclass
class AskCitation(SchemaModel):
    layer: str
    timestamp: str
    metric: str
    value: str


@dataclass
class AskResponse(SchemaModel):
    answer: str
    confidence: float
    formula_trace: str
    citations: list[AskCitation]
    insufficient_evidence: bool = False


@dataclass
class ConnectedModule(SchemaModel):
    module_id: str
    name: str
    partner_team: str
    category: Literal[
        "data_connector",
        "hazard_module",
        "exposure_module",
        "economic_model",
        "decision_workflow",
    ]
    description: str
    status: Literal["active", "standby", "error"] = "active"
    layer_id: Optional[str] = None
    action_label: Optional[str] = None
    kpis: dict[str, str] = field(default_factory=dict)
    webhook_url: Optional[str] = None


@dataclass
class WebhookSubscription(SchemaModel):
    webhook_id: str
    partner_name: str
    url: str
    events: list[str]
    active: bool = True


@dataclass
class WebhookDelivery(SchemaModel):
    delivery_id: str
    webhook_id: str
    event_type: str
    timestamp: str
    status: str
    payload_preview: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventFullResponse(SchemaModel):
    event: EventSummary
    damage: dict[str, Any]
    exposure: dict[str, Any]
    loss: LossEstimate
    decisions: DecisionBundle
