from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schemas import (
    ConnectedModule,
    DecisionBundle,
    DisasterType,
    EventRequest,
    LossEstimate,
)


class DataConnector(ABC):
    """Extension point for satellite, weather, insurance, IoT, drone, or government datasets."""

    connector_id: str
    provider: str

    @abstractmethod
    def fetch(self, event_req: EventRequest) -> dict[str, Any]:
        """Fetch raw or cached spatial/tabular data for the given event request."""


class HazardModule(ABC):
    """Extension point for hazard models: flood, wildfire, cyclone, earthquake, landslide, drought."""

    disaster_type: DisasterType
    module_name: str

    @abstractmethod
    def compute_damage(
        self, event_req: EventRequest, connector_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Return GeoJSON FeatureCollection + HazardMetrics summary."""


class ExposureModule(ABC):
    """Extension point for overlaying buildings, crops, roads, power, ports, population, businesses."""

    module_name: str

    @abstractmethod
    def overlay_exposure(
        self,
        event_req: EventRequest,
        damage_result: dict[str, Any],
        connector_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return GeoJSON FeatureCollection of exposed assets + ExposureSummary."""


class EconomicModel(ABC):
    """Extension point for economic valuation: insurance payout, relief, loan risk, business interruption."""

    model_name: str

    @abstractmethod
    def estimate_loss(
        self,
        event_req: EventRequest,
        damage_result: dict[str, Any],
        exposure_result: dict[str, Any],
    ) -> LossEstimate:
        """Compute sector losses, uncertainty bounds, top priority zones, and assumptions."""


class DecisionWorkflowModule(ABC):
    """Extension point for claim approval, relief disbursement, alerting, field inspection, verification."""

    workflow_name: str

    @abstractmethod
    def evaluate_decisions(
        self,
        event_req: EventRequest,
        damage_result: dict[str, Any],
        exposure_result: dict[str, Any],
        loss_estimate: LossEstimate,
    ) -> DecisionBundle:
        """Produce insurance triage, government relief ranking, lender alerts, and audit records."""


class PlatformRegistry:
    """Central registry enabling partner teams to plug in connectors, modules, and UI extensions."""

    def __init__(self) -> None:
        self.connectors: dict[str, DataConnector] = {}
        self.hazard_modules: dict[DisasterType, HazardModule] = {}
        self.exposure_modules: dict[str, ExposureModule] = {}
        self.economic_models: dict[str, EconomicModel] = {}
        self.decision_workflows: dict[str, DecisionWorkflowModule] = {}
        self.connected_ui_modules: dict[str, ConnectedModule] = {}

    def register_connector(self, connector: DataConnector) -> None:
        self.connectors[connector.connector_id] = connector

    def register_hazard(self, module: HazardModule) -> None:
        self.hazard_modules[module.disaster_type] = module

    def register_exposure(self, module: ExposureModule) -> None:
        self.exposure_modules[module.module_name] = module

    def register_economic_model(self, model: EconomicModel) -> None:
        self.economic_models[model.model_name] = model

    def register_decision_workflow(self, workflow: DecisionWorkflowModule) -> None:
        self.decision_workflows[workflow.workflow_name] = workflow

    def register_ui_module(self, module: ConnectedModule) -> None:
        self.connected_ui_modules[module.module_id] = module

    def list_ui_modules(self) -> list[ConnectedModule]:
        return list(self.connected_ui_modules.values())
