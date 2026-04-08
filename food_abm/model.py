"""
food_abm/model.py

Mesa model for a regional food supply chain ABM focused on SMEs in Uusimaa.

This module is designed to work with:
- food_abm/agents.py
- food_abm/archetypes.py
- food_abm/scenarios.py
- app.py

Design principles:
- Clear and extendable over highly detailed.
- Scenario-driven configuration.
- Easy integration with Streamlit and batch experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import copy

import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector

from .agents import (
    SMEAgent,
    RetailBlock,
    LogisticsOperator,
    NeutralHub,
    PublicBuyer,
    HorecaChannel,
)
from .archetypes import SME_ARCHETYPES
from .scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Scenario representation
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    Lightweight scenario object used by the model and UI.

    Parameters are intentionally generic. You can later expand this dataclass
    when the model grows more detailed.
    """
    name: str = "baseline"
    neutral_hub_enabled: bool = False
    public_buyer_enabled: bool = False
    horeca_enabled: bool = True
    fuel_cost_multiplier: float = 1.0
    retail_access_threshold: float = 0.65


def scenario_from_any(scenario: Any) -> ScenarioConfig:
    """
    Convert supported scenario inputs into a ScenarioConfig.
    Accepted forms:
    - ScenarioConfig instance
    - scenario name string found in SCENARIOS
    - dict with scenario fields
    """
    if isinstance(scenario, ScenarioConfig):
        return copy.deepcopy(scenario)

    if isinstance(scenario, str):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario name: {scenario}")
        raw = SCENARIOS[scenario]
        return ScenarioConfig(
            name=scenario,
            neutral_hub_enabled=raw.get("neutral_hub_enabled", False),
            public_buyer_enabled=raw.get("public_buyer_enabled", False),
            horeca_enabled=raw.get("horeca_enabled", True),
            fuel_cost_multiplier=raw.get("fuel_cost_multiplier", 1.0),
            retail_access_threshold=raw.get("retail_access_threshold", 0.65),
        )

    if isinstance(scenario, dict):
        return ScenarioConfig(
            name=scenario.get("name", "custom"),
            neutral_hub_enabled=scenario.get("neutral_hub_enabled", False),
            public_buyer_enabled=scenario.get("public_buyer_enabled", False),
            horeca_enabled=scenario.get("horeca_enabled", True),
            fuel_cost_multiplier=scenario.get("fuel_cost_multiplier", 1.0),
            retail_access_threshold=scenario.get("retail_access_threshold", 0.65),
        )

    raise TypeError("Scenario must be ScenarioConfig, str, or dict.")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FoodSupplyModel(Model):
    """
    Main Mesa model for the food supply chain prototype.

    Key actor groups:
    - SMEs
    - two retail blocks
    - one logistics operator
    - optional neutral hub
    - optional public buyer
    - optional HORECA channel

    Notes:
    - This prototype does not use a spatial grid.
    - Agents are tracked in plain lists.
    - Scheduling is explicit: first reset infrastructure, then step SMEs, while
      other agents remain mostly passive evaluators in this version.
    """

    def __init__(
        self,
        n_smes: int = 36,
        scenario: Any = "baseline",
        seed: Optional[int] = None,
        archetype_shares: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(seed=seed)

        self.num_smes = int(n_smes)
        self.scenario = scenario_from_any(scenario)

        # Actor containers
        self.smes: List[SMEAgent] = []
        self.retail_blocks: List[RetailBlock] = []
        self.logistics_operator: Optional[LogisticsOperator] = None
        self.neutral_hub: Optional[NeutralHub] = None
        self.public_buyer: Optional[PublicBuyer] = None
        self.horeca_channel: Optional[HorecaChannel] = None

        # Archetype composition
        self.archetype_shares = archetype_shares or {
            "greenhouse_vegetable_sme": 0.40,
            "farm_shop_restaurant_sme": 0.35,
            "fish_farming_sme": 0.25,
        }

        # Bookkeeping
        self.current_step = 0
        self.running = True

        # Build the world
        self._build_channels()
        self._build_smes()

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "AvgProfit": lambda m: m.avg_profit(),
                "AvgLogisticsCost": lambda m: m.avg_logistics_cost(),
                "AvgDependency": lambda m: m.avg_dependency(),
                "SurvivalRate": lambda m: m.survival_rate(),
                "AvgChannelCount": lambda m: m.avg_channel_count(),
                "TotalEmissions": lambda m: m.total_emissions(),
                "Scenario": lambda m: m.scenario.name,
                "Step": lambda m: m.current_step,
            },
            agent_reporters={
                "Archetype": lambda a: getattr(a, "archetype", None),
                "State": lambda a: getattr(a, "state", None),
                "Alive": lambda a: getattr(a, "is_alive", None),
                "LastProfit": lambda a: getattr(a, "last_profit", None),
                "LastLogisticsCost": lambda a: getattr(a, "last_logistics_cost", None),
                "LastEmissions": lambda a: getattr(a, "last_emissions", None),
                "LastChannel": lambda a: getattr(a, "last_channel_name", None),
                "LastBuyer": lambda a: getattr(a, "last_buyer_name", None),
                "LastDependencyRatio": lambda a: getattr(a, "last_dependency_ratio", None),
                "DistanceToMarketKm": lambda a: getattr(a, "distance_to_market_km", None),
            },
        )

        # Capture step 0 state
        self.datacollector.collect(self)

    # ---------------------------------------------------------------------
    # Build model actors
    # ---------------------------------------------------------------------

    def _build_channels(self) -> None:
        """
        Create non-SME actors and apply scenario-dependent parameters.
        """

        # Two dominant retail ecosystems
        retail_threshold = self.scenario.retail_access_threshold

        kesko = RetailBlock(
            model=self,
            name="Kesko",
            min_delivery_requirement=120.0,
            service_level_requirement=0.82,
            data_requirement=0.40,
            acceptance_threshold=retail_threshold,
            unit_price=1.00,
            max_volume_per_sme=850.0,
            dependency_penalty=0.20,
            coordination_cost=10.0,
        )
        s_group = RetailBlock(
            model=self,
            name="SGroup",
            min_delivery_requirement=130.0,
            service_level_requirement=0.83,
            data_requirement=0.42,
            acceptance_threshold=retail_threshold,
            unit_price=0.98,
            max_volume_per_sme=900.0,
            dependency_penalty=0.22,
            coordination_cost=10.0,
        )
        self.retail_blocks = [kesko, s_group]

        # Shared logistics operator
        self.logistics_operator = LogisticsOperator(
            model=self,
            name="RegionalLogistics",
            fleet_capacity=12000.0,
            fixed_route_cost=50.0,
            variable_cost_per_km=0.95,
            handling_cost_per_unit=0.05,
            min_profitable_load=60.0,
            bundling_capability=0.68,
            service_radius_km=220.0,
            reliability=0.92,
            neutrality_level=0.70,
        )

        # Optional neutral hub
        if self.scenario.neutral_hub_enabled:
            self.neutral_hub = NeutralHub(
                model=self,
                name="NeutralHub",
                hub_capacity=6000.0,
                entry_cost=5.0,
                handling_time_days=0.4,
                matching_efficiency=0.78,
                access_openness=0.92,
                ownership_neutrality=0.95,
            )

        # Optional public buyer
        if self.scenario.public_buyer_enabled:
            self.public_buyer = PublicBuyer(
                model=self,
                name="PublicBuyer",
                lot_size=80.0,
                allow_partial_bids=True,
                price_weight=0.35,
                reliability_weight=0.35,
                locality_weight=0.20,
                admin_complexity=0.20,
                unit_price=0.95,
                max_volume_per_sme=520.0,
            )

        # Optional HORECA
        if self.scenario.horeca_enabled:
            self.horeca_channel = HorecaChannel(
                model=self,
                name="HORECA",
                unit_price=1.10,
                demand_volatility=0.25,
                max_volume_per_sme=420.0,
            )

    def _pick_archetype_name(self) -> str:
        """
        Weighted random choice from archetype_shares.
        """
        names = list(self.archetype_shares.keys())
        probs = list(self.archetype_shares.values())
        total = sum(probs)
        if total <= 0:
            return names[0]
        normalized = [p / total for p in probs]
        return self.random.choices(names, weights=normalized, k=1)[0]

    def _sample_archetype_params(self, archetype_name: str) -> Dict[str, Any]:
        """
        Create a slightly varied SME instance around an archetype profile.

        This keeps the simulation from having perfectly identical firms.
        """
        base = copy.deepcopy(SME_ARCHETYPES[archetype_name])

        def jitter(value: float, pct: float, low: Optional[float] = None, high: Optional[float] = None) -> float:
            delta = value * pct
            sampled = self.random.uniform(value - delta, value + delta)
            if low is not None:
                sampled = max(low, sampled)
            if high is not None:
                sampled = min(high, sampled)
            return sampled

        # Volume-like variables
        base["production_volume"] = jitter(base["production_volume"], 0.18, low=50.0)
        base["delivery_batch_size"] = jitter(base["delivery_batch_size"], 0.20, low=5.0)
        base["shelf_life_days"] = jitter(base["shelf_life_days"], 0.15, low=1.0)
        base["base_unit_margin_eur"] = jitter(base["base_unit_margin_eur"], 0.15, low=0.05)
        base["cash_buffer_periods"] = jitter(base["cash_buffer_periods"], 0.20, low=1.0)

        # 0..1 variables
        for key in [
            "digital_capability",
            "collaboration_willingness",
            "trust_level",
            "quality_compliance",
            "logistics_competence",
            "risk_aversion",
            "dependency_tolerance",
            "learning_rate",
        ]:
            base[key] = max(0.0, min(1.0, jitter(base[key], 0.12, low=0.0, high=1.0)))

        # Integer-ish channel capacity
        ch_cap = int(round(jitter(base["channel_management_capacity"], 0.20, low=1.0)))
        base["channel_management_capacity"] = max(1, ch_cap)

        # Add a simple regional distance profile by archetype
        if archetype_name == "greenhouse_vegetable_sme":
            base["distance_to_market_km"] = self.random.uniform(20.0, 90.0)
        elif archetype_name == "farm_shop_restaurant_sme":
            base["distance_to_market_km"] = self.random.uniform(15.0, 80.0)
        elif archetype_name == "fish_farming_sme":
            base["distance_to_market_km"] = self.random.uniform(40.0, 130.0)
        else:
            base["distance_to_market_km"] = self.random.uniform(20.0, 120.0)

        return base

    def _build_smes(self) -> None:
        """
        Instantiate SME agents based on archetypes.
        """
        for _ in range(self.num_smes):
            archetype_name = self._pick_archetype_name()
            params = self._sample_archetype_params(archetype_name)

            sme = SMEAgent(
                model=self,
                archetype=archetype_name,
                production_volume=params["production_volume"],
                delivery_batch_size=params["delivery_batch_size"],
                shelf_life_days=params["shelf_life_days"],
                base_unit_margin_eur=params["base_unit_margin_eur"],
                cash_buffer_periods=params["cash_buffer_periods"],
                digital_capability=params["digital_capability"],
                collaboration_willingness=params["collaboration_willingness"],
                trust_level=params["trust_level"],
                quality_compliance=params["quality_compliance"],
                logistics_competence=params["logistics_competence"],
                channel_management_capacity=params["channel_management_capacity"],
                risk_aversion=params["risk_aversion"],
                dependency_tolerance=params["dependency_tolerance"],
                learning_rate=params["learning_rate"],
                preferred_channels=params.get("preferred_channels", []),
                distance_to_market_km=params.get("distance_to_market_km"),
            )
            self.smes.append(sme)

    # ---------------------------------------------------------------------
    # Metrics for DataCollector
    # ---------------------------------------------------------------------

    def active_smes(self) -> List[SMEAgent]:
        return [s for s in self.smes if s.alive]

    def avg_profit(self) -> float:
        alive = self.active_smes()
        return sum(s.last_profit for s in alive) / len(alive) if alive else 0.0

    def avg_logistics_cost(self) -> float:
        alive = self.active_smes()
        return sum(s.last_logistics_cost for s in alive) / len(alive) if alive else 0.0

    def avg_dependency(self) -> float:
        alive = self.active_smes()
        return sum(s.current_dependency for s in alive) / len(alive) if alive else 0.0

    def survival_rate(self) -> float:
        if not self.smes:
            return 0.0
        return sum(1 for s in self.smes if s.alive) / len(self.smes)

    def avg_channel_count(self) -> float:
        alive = self.active_smes()
        return sum(s.last_channel_count for s in alive) / len(alive) if alive else 0.0

    def total_emissions(self) -> float:
        return sum(s.last_emissions for s in self.smes if s.alive)

    # ---------------------------------------------------------------------
    # Simulation control
    # ---------------------------------------------------------------------

    def _reset_infrastructure(self) -> None:
        if self.logistics_operator is not None:
            self.logistics_operator.reset_step()
        if self.neutral_hub is not None:
            self.neutral_hub.reset_step()

    def step(self) -> None:
        """
        Advance one simulation step.

        Scheduling logic:
        1. Reset shared infrastructure
        2. Step active SMEs in random order
        3. Collect data
        """
        self._reset_infrastructure()

        smes = [s for s in self.smes if s.alive]
        self.random.shuffle(smes)

        for sme in smes:
            sme.step()

        self.current_step += 1
        self.datacollector.collect(self)

        if not any(s.alive for s in self.smes):
            self.running = False

    def run_model(self, steps: int = 12) -> None:
        """
        Run the model for a fixed number of steps or until all SMEs exit.
        """
        for _ in range(steps):
            if not self.running:
                break
            self.step()

    # ---------------------------------------------------------------------
    # Data access helpers
    # ---------------------------------------------------------------------

    def get_model_dataframe(self) -> pd.DataFrame:
        df = self.datacollector.get_model_vars_dataframe().copy()
        if df.empty:
            return df
        return df

    def get_agent_dataframe(self) -> pd.DataFrame:
        df = self.datacollector.get_agent_vars_dataframe().copy()
        if df.empty:
            return df
        return df

    def summary_dict(self) -> Dict[str, Any]:
        """
        Compact summary for UI/debugging.
        """
        return {
            "scenario": self.scenario.name,
            "step": self.current_step,
            "avg_profit": self.avg_profit(),
            "avg_logistics_cost": self.avg_logistics_cost(),
            "avg_dependency": self.avg_dependency(),
            "survival_rate": self.survival_rate(),
            "avg_channel_count": self.avg_channel_count(),
            "total_emissions": self.total_emissions(),
        }


__all__ = [
    "ScenarioConfig",
    "scenario_from_any",
    "FoodSupplyModel",
]
