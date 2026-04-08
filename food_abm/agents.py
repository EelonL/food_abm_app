"""
food_abm/agents.py

Agent definitions for a Mesa-based agent-based simulation of regional food
supply chains in Uusimaa, Finland.

Design goals:
- Keep the code readable and easy to extend.
- Represent the key actors discussed in the study:
  * SME food firms
  * two dominant retail blocks
  * logistics operators
  * neutral hubs
  * public buyers
  * HORECA channel
- Support scenario-based experimentation rather than detailed forecasting.

This file is intended to be copied into a GitHub repository as:
    food_abm/agents.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

from mesa import Agent


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float into [low, high]."""
    return max(low, min(high, value))


def safe_mean(values: List[float]) -> float:
    """Return mean or 0.0 if list is empty."""
    return sum(values) / len(values) if values else 0.0


def logistic(x: float) -> float:
    """Simple logistic transform."""
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Channel offer object
# ---------------------------------------------------------------------------

@dataclass
class ChannelOffer:
    """
    Represents one potential commercial channel for an SME in one simulation step.

    Attributes
    ----------
    channel_name:
        Logical name of the channel.
    buyer_name:
        Human-readable buyer / channel owner name.
    acceptance_probability:
        Probability that the SME is accepted / matched this step.
    unit_price:
        Selling price per unit.
    unit_logistics_cost:
        Logistics cost per unit for this channel.
    stability:
        How stable/predictable the demand is (0..1).
    dependency_penalty:
        Penalty for strategic dependency on this channel (0..1 or cost-like).
    coordination_cost:
        Extra effort cost for managing this channel.
    max_volume:
        Maximum volume this channel can absorb this step.
    requires_hub:
        Whether the offer assumes use of a neutral hub.
    metadata:
        Optional channel-specific diagnostics.
    """
    channel_name: str
    buyer_name: str
    acceptance_probability: float
    unit_price: float
    unit_logistics_cost: float
    stability: float
    dependency_penalty: float
    coordination_cost: float
    max_volume: float
    requires_hub: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def expected_profit(self, volume: float, base_unit_margin: float) -> float:
        """
        Expected contribution margin from this channel for a given volume.

        The model assumes:
        - base_unit_margin is the SME's own per-unit margin before channel-specific
          logistics and coordination effects
        - unit_price acts as a market attractiveness proxy
        """
        effective_volume = min(volume, self.max_volume)
        expected_sales = self.acceptance_probability * effective_volume
        return expected_sales * (self.unit_price + base_unit_margin - self.unit_logistics_cost) - self.coordination_cost


# ---------------------------------------------------------------------------
# Base channel / buyer agents
# ---------------------------------------------------------------------------

class RetailBlock(Agent):
    """
    Represents one dominant vertically integrated retail ecosystem
    (for example Kesko or S-ryhmä).

    The retail block acts as a gatekeeper:
    - minimum delivery sizes
    - quality/compliance requirements
    - digital/data requirements
    - preference for service level and volume fit
    """

    def __init__(
        self,
        model,
        name: str,
        min_delivery_requirement: float = 120.0,
        service_level_requirement: float = 0.80,
        data_requirement: float = 0.40,
        acceptance_threshold: float = 0.60,
        unit_price: float = 1.0,
        max_volume_per_sme: float = 800.0,
        price_weight: float = 0.25,
        service_weight: float = 0.25,
        compliance_weight: float = 0.25,
        volume_weight: float = 0.15,
        locality_weight: float = 0.10,
        dependency_penalty: float = 0.18,
        coordination_cost: float = 10.0,
    ) -> None:
        super().__init__(model)
        self.name = name
        self.min_delivery_requirement = min_delivery_requirement
        self.service_level_requirement = service_level_requirement
        self.data_requirement = data_requirement
        self.acceptance_threshold = acceptance_threshold
        self.unit_price = unit_price
        self.max_volume_per_sme = max_volume_per_sme
        self.price_weight = price_weight
        self.service_weight = service_weight
        self.compliance_weight = compliance_weight
        self.volume_weight = volume_weight
        self.locality_weight = locality_weight
        self.dependency_penalty = dependency_penalty
        self.coordination_cost = coordination_cost

    def evaluate_sme(self, sme: "SMEAgent") -> Tuple[float, Dict[str, float]]:
        """
        Returns an acceptance probability proxy and component scores.
        """
        # Hard gates
        if sme.delivery_batch_size < self.min_delivery_requirement:
            return 0.0, {"hard_fail": 1.0}
        if sme.quality_compliance < self.acceptance_threshold:
            return 0.0, {"hard_fail": 1.0}
        if sme.digital_capability < self.data_requirement:
            return 0.0, {"hard_fail": 1.0}

        price_score = clamp(0.5 + 0.1 * (2.0 - sme.base_unit_margin_eur))
        service_score = clamp((sme.logistics_competence + sme.quality_compliance) / 2.0)
        compliance_score = clamp(sme.quality_compliance)
        volume_score = clamp(sme.delivery_batch_size / max(self.min_delivery_requirement * 2.0, 1.0))
        locality_score = 0.7  # Simplified: same broad region in this prototype

        score = (
            self.price_weight * price_score
            + self.service_weight * service_score
            + self.compliance_weight * compliance_score
            + self.volume_weight * volume_score
            + self.locality_weight * locality_score
        )

        acceptance_probability = clamp(score)
        details = {
            "price_score": price_score,
            "service_score": service_score,
            "compliance_score": compliance_score,
            "volume_score": volume_score,
            "locality_score": locality_score,
            "total_score": score,
        }
        return acceptance_probability, details

    def create_offer(self, sme: "SMEAgent") -> Optional[ChannelOffer]:
        acceptance_probability, details = self.evaluate_sme(sme)
        if acceptance_probability <= 0.0:
            return None

        scenario = getattr(self.model, "scenario", None)
        fuel_multiplier = getattr(scenario, "fuel_cost_multiplier", 1.0) if scenario else 1.0

        unit_logistics_cost = (
            0.20
            + 0.002 * sme.distance_to_market_km
            + 0.12 * (1.0 - sme.logistics_competence)
        ) * fuel_multiplier

        return ChannelOffer(
            channel_name="retail",
            buyer_name=self.name,
            acceptance_probability=acceptance_probability,
            unit_price=self.unit_price,
            unit_logistics_cost=unit_logistics_cost,
            stability=0.75,
            dependency_penalty=self.dependency_penalty,
            coordination_cost=self.coordination_cost,
            max_volume=self.max_volume_per_sme,
            requires_hub=False,
            metadata=details,
        )

    def step(self) -> None:
        # Retail blocks are passive in this prototype.
        return


class PublicBuyer(Agent):
    """
    Institutional buyer representing public procurement / school meals / welfare areas.

    Key mechanism:
    - Can be made more SME-friendly via scenario settings.
    """

    def __init__(
        self,
        model,
        name: str = "PublicBuyer",
        lot_size: float = 80.0,
        allow_partial_bids: bool = True,
        price_weight: float = 0.35,
        reliability_weight: float = 0.35,
        locality_weight: float = 0.20,
        admin_complexity: float = 0.20,
        unit_price: float = 0.95,
        max_volume_per_sme: float = 500.0,
    ) -> None:
        super().__init__(model)
        self.name = name
        self.lot_size = lot_size
        self.allow_partial_bids = allow_partial_bids
        self.price_weight = price_weight
        self.reliability_weight = reliability_weight
        self.locality_weight = locality_weight
        self.admin_complexity = admin_complexity
        self.unit_price = unit_price
        self.max_volume_per_sme = max_volume_per_sme

    def create_offer(self, sme: "SMEAgent") -> Optional[ChannelOffer]:
        if (not self.allow_partial_bids) and (sme.delivery_batch_size < self.lot_size):
            return None

        fit = clamp(sme.delivery_batch_size / max(self.lot_size, 1.0))
        reliability = clamp((sme.logistics_competence + sme.quality_compliance) / 2.0)
        locality = 0.8
        affordability = clamp(0.5 + 0.1 * (2.0 - sme.base_unit_margin_eur))

        score = (
            self.price_weight * affordability
            + self.reliability_weight * reliability
            + self.locality_weight * locality
        )

        acceptance_probability = clamp(0.4 + 0.5 * score)
        coordination_cost = 6.0 + 12.0 * self.admin_complexity

        return ChannelOffer(
            channel_name="public",
            buyer_name=self.name,
            acceptance_probability=acceptance_probability,
            unit_price=self.unit_price,
            unit_logistics_cost=0.16 + 0.0018 * sme.distance_to_market_km,
            stability=0.85,
            dependency_penalty=0.08,
            coordination_cost=coordination_cost,
            max_volume=self.max_volume_per_sme,
            requires_hub=False,
            metadata={"fit": fit, "score": score},
        )

    def step(self) -> None:
        return


class HorecaChannel(Agent):
    """
    Represents restaurants / catering / foodservice as a diversified market channel.
    """

    def __init__(
        self,
        model,
        name: str = "HORECA",
        unit_price: float = 1.10,
        demand_volatility: float = 0.25,
        max_volume_per_sme: float = 400.0,
    ) -> None:
        super().__init__(model)
        self.name = name
        self.unit_price = unit_price
        self.demand_volatility = demand_volatility
        self.max_volume_per_sme = max_volume_per_sme

    def create_offer(self, sme: "SMEAgent") -> Optional[ChannelOffer]:
        fit = clamp(0.5 + 0.2 * sme.channel_management_capacity / 4.0)
        freshness_fit = clamp(1.0 - min(sme.shelf_life_days, 14) / 20.0 + 0.4)
        acceptance_probability = clamp(0.35 + 0.35 * fit + 0.20 * freshness_fit)

        stability = clamp(0.55 - 0.25 * self.demand_volatility)
        unit_logistics_cost = 0.18 + 0.0022 * sme.distance_to_market_km

        return ChannelOffer(
            channel_name="horeca",
            buyer_name=self.name,
            acceptance_probability=acceptance_probability,
            unit_price=self.unit_price,
            unit_logistics_cost=unit_logistics_cost,
            stability=stability,
            dependency_penalty=0.06,
            coordination_cost=8.0,
            max_volume=self.max_volume_per_sme,
            requires_hub=False,
            metadata={"fit": fit, "freshness_fit": freshness_fit},
        )

    def step(self) -> None:
        return


class NeutralHub(Agent):
    """
    Neutral logistics hub / cross-docking / shared logistics service.

    The hub itself does not buy products. It improves logistics economics when used.
    """

    def __init__(
        self,
        model,
        name: str = "NeutralHub",
        hub_capacity: float = 5000.0,
        entry_cost: float = 5.0,
        handling_time_days: float = 0.4,
        matching_efficiency: float = 0.75,
        access_openness: float = 0.90,
        ownership_neutrality: float = 0.95,
    ) -> None:
        super().__init__(model)
        self.name = name
        self.hub_capacity = hub_capacity
        self.entry_cost = entry_cost
        self.handling_time_days = handling_time_days
        self.matching_efficiency = matching_efficiency
        self.access_openness = access_openness
        self.ownership_neutrality = ownership_neutrality
        self.current_load = 0.0

    def reset_step(self) -> None:
        self.current_load = 0.0

    def can_accept(self, volume: float) -> bool:
        return (self.current_load + volume) <= self.hub_capacity

    def expected_logistics_multiplier(self, sme: "SMEAgent") -> float:
        """
        Lower is better. A multiplier < 1.0 means cheaper logistics.
        """
        bundle_bonus = 0.30 * self.matching_efficiency
        collaboration_bonus = 0.15 * sme.collaboration_willingness
        trust_bonus = 0.10 * sme.trust_level
        raw = 1.0 - bundle_bonus - collaboration_bonus - trust_bonus
        return max(0.45, raw)

    def join_probability(self, sme: "SMEAgent", expected_benefit: float) -> float:
        z = (
            1.4 * sme.trust_level
            + 1.2 * sme.collaboration_willingness
            - 1.0 * sme.risk_aversion
            + 0.04 * expected_benefit
            + 0.6 * self.access_openness
            + 0.6 * self.ownership_neutrality
            - 0.1 * self.entry_cost
        )
        return clamp(logistic(z))

    def allocate(self, volume: float) -> bool:
        if self.can_accept(volume):
            self.current_load += volume
            return True
        return False

    def step(self) -> None:
        return


class LogisticsOperator(Agent):
    """
    Shared logistics operator. Can be neutral or tied to stronger market actors.

    Core role:
    - determines whether small deliveries are economically viable
    """

    def __init__(
        self,
        model,
        name: str = "LogisticsOperator",
        fleet_capacity: float = 10000.0,
        fixed_route_cost: float = 45.0,
        variable_cost_per_km: float = 0.9,
        handling_cost_per_unit: float = 0.05,
        min_profitable_load: float = 60.0,
        bundling_capability: float = 0.65,
        service_radius_km: float = 180.0,
        reliability: float = 0.92,
        neutrality_level: float = 0.70,
    ) -> None:
        super().__init__(model)
        self.name = name
        self.fleet_capacity = fleet_capacity
        self.fixed_route_cost = fixed_route_cost
        self.variable_cost_per_km = variable_cost_per_km
        self.handling_cost_per_unit = handling_cost_per_unit
        self.min_profitable_load = min_profitable_load
        self.bundling_capability = bundling_capability
        self.service_radius_km = service_radius_km
        self.reliability = reliability
        self.neutrality_level = neutrality_level
        self.used_capacity = 0.0

    def reset_step(self) -> None:
        self.used_capacity = 0.0

    def can_serve(self, volume: float, distance_km: float) -> bool:
        if distance_km > self.service_radius_km:
            return False
        if self.used_capacity + volume > self.fleet_capacity:
            return False
        return True

    def estimate_unit_cost(self, volume: float, distance_km: float, use_hub: bool = False) -> float:
        scenario = getattr(self.model, "scenario", None)
        fuel_multiplier = getattr(scenario, "fuel_cost_multiplier", 1.0) if scenario else 1.0

        route_cost = self.fixed_route_cost + distance_km * self.variable_cost_per_km * fuel_multiplier
        if use_hub:
            route_cost *= max(0.55, 1.0 - 0.35 * self.bundling_capability)

        effective_volume = max(volume, 1.0)
        unit_cost = route_cost / effective_volume + self.handling_cost_per_unit
        return unit_cost

    def delivery_probability(self, volume: float, distance_km: float, sme_competence: float) -> float:
        load_fit = clamp(volume / max(self.min_profitable_load, 1.0))
        distance_fit = clamp(1.0 - distance_km / max(self.service_radius_km, 1.0))
        score = 0.4 * self.reliability + 0.25 * load_fit + 0.20 * distance_fit + 0.15 * sme_competence
        return clamp(score)

    def register_delivery(self, volume: float) -> None:
        self.used_capacity += volume

    def step(self) -> None:
        return


# ---------------------------------------------------------------------------
# SME agent
# ---------------------------------------------------------------------------

class SMEAgent(Agent):
    """
    SME food business.

    This is the key decision-making agent in the model.
    It evaluates available channels every step and chooses the best one based on:
    - expected profit
    - market access
    - stability
    - strategic fit
    - dependency penalty
    - coordination burden

    The current implementation keeps one dominant channel per step to keep the
    prototype readable. That can later be extended to multi-channel allocation.
    """

    def __init__(
        self,
        model,
        archetype: str,
        production_volume: float,
        delivery_batch_size: float,
        shelf_life_days: float,
        base_unit_margin_eur: float,
        cash_buffer_periods: float,
        digital_capability: float,
        collaboration_willingness: float,
        trust_level: float,
        quality_compliance: float,
        logistics_competence: float,
        channel_management_capacity: int,
        risk_aversion: float,
        dependency_tolerance: float,
        learning_rate: float,
        preferred_channels: Optional[List[str]] = None,
        distance_to_market_km: Optional[float] = None,
    ) -> None:
        super().__init__(model)
        self.archetype = archetype
        self.production_volume = float(production_volume)
        self.delivery_batch_size = float(delivery_batch_size)
        self.shelf_life_days = float(shelf_life_days)
        self.base_unit_margin_eur = float(base_unit_margin_eur)
        self.cash_buffer_periods = float(cash_buffer_periods)
        self.digital_capability = clamp(float(digital_capability))
        self.collaboration_willingness = clamp(float(collaboration_willingness))
        self.trust_level = clamp(float(trust_level))
        self.quality_compliance = clamp(float(quality_compliance))
        self.logistics_competence = clamp(float(logistics_competence))
        self.channel_management_capacity = int(channel_management_capacity)
        self.risk_aversion = clamp(float(risk_aversion))
        self.dependency_tolerance = clamp(float(dependency_tolerance))
        self.learning_rate = clamp(float(learning_rate), 0.0, 1.0)
        self.preferred_channels = preferred_channels or []
        self.distance_to_market_km = (
            float(distance_to_market_km)
            if distance_to_market_km is not None
            else self.random.uniform(20.0, 120.0)
        )

        # Dynamic state
        self.state = "independent"
        self.alive = True
        self.last_profit = 0.0
        self.last_revenue = 0.0
        self.last_volume_sold = 0.0
        self.last_emissions = 0.0
        self.last_logistics_cost = 0.0
        self.last_channel_name = "none"
        self.last_buyer_name = "none"
        self.last_dependency_ratio = 0.0
        self.last_channel_count = 1
        self.current_dependency = 0.0
        self.market_access_success = 0.0
        self.periods_in_distress = 0

        # History
        self.channel_history: List[str] = []
        self.profit_history: List[float] = []

        # Utility weights (bounded rationality)
        self.alpha_profit = 0.45
        self.beta_access = 0.20
        self.gamma_stability = 0.15
        self.delta_strategic_fit = 0.10
        self.lambda_dependency = 0.15
        self.mu_coordination = 0.08

    # ---------------------------------------------------------------------
    # Internal calculations
    # ---------------------------------------------------------------------

    def preferred_channel_bonus(self, channel_name: str) -> float:
        return 0.15 if channel_name in self.preferred_channels else 0.0

    def strategic_fit(self, offer: ChannelOffer) -> float:
        """
        Channel fit based on SME properties and preferred channels.
        """
        freshness_component = 0.0
        if offer.channel_name in {"horeca", "retail"}:
            freshness_component = clamp(1.0 - self.shelf_life_days / 15.0 + 0.3)
        elif offer.channel_name == "public":
            freshness_component = 0.6
        elif offer.channel_name == "hub":
            freshness_component = 0.7
        else:
            freshness_component = 0.5

        capability_component = clamp(
            0.4 * self.digital_capability
            + 0.3 * self.logistics_competence
            + 0.3 * (self.channel_management_capacity / 5.0)
        )

        preference_component = self.preferred_channel_bonus(offer.channel_name)

        return clamp(0.4 * freshness_component + 0.5 * capability_component + preference_component)

    def expected_channel_utility(self, offer: ChannelOffer) -> float:
        volume = min(self.production_volume, offer.max_volume)
        expected_profit = offer.expected_profit(volume, self.base_unit_margin_eur)
        strategic_fit = self.strategic_fit(offer)

        utility = (
            self.alpha_profit * expected_profit
            + self.beta_access * offer.acceptance_probability * 100.0
            + self.gamma_stability * offer.stability * 100.0
            + self.delta_strategic_fit * strategic_fit * 100.0
            - self.lambda_dependency * offer.dependency_penalty * 100.0
            - self.mu_coordination * offer.coordination_cost
        )
        return utility

    def estimate_emissions(self, volume: float, distance_km: float, use_hub: bool) -> float:
        """
        Very simple emissions proxy for prototype purposes.
        """
        base_factor = 0.015
        if use_hub:
            base_factor *= 0.82
        return volume * distance_km * base_factor

    def update_learning(self, success: bool, used_hub: bool) -> None:
        """
        Simple adaptive learning mechanism.
        """
        delta = self.learning_rate if success else -self.learning_rate * 0.6

        if used_hub:
            self.trust_level = clamp(self.trust_level + 0.5 * delta)
            self.collaboration_willingness = clamp(self.collaboration_willingness + 0.6 * delta)
        else:
            self.trust_level = clamp(self.trust_level + 0.2 * delta)

        if success:
            self.logistics_competence = clamp(self.logistics_competence + 0.15 * self.learning_rate)

    def decide_hub_usage(self, neutral_hub: Optional[NeutralHub], operator: LogisticsOperator) -> bool:
        if neutral_hub is None:
            return False

        if not neutral_hub.can_accept(self.delivery_batch_size):
            return False

        base_cost = operator.estimate_unit_cost(
            volume=self.delivery_batch_size,
            distance_km=self.distance_to_market_km,
            use_hub=False,
        )
        hub_cost = operator.estimate_unit_cost(
            volume=self.delivery_batch_size,
            distance_km=self.distance_to_market_km,
            use_hub=True,
        ) * neutral_hub.expected_logistics_multiplier(self)

        expected_benefit = max(0.0, (base_cost - hub_cost) * self.delivery_batch_size - neutral_hub.entry_cost)
        p_join = neutral_hub.join_probability(self, expected_benefit)

        return self.random.random() < p_join

    def create_hub_offer(self, neutral_hub: NeutralHub, operator: LogisticsOperator) -> Optional[ChannelOffer]:
        if not neutral_hub.can_accept(self.delivery_batch_size):
            return None

        unit_log_cost = operator.estimate_unit_cost(
            volume=self.delivery_batch_size,
            distance_km=self.distance_to_market_km,
            use_hub=True,
        ) * neutral_hub.expected_logistics_multiplier(self)

        acceptance_probability = clamp(
            0.45
            + 0.20 * self.collaboration_willingness
            + 0.15 * self.trust_level
            + 0.15 * operator.reliability
        )

        return ChannelOffer(
            channel_name="hub",
            buyer_name=neutral_hub.name,
            acceptance_probability=acceptance_probability,
            unit_price=0.98,
            unit_logistics_cost=unit_log_cost,
            stability=0.68,
            dependency_penalty=0.05,
            coordination_cost=neutral_hub.entry_cost,
            max_volume=min(self.production_volume, neutral_hub.hub_capacity),
            requires_hub=True,
            metadata={"hub_multiplier": neutral_hub.expected_logistics_multiplier(self)},
        )

    def gather_offers(self) -> List[ChannelOffer]:
        offers: List[ChannelOffer] = []

        operator = getattr(self.model, "logistics_operator", None)
        neutral_hub = getattr(self.model, "neutral_hub", None)

        for retail in getattr(self.model, "retail_blocks", []):
            offer = retail.create_offer(self)
            if offer is not None:
                # update logistics estimate via shared operator, if present
                if operator and operator.can_serve(self.delivery_batch_size, self.distance_to_market_km):
                    offer.unit_logistics_cost = operator.estimate_unit_cost(
                        self.delivery_batch_size,
                        self.distance_to_market_km,
                        use_hub=False,
                    )
                offers.append(offer)

        public_buyer = getattr(self.model, "public_buyer", None)
        if public_buyer is not None:
            offer = public_buyer.create_offer(self)
            if offer is not None:
                offers.append(offer)

        horeca = getattr(self.model, "horeca_channel", None)
        if horeca is not None:
            offer = horeca.create_offer(self)
            if offer is not None:
                offers.append(offer)

        if neutral_hub is not None and operator is not None:
            if self.decide_hub_usage(neutral_hub, operator):
                offer = self.create_hub_offer(neutral_hub, operator)
                if offer is not None:
                    offers.append(offer)

        return offers

    def choose_best_offer(self, offers: List[ChannelOffer]) -> Optional[ChannelOffer]:
        if not offers:
            return None

        scored = [(offer, self.expected_channel_utility(offer)) for offer in offers]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0]

    def execute_trade(self, offer: ChannelOffer) -> None:
        operator = getattr(self.model, "logistics_operator", None)
        neutral_hub = getattr(self.model, "neutral_hub", None)

        volume = min(self.production_volume, offer.max_volume)
        accepted = self.random.random() < offer.acceptance_probability

        if not accepted:
            self.last_profit = -offer.coordination_cost * 0.5
            self.last_revenue = 0.0
            self.last_volume_sold = 0.0
            self.last_emissions = 0.0
            self.last_logistics_cost = 0.0
            self.last_channel_name = offer.channel_name
            self.last_buyer_name = offer.buyer_name
            self.market_access_success = 0.0
            self.periods_in_distress += 1
            self.update_learning(success=False, used_hub=offer.requires_hub)
            return

        if operator is not None:
            if operator.can_serve(volume, self.distance_to_market_km):
                operator.register_delivery(volume)

        if offer.requires_hub and neutral_hub is not None:
            neutral_hub.allocate(volume)

        revenue = volume * offer.unit_price
        logistics_cost = volume * offer.unit_logistics_cost
        gross_margin = volume * self.base_unit_margin_eur
        profit = gross_margin + revenue - logistics_cost - offer.coordination_cost

        emissions = self.estimate_emissions(volume, self.distance_to_market_km, use_hub=offer.requires_hub)

        self.last_profit = profit
        self.last_revenue = revenue
        self.last_volume_sold = volume
        self.last_emissions = emissions
        self.last_logistics_cost = logistics_cost
        self.last_channel_name = offer.channel_name
        self.last_buyer_name = offer.buyer_name
        self.last_dependency_ratio = offer.dependency_penalty
        self.current_dependency = offer.dependency_penalty
        self.last_channel_count = 1
        self.market_access_success = 1.0

        self.channel_history.append(offer.channel_name)
        self.profit_history.append(profit)

        # dynamic state transitions
        if profit < 0:
            self.periods_in_distress += 1
        else:
            self.periods_in_distress = 0

        if offer.channel_name in {"retail"}:
            self.state = "duopoly_supplier"
        elif offer.channel_name in {"public", "horeca"}:
            self.state = "multichannel"
        elif offer.channel_name == "hub":
            self.state = "hub_user"
        else:
            self.state = "independent"

        self.update_learning(success=True, used_hub=offer.requires_hub)

    def handle_no_trade(self) -> None:
        idle_cost = 0.20 * self.production_volume
        spoilage_penalty = 0.0
        if self.shelf_life_days <= 7:
            spoilage_penalty = 0.10 * self.production_volume

        loss = idle_cost + spoilage_penalty
        self.last_profit = -loss
        self.last_revenue = 0.0
        self.last_volume_sold = 0.0
        self.last_emissions = 0.0
        self.last_logistics_cost = 0.0
        self.last_channel_name = "none"
        self.last_buyer_name = "none"
        self.last_dependency_ratio = 0.0
        self.current_dependency = 0.0
        self.market_access_success = 0.0
        self.periods_in_distress += 1
        self.state = "distressed"
        self.update_learning(success=False, used_hub=False)

    def survival_check(self) -> None:
        """
        Basic survival rule. Firms exit after repeated distress.
        """
        survival_limit = 4
        if self.cash_buffer_periods <= 2:
            survival_limit = 3
        if self.cash_buffer_periods >= 5:
            survival_limit = 5

        if self.periods_in_distress >= survival_limit:
            self.alive = False
            self.state = "exit"

    # ---------------------------------------------------------------------
    # Mesa step
    # ---------------------------------------------------------------------

    def step(self) -> None:
        if not self.alive:
            return

        offers = self.gather_offers()
        chosen_offer = self.choose_best_offer(offers)

        if chosen_offer is None:
            self.handle_no_trade()
        else:
            self.execute_trade(chosen_offer)

        self.survival_check()

    # ---------------------------------------------------------------------
    # Reporting helpers for DataCollector
    # ---------------------------------------------------------------------

    @property
    def is_alive(self) -> int:
        return 1 if self.alive else 0

    @property
    def last_profit_value(self) -> float:
        return self.last_profit

    @property
    def last_dependency_value(self) -> float:
        return self.last_dependency_ratio

    @property
    def last_channel_count_value(self) -> int:
        return self.last_channel_count


__all__ = [
    "ChannelOffer",
    "SMEAgent",
    "RetailBlock",
    "LogisticsOperator",
    "NeutralHub",
    "PublicBuyer",
    "HorecaChannel",
]
