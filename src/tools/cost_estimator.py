"""
Cost Estimator.

Pure Python calculation -- no external API calls. Combines hotel,
transport, food, and activity costs into a total, checks it against the
stated budget, and, if over budget, returns concrete, quantified
suggestions for closing the gap.

Transport rates are loaded from src/data/transport_rates.json, which
stores per-km prices for bus, car, bike, etc. When a routing distance
is available, the estimator computes cost estimates for ALL available
transport modes so the planner can show the user options like:

    Lahore → Hunza
    Bus   ≈ PKR 7,000
    Car   ≈ PKR 18,000
    Bike  ≈ PKR 4,200
"""

import json
import os
from typing import List, Literal, Optional

from src.schemas.cost import CostBreakdown, CostEstimateResult, TransportEstimate
from src.utils.exceptions import CostEstimatorError
from src.utils.logger import get_logger

logger = get_logger(__name__)

Tier = Literal["budget", "mid-range", "luxury"]
TransportMode = Literal["bus", "car", "bike", "driving", "public"]

TIER_ORDER: List[Tier] = ["budget", "mid-range", "luxury"]

# Rough PKR/day heuristics for domestic Pakistani travel -- placeholders,
# not live pricing. Swap in real data (a Hotel Tool, actual quotes) as
# it becomes available; these exist so the estimator is usable today.
HOTEL_NIGHTLY_RATES = {"budget": 3500.0, "mid-range": 8000.0, "luxury": 18000.0}
FOOD_DAILY_RATES = {"budget": 1200.0, "mid-range": 2500.0, "luxury": 5000.0}
ACTIVITY_DAILY_RATES = {"budget": 800.0, "mid-range": 2000.0, "luxury": 5000.0}

DEFAULT_MISC_PERCENTAGE = 0.10
ACTIVITY_REDUCTION_SUGGESTION_PCT = 0.25

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_TRANSPORT_RATES_PATH = os.path.join(_DATA_DIR, "transport_rates.json")

# Loaded once at module import time.
with open(_TRANSPORT_RATES_PATH, "r", encoding="utf-8") as _f:
    TRANSPORT_RATE_PER_KM = json.load(_f)


def _rooms_needed(num_travelers: int) -> int:
    """Assume double occupancy: 2 travelers share a room."""
    return max((num_travelers + 1) // 2, 1)


def _validate_inputs(
    budget: float,
    duration_days: int,
    num_travelers: int,
    hotel_tier: str,
    food_tier: str,
    activity_tier: str,
) -> None:
    if budget <= 0:
        raise CostEstimatorError("budget must be greater than zero.")
    if duration_days <= 0:
        raise CostEstimatorError("duration_days must be at least 1.")
    if num_travelers <= 0:
        raise CostEstimatorError("num_travelers must be at least 1.")
    for label, tier in (
        ("hotel_tier", hotel_tier),
        ("food_tier", food_tier),
        ("activity_tier", activity_tier),
    ):
        if tier not in TIER_ORDER:
            raise CostEstimatorError(f"{label} must be one of {TIER_ORDER}, got '{tier}'.")


def _transport_cost_for_mode(
    mode: str,
    distance_km: Optional[float],
    num_travelers: int,
) -> float:
    if distance_km is None:
        return 0.0
    rate = TRANSPORT_RATE_PER_KM.get(mode, {}).get("price_per_km", 0.0)
    if mode in ("bus", "public"):
        return distance_km * rate * num_travelers
    return distance_km * rate


def _build_transport_estimates(
    distance_km: Optional[float],
    num_travelers: int,
) -> List[TransportEstimate]:
    if distance_km is None:
        return []
    estimates = []
    for mode, meta in TRANSPORT_RATE_PER_KM.items():
        if mode in ("driving", "public"):
            continue  # legacy aliases, skip to avoid duplicates
        total = _transport_cost_for_mode(mode, distance_km, num_travelers)
        estimates.append(
            TransportEstimate(
                mode=mode,
                price_per_km=meta["price_per_km"],
                distance_km=distance_km,
                total_pkr=round(total, 2),
                description=meta.get("description", mode),
            )
        )
    return estimates


def estimate_trip_cost(
    budget: float,
    duration_days: int,
    num_travelers: int = 1,
    hotel_nightly_rate: Optional[float] = None,
    hotel_tier: Tier = "mid-range",
    transport_distance_km: Optional[float] = None,
    transport_mode: TransportMode = "car",
    food_daily_rate: Optional[float] = None,
    food_tier: Tier = "mid-range",
    activity_costs: Optional[List[float]] = None,
    activity_tier: Tier = "mid-range",
    misc_percentage: float = DEFAULT_MISC_PERCENTAGE,
    currency: str = "PKR",
) -> CostEstimateResult:
    """
    Estimate total trip cost and check it against budget.

    Hotel/food/activities each accept either an explicit rate (real data,
    when available) or fall back to a tier default. Transport is always
    computed from distance -- there's no sane "tier default" for a cost
    that depends entirely on how far the destination actually is; pass
    `transport_distance_km` (e.g. from the Routing Tool) to include it.

    Nights are assumed to be duration_days - 1 (a 5-day trip needs 4
    nights of lodging); a 1-day trip has 0 nights and therefore no hotel
    cost, which is intentional, not a bug.
    """
    _validate_inputs(budget, duration_days, num_travelers, hotel_tier, food_tier, activity_tier)

    nights = max(duration_days - 1, 0)
    rooms = _rooms_needed(num_travelers)

    used_hotel_tier = hotel_nightly_rate is None
    nightly_rate = (
        hotel_nightly_rate if hotel_nightly_rate is not None else HOTEL_NIGHTLY_RATES[hotel_tier]
    )
    hotel_cost = nightly_rate * nights * rooms

    transport_cost = _transport_cost_for_mode(transport_mode, transport_distance_km, num_travelers)

    used_food_tier = food_daily_rate is None
    daily_food_rate = food_daily_rate if food_daily_rate is not None else FOOD_DAILY_RATES[food_tier]
    food_cost = daily_food_rate * duration_days * num_travelers

    if activity_costs is not None:
        activities_cost = sum(activity_costs)
        used_activity_tier = False
    else:
        activities_cost = ACTIVITY_DAILY_RATES[activity_tier] * duration_days
        used_activity_tier = True

    subtotal = hotel_cost + transport_cost + food_cost + activities_cost
    miscellaneous = round(subtotal * misc_percentage, 2)
    total = round(subtotal + miscellaneous, 2)

    remaining_budget = round(budget - total, 2)
    percentage_used = round((total / budget) * 100, 1)
    within_budget = total <= budget

    breakdown = CostBreakdown(
        hotel=round(hotel_cost, 2),
        transport=round(transport_cost, 2),
        food=round(food_cost, 2),
        activities=round(activities_cost, 2),
        miscellaneous=miscellaneous,
        total=total,
    )

    transport_estimates = _build_transport_estimates(transport_distance_km, num_travelers)

    recommendations: List[str] = []
    if not within_budget:
        recommendations = _build_recommendations(
            overage=total - budget,
            currency=currency,
            hotel_tier=hotel_tier,
            used_hotel_tier=used_hotel_tier,
            nights=nights,
            rooms=rooms,
            food_tier=food_tier,
            used_food_tier=used_food_tier,
            duration_days=duration_days,
            num_travelers=num_travelers,
            activities_cost=activities_cost,
            transport_cost=transport_cost,
            transport_mode=transport_mode,
            transport_distance_km=transport_distance_km,
        )

    logger.info(
        "cost_estimator: total=%.2f budget=%.2f within_budget=%s recommendations=%d transport_estimates=%d",
        total,
        budget,
        within_budget,
        len(recommendations),
        len(transport_estimates),
    )

    return CostEstimateResult(
        currency=currency,
        breakdown=breakdown,
        budget=budget,
        remaining_budget=remaining_budget,
        budget_percentage_used=percentage_used,
        within_budget=within_budget,
        recommendations=recommendations,
        transport_estimates=transport_estimates,
    )


def _build_recommendations(
    *,
    overage: float,
    currency: str,
    hotel_tier: Tier,
    used_hotel_tier: bool,
    nights: int,
    rooms: int,
    food_tier: Tier,
    used_food_tier: bool,
    duration_days: int,
    num_travelers: int,
    activities_cost: float,
    transport_cost: float,
    transport_mode: str,
    transport_distance_km: Optional[float],
) -> List[str]:
    """
    Build concrete, quantified suggestions for closing the budget gap.
    Every suggestion states its estimated savings -- "spend less" isn't
    actionable, "switch to a budget hotel and save ~14,000 PKR" is. If
    the suggestions here can't close the whole gap, that's said
    explicitly rather than left implied.
    """
    suggestions: List[str] = []
    total_potential_savings = 0.0

    if used_hotel_tier and hotel_tier != "budget":
        lower_tier = TIER_ORDER[TIER_ORDER.index(hotel_tier) - 1]
        savings = (HOTEL_NIGHTLY_RATES[hotel_tier] - HOTEL_NIGHTLY_RATES[lower_tier]) * nights * rooms
        if savings > 0:
            suggestions.append(
                f"Switch hotel tier from '{hotel_tier}' to '{lower_tier}': "
                f"~{savings:,.0f} {currency} saved over {nights} night(s)."
            )
            total_potential_savings += savings

    if used_food_tier and food_tier != "budget":
        lower_tier = TIER_ORDER[TIER_ORDER.index(food_tier) - 1]
        savings = (
            (FOOD_DAILY_RATES[food_tier] - FOOD_DAILY_RATES[lower_tier]) * duration_days * num_travelers
        )
        if savings > 0:
            suggestions.append(
                f"Switch food budget from '{food_tier}' to '{lower_tier}': "
                f"~{savings:,.0f} {currency} saved over {duration_days} day(s)."
            )
            total_potential_savings += savings

    if activities_cost > 0:
        savings = round(activities_cost * ACTIVITY_REDUCTION_SUGGESTION_PCT, 2)
        suggestions.append(
            f"Trim activities by ~{ACTIVITY_REDUCTION_SUGGESTION_PCT:.0%} "
            f"(fewer or cheaper paid attractions): ~{savings:,.0f} {currency} saved."
        )
        total_potential_savings += savings

    if transport_distance_km and transport_mode in ("car", "driving"):
        bus_cost = _transport_cost_for_mode("bus", transport_distance_km, num_travelers)
        if bus_cost < transport_cost:
            savings = transport_cost - bus_cost
            suggestions.append(
                f"Use bus instead of private car: ~{savings:,.0f} {currency} saved."
            )
            total_potential_savings += savings

    if total_potential_savings < overage:
        suggestions.append(
            f"Even applying every suggestion above, the plan is still short by "
            f"~{overage - total_potential_savings:,.0f} {currency}. Consider a "
            f"shorter trip, a closer/cheaper destination, or a higher budget."
        )

    return suggestions
