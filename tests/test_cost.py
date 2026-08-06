"""
Unit tests for cost_estimator.py.

No mocking needed here at all -- this module makes zero network calls,
so every test just calls the function directly with different inputs.
"""

import pytest

from src.tools.cost_estimator import estimate_trip_cost
from src.utils.exceptions import CostEstimatorError


def test_within_budget_generous_budget():
    result = estimate_trip_cost(budget=500_000, duration_days=5, num_travelers=2)

    assert result.within_budget is True
    assert result.recommendations == []
    assert result.remaining_budget > 0
    assert result.breakdown.total == pytest.approx(
        result.breakdown.hotel
        + result.breakdown.transport
        + result.breakdown.food
        + result.breakdown.activities
        + result.breakdown.miscellaneous
    )


def test_over_budget_tier_defaults_produce_downgrade_suggestions():
    result = estimate_trip_cost(
        budget=10_000, duration_days=5, num_travelers=2, hotel_tier="luxury", food_tier="luxury"
    )

    assert result.within_budget is False
    assert result.remaining_budget < 0
    assert any("hotel tier" in r.lower() for r in result.recommendations)
    assert any("food budget" in r.lower() for r in result.recommendations)
    assert any("activities" in r.lower() for r in result.recommendations)


def test_custom_hotel_rate_skips_tier_downgrade_suggestion():
    # Explicit rate given -> we don't know the tier ladder, so no
    # "switch tier" suggestion should be generated for hotel specifically.
    result = estimate_trip_cost(
        budget=5_000,
        duration_days=5,
        num_travelers=1,
        hotel_nightly_rate=20_000,  # deliberately expensive, over budget
    )

    assert result.within_budget is False
    assert not any("hotel tier" in r.lower() for r in result.recommendations)


def test_zero_budget_raises():
    with pytest.raises(CostEstimatorError, match="budget"):
        estimate_trip_cost(budget=0, duration_days=5)


def test_negative_budget_raises():
    with pytest.raises(CostEstimatorError, match="budget"):
        estimate_trip_cost(budget=-100, duration_days=5)


def test_zero_duration_raises():
    with pytest.raises(CostEstimatorError, match="duration_days"):
        estimate_trip_cost(budget=100_000, duration_days=0)


def test_negative_duration_raises():
    with pytest.raises(CostEstimatorError, match="duration_days"):
        estimate_trip_cost(budget=100_000, duration_days=-3)


def test_zero_travelers_raises():
    with pytest.raises(CostEstimatorError, match="num_travelers"):
        estimate_trip_cost(budget=100_000, duration_days=5, num_travelers=0)


def test_invalid_tier_raises():
    with pytest.raises(CostEstimatorError, match="hotel_tier"):
        estimate_trip_cost(budget=100_000, duration_days=5, hotel_tier="ultra-luxury")


def test_one_day_trip_has_no_hotel_cost():
    result = estimate_trip_cost(budget=100_000, duration_days=1, num_travelers=2)
    assert result.breakdown.hotel == 0.0


def test_thirty_day_trip_scales_up():
    result = estimate_trip_cost(budget=10_000_000, duration_days=30, num_travelers=1)
    assert result.breakdown.hotel > 0
    assert result.breakdown.food > 0
    assert result.within_budget is True


def test_multiple_travelers_room_sharing():
    # Double occupancy: 2 travelers -> 1 room, 4 travelers -> 2 rooms.
    # So going from 2 to 4 travelers doubles the hotel cost (2x the rooms),
    # while food cost also doubles (it's per-person, not per-room).
    result_2 = estimate_trip_cost(budget=1_000_000, duration_days=5, num_travelers=2)
    result_4 = estimate_trip_cost(budget=1_000_000, duration_days=5, num_travelers=4)

    assert result_4.breakdown.hotel == pytest.approx(result_2.breakdown.hotel * 2)
    assert result_4.breakdown.food == pytest.approx(result_2.breakdown.food * 2)


def test_odd_number_of_travelers_still_shares_rooms():
    # 3 travelers -> 2 rooms (one room has a single occupant), not 3
    # separate rooms -- ceil division, not a flat per-person hotel cost.
    result_2 = estimate_trip_cost(budget=1_000_000, duration_days=5, num_travelers=2)
    result_3 = estimate_trip_cost(budget=1_000_000, duration_days=5, num_travelers=3)

    assert result_3.breakdown.hotel == pytest.approx(result_2.breakdown.hotel * 2)


def test_transport_cost_zero_when_no_distance_given():
    result = estimate_trip_cost(budget=100_000, duration_days=5)
    assert result.breakdown.transport == 0.0


def test_transport_driving_cost_from_distance():
    result = estimate_trip_cost(budget=1_000_000, duration_days=3, transport_distance_km=200, transport_mode="driving")
    assert result.breakdown.transport == pytest.approx(200 * 30.0)


def test_transport_public_is_per_traveler():
    result = estimate_trip_cost(
        budget=1_000_000,
        duration_days=3,
        num_travelers=3,
        transport_distance_km=200,
        transport_mode="public",
    )
    assert result.breakdown.transport == pytest.approx(200 * 12.0 * 3)


def test_explicit_activity_costs_are_summed():
    result = estimate_trip_cost(
        budget=1_000_000, duration_days=3, activity_costs=[1000, 2500, 500]
    )
    assert result.breakdown.activities == 4000.0


def test_insufficient_savings_message_when_gap_too_large():
    # Deliberately extreme: even every suggestion combined can't close
    # a budget this small relative to spend -- the shortfall message
    # must appear.
    result = estimate_trip_cost(
        budget=1,
        duration_days=10,
        num_travelers=4,
        hotel_tier="luxury",
        food_tier="luxury",
        activity_tier="luxury",
        transport_distance_km=1000,
    )
    assert result.within_budget is False
    assert any("still short by" in r for r in result.recommendations)


def test_budget_percentage_used_calculation():
    result = estimate_trip_cost(
        budget=10_000, duration_days=1, num_travelers=1, food_daily_rate=5_000, activity_costs=[0]
    )
    # total = 5000 (food) + 0 (activities) + 0 (transport) + 0 (hotel) + 10% misc = 5500
    assert result.breakdown.total == 5500.0
    assert result.budget_percentage_used == 55.0
    assert result.remaining_budget == 4500.0
    assert result.within_budget is True


def test_public_transport_cheaper_suggestion(mocker):
    # Solo traveler, driving mode: bus per-person cost (6/km * 1 traveler)
    # beats a private car's flat cost (35/km) when there's only 1 traveler
    # -- the suggestion should fire.
    result = estimate_trip_cost(
        budget=1_000,
        duration_days=1,
        num_travelers=1,
        hotel_tier="budget",
        food_tier="budget",
        transport_mode="driving",
        transport_distance_km=100,
    )

    assert result.within_budget is False
    assert any("bus" in r.lower() for r in result.recommendations)
