
"""
Tests for the /plan and /estimate-cost endpoints.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_estimate_cost_endpoint():
    payload = {
        "budget": 100_000,
        "duration_days": 3,
        "num_travelers": 2,
        "transport_distance_km": 200,
        "transport_mode": "driving",
        "hotel_tier": "mid-range",
        "food_tier": "mid-range",
        "activity_tier": "mid-range",
    }
    response = client.post("/estimate-cost", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "breakdown" in body
    assert "within_budget" in body
    assert body["currency"] == "PKR"


def test_estimate_cost_over_budget():
    payload = {
        "budget": 1_000,
        "duration_days": 5,
        "num_travelers": 2,
        "hotel_tier": "luxury",
        "food_tier": "luxury",
    }
    response = client.post("/estimate-cost", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["within_budget"] is False
    assert len(body["recommendations"]) > 0


def test_estimate_cost_invalid_payload():
    response = client.post("/estimate-cost", json={"budget": -100, "duration_days": 1})
    assert response.status_code == 422


@patch("src.api.routes.graph.ainvoke", new_callable=AsyncMock)
async def test_plan_endpoint_success(mock_invoke):
    mock_invoke.return_value = {
        "messages": [],
        "user_input": None,
        "tool_results": {},
        "itinerary": "Day 1: Visit the museum...",
        "errors": [],
        "retry_count": 0,
        "reflection_notes": ["Plan looks good."],
    }

    payload = {
        "destination": "Hunza",
        "budget": 150_000,
        "duration_days": 5,
        "num_travelers": 2,
    }
    response = client.post("/plan", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["destination"] == "Hunza"
    assert "itinerary" in body
    assert "reflection_notes" in body


@patch("src.api.routes.graph.ainvoke", new_callable=AsyncMock)
async def test_plan_endpoint_with_errors(mock_invoke):
    mock_invoke.return_value = {
        "messages": [],
        "user_input": None,
        "tool_results": {},
        "itinerary": "Partial itinerary due to errors...",
        "errors": ["weather_tool: API unreachable"],
        "retry_count": 0,
        "reflection_notes": ["Missing weather data."],
    }

    payload = {
        "destination": "Skardu",
        "budget": 200_000,
        "duration_days": 4,
    }
    response = client.post("/plan", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["errors"]) > 0
    assert len(body["reflection_notes"]) > 0


def test_plan_endpoint_invalid_payload():
    response = client.post("/plan", json={"budget": -1, "duration_days": 0})
    assert response.status_code == 422
