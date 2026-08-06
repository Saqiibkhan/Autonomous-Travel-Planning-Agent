
"""
Tests for the LangGraph workflow (graph, nodes, state).
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.state import AgentState, ToolResults, UserInput


def _base_state() -> AgentState:
    return {
        "messages": [],
        "user_input": UserInput(
            destination="Hunza", budget=150_000, duration_days=5, num_travelers=2
        ),
        "tool_results": ToolResults(),
        "itinerary": None,
        "errors": [],
        "retry_count": 0,
        "reflection_notes": [],
    }


def test_input_parser_node_success():
    from src.agent.nodes import input_parser_node

    state = _base_state()
    result = input_parser_node(state)
    assert "errors" not in result or not result["errors"]


def test_input_parser_node_no_input():
    from src.agent.nodes import input_parser_node

    state: AgentState = {
        "messages": [],
        "user_input": None,
        "tool_results": ToolResults(),
        "itinerary": None,
        "errors": [],
        "retry_count": 0,
        "reflection_notes": [],
    }
    result = input_parser_node(state)
    assert any("No user input" in e for e in result.get("errors", []))


@pytest.mark.asyncio
async def test_planner_node_generates_itinerary():
    from src.agent.nodes import planner_node
    from langchain_core.messages import AIMessage

    state = _base_state()
    state["tool_results"] = {
        "search": {"overview": "Hunza is beautiful.", "culture": "Rich culture.", "best_season": "Summer", "travel_tips": ["Pack warm clothes."], "attractions_mentioned": ["Altit Fort"], "sources": []},
        "places": {"destination": "Hunza", "latitude": 36.3, "longitude": 74.6, "places": [{"name": "Altit Fort", "category": "landmark", "latitude": 36.3, "longitude": 74.6}]},
        "weather": {"destination": "Hunza", "forecast": [{"date": "2024-01-01", "temperature_min_c": -5, "temperature_max_c": 10, "condition": "Sunny", "rainfall_probability": 0.1, "warnings": []}], "notes": []},
        "routing": {"origin": (36.3, 74.6), "destination": (36.3, 74.6), "driving_distance_km": 10, "driving_duration_min": 15},
        "cost": {"currency": "PKR", "breakdown": {"hotel": 32000, "transport": 300, "food": 12500, "activities": 8000, "miscellaneous": 5580, "total": 58380}, "budget": 150000, "remaining_budget": 91620, "budget_percentage_used": 38.9, "within_budget": True, "recommendations": []},
    }

    with patch("src.agent.nodes._llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Day 1: Explore Altit Fort..."))
        result = await planner_node(state)
        assert result.get("itinerary") is not None
        assert "Day 1" in result["itinerary"]


@pytest.mark.asyncio
async def test_reflection_node_success():
    from src.agent.reflection import reflection_node

    state = _base_state()
    state["itinerary"] = "A nice plan."
    state["tool_results"] = {
        "search": {"overview": "x"},
        "places": {"places": [{"name": "Fort"}]},
        "weather": {"forecast": [{"date": "2024-01-01", "condition": "Sunny"}]},
        "routing": {"driving_distance_km": 10},
        "cost": {"within_budget": True, "breakdown": {"total": 50000}, "budget": 150000, "recommendations": []},
    }
    result = await reflection_node(state)
    assert "Plan looks good." in result["reflection_notes"]


@pytest.mark.asyncio
async def test_reflection_node_budget_exceeded():
    from src.agent.reflection import reflection_node

    state = _base_state()
    state["itinerary"] = "A plan."
    state["tool_results"] = {
        "search": {"overview": "x"},
        "places": {"places": []},
        "weather": {"forecast": []},
        "routing": {},
        "cost": {"within_budget": False, "breakdown": {"total": 200000}, "budget": 150000, "recommendations": ["Switch to budget hotel"]},
    }
    result = await reflection_node(state)
    assert any("exceeds budget by" in note for note in result["reflection_notes"])


@pytest.mark.asyncio
async def test_reflection_node_missing_tools():
    from src.agent.reflection import reflection_node

    state = _base_state()
    state["itinerary"] = "A plan."
    state["tool_results"] = {}
    result = await reflection_node(state)
    assert any("Missing data" in note for note in result["reflection_notes"])
