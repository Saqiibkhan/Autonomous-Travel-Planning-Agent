
"""
API routes.

Endpoints:
- GET  /health        -- liveness
- GET  /status        -- readiness / config presence (no secrets leaked)
- POST /plan          -- run the full LangGraph workflow and return an itinerary
- POST /estimate-cost -- run only the cost estimator
"""

from fastapi import APIRouter

from src.agent.graph import graph
from src.agent.state import AgentState, UserInput
from src.config import settings
from src.schemas.plan import CostEstimateRequest, CostEstimateResponse, PlanRequest, PlanResponse
from src.tools.cost_estimator import estimate_trip_cost
from src.utils.constants import APP_VERSION

router = APIRouter()


@router.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok"}


@router.get("/status", tags=["system"])
async def status() -> dict:
    configured_keys = {
        "llm_api_key": bool(settings.llm_api_key),
        "search_api_key": bool(settings.search_api_key),
        "places_api_key": bool(settings.places_api_key),
        "weather_api_key": bool(settings.weather_api_key),
        "routing_api_key": bool(settings.routing_api_key),
        "currency_api_key": bool(settings.currency_api_key),
    }
    return {
        "app_name": settings.app_name,
        "version": APP_VERSION,
        "environment": settings.app_env,
        "llm_model": settings.llm_model,
        "configured_keys": configured_keys,
    }


@router.post("/plan", response_model=PlanResponse, tags=["planning"])
async def plan_trip(request: PlanRequest) -> PlanResponse:
    initial_state: AgentState = {
        "messages": [],
        "user_input": UserInput(
            destination=request.destination,
            budget=request.budget,
            duration_days=request.duration_days,
            num_travelers=request.num_travelers,
            origin=request.origin,
            preferences=request.preferences,
        ),
        "tool_results": {},
        "itinerary": None,
        "errors": [],
        "retry_count": 0,
        "reflection_notes": [],
    }

    final_state = await graph.ainvoke(initial_state)

    return PlanResponse(
        destination=request.destination,
        budget=request.budget,
        duration_days=request.duration_days,
        itinerary=final_state.get("itinerary", ""),
        reflection_notes=final_state.get("reflection_notes", []),
        errors=final_state.get("errors", []),
        tool_results=final_state.get("tool_results"),
    )


@router.post("/estimate-cost", response_model=CostEstimateResponse, tags=["planning"])
async def estimate_cost(request: CostEstimateRequest) -> CostEstimateResponse:
    result = estimate_trip_cost(
        budget=request.budget,
        duration_days=request.duration_days,
        num_travelers=request.num_travelers,
        transport_distance_km=request.transport_distance_km,
        transport_mode=request.transport_mode,
        hotel_tier=request.hotel_tier,
        food_tier=request.food_tier,
        activity_tier=request.activity_tier,
    )
    return CostEstimateResponse(
        currency=result.currency,
        breakdown=result.breakdown.model_dump(),
        budget=result.budget,
        remaining_budget=result.remaining_budget,
        budget_percentage_used=result.budget_percentage_used,
        within_budget=result.within_budget,
        recommendations=result.recommendations,
        transport_estimates=[te.model_dump() for te in result.transport_estimates],
    )
