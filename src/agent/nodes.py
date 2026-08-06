
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.config import settings
from src.tools.cost_estimator import estimate_trip_cost
from src.tools.places_tool import find_places
from src.tools.routing_tool import get_route
from src.tools.search_tool import search_destination
from src.tools.weather_tool import get_weather
from src.utils.logger import get_logger

logger = get_logger(__name__)

_llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.llm_api_key or "dummy",
    base_url=settings.llm_base_url,
    temperature=0.7,
)


def input_parser_node(state: AgentState) -> Dict[str, Any]:
    user_input = state.get("user_input")
    if user_input is None:
        return {"errors": state.get("errors", []) + ["No user input provided."]}
    logger.info("input_parser: destination=%s budget=%s days=%d travelers=%d",
                user_input.destination, user_input.budget, user_input.duration_days, user_input.num_travelers)
    return {"user_input": user_input}


async def research_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    errors = state.get("errors", [])
    try:
        result = await search_destination(user_input.destination)
        tool_results = dict(state.get("tool_results", {}))
        tool_results["search"] = result.model_dump()
        return {"tool_results": tool_results}
    except Exception as exc:
        logger.warning("research_node failed: %s", exc)
        errors.append(f"search_tool: {exc}")
        return {"errors": errors}


async def places_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    errors = state.get("errors", [])
    try:
        result = await find_places(user_input.destination)
        tool_results = dict(state.get("tool_results", {}))
        tool_results["places"] = result.model_dump()
        return {"tool_results": tool_results}
    except Exception as exc:
        logger.warning("places_node failed: %s", exc)
        errors.append(f"places_tool: {exc}")
        return {"errors": errors}


async def weather_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    errors = state.get("errors", [])
    try:
        result = await get_weather(user_input.destination, user_input.duration_days)
        tool_results = dict(state.get("tool_results", {}))
        tool_results["weather"] = result.model_dump()
        return {"tool_results": tool_results}
    except Exception as exc:
        logger.warning("weather_node failed: %s", exc)
        errors.append(f"weather_tool: {exc}")
        return {"errors": errors}


async def routing_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    errors = state.get("errors", [])
    try:
        places_data = state.get("tool_results", {}).get("places")
        if not places_data or not places_data.get("places"):
            logger.warning("routing_node: no places data available for routing")
            return {"errors": errors + ["routing_tool: no places data available"]}

        origin_name = user_input.origin
        if not origin_name:
            logger.info("routing_node: no origin provided, skipping routing")
            return {"errors": errors + ["routing_tool: no origin provided"]}

        from src.services.geocoding import geocode
        origin_lat, origin_lon = await geocode(origin_name)
        dest_lat = places_data["latitude"]
        dest_lon = places_data["longitude"]

        result = await get_route(origin=(origin_lat, origin_lon), destination=(dest_lat, dest_lon))
        tool_results = dict(state.get("tool_results", {}))
        tool_results["routing"] = result.model_dump()
        return {"tool_results": tool_results}
    except Exception as exc:
        logger.warning("routing_node failed: %s", exc)
        errors.append(f"routing_tool: {exc}")
        return {"errors": errors}


async def cost_estimator_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    errors = state.get("errors", [])
    try:
        routing_data = state.get("tool_results", {}).get("routing")
        transport_distance_km = None
        if routing_data and routing_data.get("driving_distance_km"):
            transport_distance_km = routing_data["driving_distance_km"]
        else:
            transport_distance_km = 200.0

        result = estimate_trip_cost(
            budget=user_input.budget,
            duration_days=user_input.duration_days,
            num_travelers=user_input.num_travelers,
            transport_distance_km=transport_distance_km,
        )
        tool_results = dict(state.get("tool_results", {}))
        tool_results["cost"] = result.model_dump()
        return {"tool_results": tool_results}
    except Exception as exc:
        logger.warning("cost_estimator_node failed: %s", exc)
        errors.append(f"cost_estimator: {exc}")
        return {"errors": errors}


async def planner_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    tool_results = state.get("tool_results", {})
    errors = state.get("errors", [])

    search_text = ""
    if tool_results.get("search"):
        s = tool_results["search"]
        search_text = f"Overview: {s.get('overview', '')}\nCulture: {s.get('culture', '')}\nBest season: {s.get('best_season', '')}\nTips: {', '.join(s.get('travel_tips', []))}"

    places_text = ""
    if tool_results.get("places"):
        places = tool_results["places"].get("places", [])[:10]
        places_text = "\n".join([f"- {p['name']} ({p['category']})" for p in places])

    weather_text = ""
    if tool_results.get("weather"):
        days = tool_results["weather"].get("forecast", [])[:5]
        weather_text = "\n".join([f"{d['date']}: {d['condition']}, {d['temperature_min_c']}-{d['temperature_max_c']}°C, rain {d['rainfall_probability']:.0%}" for d in days])
        notes = tool_results["weather"].get("notes", [])
        if notes:
            weather_text += f"\nNotes: {'; '.join(notes)}"

    cost_text = ""
    if tool_results.get("cost"):
        c = tool_results["cost"]
        cost_text = f"Total: {c['breakdown']['total']:,.0f} {c['currency']} (budget: {c['budget']:,.0f}, within budget: {c['within_budget']})\n"
        cost_text += f"Breakdown: hotel {c['breakdown']['hotel']:,.0f}, transport {c['breakdown']['transport']:,.0f}, food {c['breakdown']['food']:,.0f}, activities {c['breakdown']['activities']:,.0f}\n"
        if c.get("transport_estimates"):
            cost_text += "Transport estimates by mode:\n"
            for te in c["transport_estimates"]:
                cost_text += f"  - {te['mode']}: {te['total_pkr']:,.0f} PKR ({te['price_per_km']} PKR/km x {te['distance_km']} km)\n"
        if c.get("recommendations"):
            cost_text += f"Recommendations: {'; '.join(c['recommendations'])}"

    routing_text = ""
    if tool_results.get("routing"):
        r = tool_results["routing"]
        routing_text = f"Driving: {r.get('driving_distance_km')} km, {r.get('driving_duration_min')} min"
        if r.get("walking_distance_km"):
            routing_text += f"\nWalking: {r['walking_distance_km']} km, {r['walking_duration_min']} min"
        if r.get("warnings"):
            routing_text += f"\nWarnings: {'; '.join(r['warnings'])}"

    prefs = user_input.preferences or "No specific preferences provided."

    system_prompt = (
        "You are an expert travel planner. Produce a concise, day-by-day itinerary "
        "based ONLY on the provided research. Be specific about which attractions to visit "
        "each day, factoring in weather and routing. If something is missing from the data, "
        "say so honestly rather than inventing details. "
        "Include cost notes where relevant. "
        "IMPORTANT: Do NOT invent adjusted cost totals. Use ONLY the exact cost figures "
        "provided in the COST section. If the estimate exceeds budget, state that clearly "
        "and list the provided recommendations. "
        "When transport_estimates are provided in the COST section, include them in the "
        "itinerary as 'Transport options: [mode]: [total] PKR' for each mode."
    )

    human_prompt = f"""Plan a {user_input.duration_days}-day trip to {user_input.destination} for {user_input.num_travelers} traveler(s) with a budget of {user_input.budget:,.0f}.

Preferences: {prefs}

=== RESEARCH ===
{search_text}

=== PLACES ===
{places_text}

=== WEATHER ===
{weather_text}

=== ROUTING ===
{routing_text}

=== COST ===
{cost_text}

=== ERRORS ===
{'; '.join(errors) if errors else 'None'}

Generate a day-by-day itinerary with morning, afternoon, and evening suggestions."""

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        return {"itinerary": response.content, "messages": [HumanMessage(content=human_prompt), response]}
    except Exception as exc:
        logger.error("planner_node failed: %s", exc)
        return {
            "itinerary": f"Failed to generate itinerary: {exc}",
            "errors": errors + [f"planner: {exc}"],
        }
