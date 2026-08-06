
from typing import Any, Dict

from src.agent.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def reflection_node(state: AgentState) -> Dict[str, Any]:
    errors = state.get("errors", [])
    itinerary = state.get("itinerary")
    tool_results = state.get("tool_results", {})
    retry_count = state.get("retry_count", 0)
    reflection_notes: list[str] = list(state.get("reflection_notes", []))

    # --- 1. Check if planner succeeded ---
    if not itinerary or itinerary.startswith("Failed to generate itinerary"):
        reflection_notes.append("Itinerary generation failed.")
        if retry_count < 3:
            reflection_notes.append(f"Scheduling retry ({retry_count + 1}/3).")
            return {
                "reflection_notes": reflection_notes,
                "retry_count": retry_count + 1,
            }
        reflection_notes.append("Max retries reached; returning partial plan.")
        return {"reflection_notes": reflection_notes}

    # --- 2. Budget check ---
    cost_data = tool_results.get("cost") if isinstance(tool_results, dict) else None
    if cost_data and not cost_data.get("within_budget", True):
        overage = cost_data["breakdown"]["total"] - cost_data["budget"]
        recommendations = cost_data.get("recommendations", [])
        rec_text = "; ".join(recommendations) if recommendations else "No specific recommendations available."
        reflection_notes.append(
            f"Original cost estimate exceeds budget by {overage:,.0f} PKR. "
            f"To stay within budget: {rec_text}"
        )

    # --- 3. Data quality / missing tool results ---
    missing = []
    if not tool_results:
        missing.append("all tool results")
    else:
        if not tool_results.get("search"):
            missing.append("destination research")
        if not tool_results.get("places"):
            missing.append("places/attractions")
        if not tool_results.get("weather"):
            missing.append("weather forecast")
        if not tool_results.get("routing"):
            missing.append("routing info")
        if not tool_results.get("cost"):
            missing.append("cost estimate")

    if missing:
        reflection_notes.append(f"Missing data: {', '.join(missing)}. Itinerary may be incomplete.")

    # --- 4. Tool error summary ---
    if errors:
        reflection_notes.append(f"Tool failures encountered: {'; '.join(errors)}")

    if not reflection_notes:
        reflection_notes.append("Plan looks good.")

    logger.info("reflection: notes=%d errors=%d retry_count=%d", len(reflection_notes), len(errors), retry_count)
    return {"reflection_notes": reflection_notes}
