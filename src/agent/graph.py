
from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    cost_estimator_node,
    input_parser_node,
    places_node,
    planner_node,
    research_node,
    routing_node,
    weather_node,
)
from src.agent.reflection import reflection_node
from src.agent.state import AgentState


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("input_parser", input_parser_node)
    workflow.add_node("research", research_node)
    workflow.add_node("places", places_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("routing", routing_node)
    workflow.add_node("cost_estimator", cost_estimator_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("reflection", reflection_node)

    workflow.set_entry_point("input_parser")
    workflow.add_edge("input_parser", "research")
    workflow.add_edge("research", "places")
    workflow.add_edge("places", "weather")
    workflow.add_edge("weather", "routing")
    workflow.add_edge("routing", "cost_estimator")
    workflow.add_edge("cost_estimator", "planner")
    workflow.add_edge("planner", "reflection")
    workflow.add_edge("reflection", END)

    return workflow.compile()


graph = build_graph()
