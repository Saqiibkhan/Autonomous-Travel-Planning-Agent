"""
Custom exceptions for tool failures.

Every tool wraps its failures in one of these instead of letting raw
httpx/network exceptions escape. This matters later: the reflection node
(Task 6) needs to distinguish "a tool cleanly failed and told us why"
from an unexpected bug, so it can decide whether to retry, substitute,
or flag the shortfall in the final output.
"""


class ToolError(Exception):
    """Base class for all tool-level failures."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"[{tool_name}] {message}")


class SearchToolError(ToolError):
    def __init__(self, message: str):
        super().__init__("search_tool", message)


class PlacesToolError(ToolError):
    def __init__(self, message: str):
        super().__init__("places_tool", message)


class WeatherToolError(ToolError):
    def __init__(self, message: str):
        super().__init__("weather_tool", message)


class RoutingToolError(ToolError):
    def __init__(self, message: str):
        super().__init__("routing_tool", message)


class CostEstimatorError(ToolError):
    def __init__(self, message: str):
        super().__init__("cost_estimator", message)
