"""
App-wide constants.

Kept deliberately small right now -- this file will grow as we build the
cost estimator and reflection node (budget thresholds, retry limits per
node, etc.). Putting them here instead of scattering magic numbers through
the codebase means one place to tune behaviour later.
"""

APP_VERSION = "0.1.0"

# Node names used in LangGraph state/logging (populated as nodes are built).
class NodeNames:
    INPUT_PARSER = "input_parser"
    DESTINATION_RECOMMENDATION = "destination_recommendation"
    RESEARCH = "research"
    PLACES = "places"
    WEATHER = "weather"
    ROUTING = "routing"
    HOTEL = "hotel"
    COST_ESTIMATOR = "cost_estimator"
    PLANNER = "planner"
    REFLECTION = "reflection"
