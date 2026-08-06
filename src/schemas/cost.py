"""
Cost estimation data models.

Unlike the other tools (which keep their Pydantic models co-located in
the tool file), these live here because CostBreakdown/CostEstimateResult
get reused directly by the POST /estimate-cost API response and by the
reflection node's budget checks -- they're a shared data contract, not
something internal to cost_estimator.py.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TransportEstimate(BaseModel):
    mode: str
    price_per_km: float
    distance_km: float
    total_pkr: float
    description: str


class CostBreakdown(BaseModel):
    hotel: float
    transport: float
    food: float
    activities: float
    miscellaneous: float
    total: float


class CostEstimateResult(BaseModel):
    currency: str = "PKR"
    breakdown: CostBreakdown
    budget: float
    remaining_budget: float
    budget_percentage_used: float  # can exceed 100 if over budget
    within_budget: bool
    recommendations: List[str] = Field(default_factory=list)
    transport_estimates: List[TransportEstimate] = Field(default_factory=list)
