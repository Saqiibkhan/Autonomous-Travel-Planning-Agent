
from typing import List, Optional

from pydantic import BaseModel, Field


class TransportEstimate(BaseModel):
    mode: str
    price_per_km: float
    distance_km: float
    total_pkr: float
    description: str


class PlanRequest(BaseModel):
    destination: str = Field(..., description="City or region to visit")
    budget: float = Field(..., gt=0, description="Total budget in local currency")
    duration_days: int = Field(..., gt=0, le=30, description="Trip length in days")
    num_travelers: int = Field(default=1, ge=1, le=20)
    origin: Optional[str] = Field(default=None, description="Departure city (optional)")
    preferences: Optional[str] = Field(default=None, description="Free-text preferences / constraints")


class CostEstimateRequest(BaseModel):
    budget: float = Field(..., gt=0)
    duration_days: int = Field(..., gt=0, le=30)
    num_travelers: int = Field(default=1, ge=1, le=20)
    transport_distance_km: Optional[float] = Field(default=None, ge=0)
    transport_mode: str = Field(default="car", pattern="^(bus|car|bike|driving|public)$")
    hotel_tier: str = Field(default="mid-range", pattern="^(budget|mid-range|luxury)$")
    food_tier: str = Field(default="mid-range", pattern="^(budget|mid-range|luxury)$")
    activity_tier: str = Field(default="mid-range", pattern="^(budget|mid-range|luxury)$")


class PlanResponse(BaseModel):
    destination: str
    budget: float
    duration_days: int
    itinerary: str
    reflection_notes: List[str]
    errors: List[str]
    tool_results: Optional[dict] = None


class CostEstimateResponse(BaseModel):
    currency: str
    breakdown: dict
    budget: float
    remaining_budget: float
    budget_percentage_used: float
    within_budget: bool
    recommendations: List[str]
    transport_estimates: List[TransportEstimate] = Field(default_factory=list)
