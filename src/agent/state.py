
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class UserInput(BaseModel):
    destination: str
    budget: float
    duration_days: int
    num_travelers: int = 1
    origin: Optional[str] = None
    preferences: Optional[str] = None


class ToolResults(BaseModel):
    search: Optional[dict] = None
    places: Optional[dict] = None
    weather: Optional[dict] = None
    routing: Optional[dict] = None
    cost: Optional[dict] = None


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "conversation history"]
    user_input: Optional[UserInput]
    tool_results: ToolResults
    itinerary: Optional[str]
    errors: Annotated[List[str], "tool/validation errors encountered"]
    retry_count: int
    reflection_notes: List[str]
