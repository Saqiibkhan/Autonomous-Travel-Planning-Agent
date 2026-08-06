"""
Application entrypoint.

Run locally with:
    uvicorn src.main:app --reload

This file's only job is to build the FastAPI app object and wire in
routers and static files. Business logic (tools, the LangGraph workflow, etc.)
intentionally does NOT live here.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import router as system_router
from src.config import settings
from src.utils.constants import APP_VERSION
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s (env=%s, model=%s)",
        settings.app_name,
        APP_VERSION,
        settings.app_env,
        settings.llm_model,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=APP_VERSION,
    description=(
        "An autonomous agent that researches a destination, checks weather "
        "and routes, estimates costs, and produces a budget-aware, "
        "day-by-day travel itinerary with reasoning behind every pick."
    ),
    lifespan=lifespan,
)

app.include_router(system_router)

# Serve the simple frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
