# Autonomous Travel Planning Agent — Full Project Documentation

> Status: **Complete** — FastAPI backend with a LangGraph agent workflow,
> a reflection node, `/plan` and `/estimate-cost` endpoints, a full test
> suite, and a simple web frontend.

This document is the complete, end-to-end reference for the project. It
covers what the project does, why it is built the way it is, how every
part works, how a request flows through the whole system, how the
external APIs are used, and how each piece is costed, tested, and run.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Application Startup](#3-application-startup)
4. [Request Lifecycle](#4-request-lifecycle)
5. [LangGraph Explained](#5-langgraph-explained)
6. [Every Agent Node](#6-every-agent-node)
7. [State Object](#7-state-object)
8. [Tools](#8-tools)
9. [Services Layer](#9-services-layer)
10. [Configuration](#10-configuration)
11. [API Reference](#11-api-reference)
12. [Cost Estimation Logic](#12-cost-estimation-logic)
13. [Error Handling](#13-error-handling)
14. [Frontend](#14-frontend)
15. [Testing](#15-testing)
16. [Sequence Diagram](#16-sequence-diagram)
17. [External APIs](#17-external-apis)
18. [Complete End-to-End Example](#18-complete-end-to-end-example)

---

## 1. Project Overview

### What problem does the project solve?

The project is an **autonomous travel planning agent**. Given:

- a **destination** (e.g. "Hunza", "Murree", "Paris"),
- a **budget**,
- a **trip duration**,

…it researches the destination, discovers attractions, checks the
weather, computes routes, estimates the total cost, and finally generates
a **day-by-day itinerary**. Critically, it gives a *reason* behind every
recommendation and honestly admits when data is missing — rather than
inventing facts.

It was built for the **AI Summer Internship 2026 travel-agent task**.

### Why LangGraph instead of a simple chain?

A simple chain runs functions in a fixed sequence and returns the final
result. LangGraph is a **state machine** framework that models the agent
as a **graph of nodes** that share a **typed, mutable state object**. This
was chosen because:

1. **Shared state across steps.** Each node reads the results of previous
   nodes and writes new results. With a chain you pass data explicitly
   between steps; with a graph you have one shared state that every node
   reads and writes. This is cleaner for a workflow where later steps
   (like the planner) need data from *all* earlier steps (research,
   places, weather, routing, cost).

2. **Retry / reflection is natural.** The reflection node inspects the
   state for problems (budget overrun, missing data, tool failures) and
   can decide to **retry** or flag shortfalls. This kind of conditional,
   cyclic behaviour is awkward in a linear chain but native to a graph.

3. **Extensibility.** Adding a node (e.g. a hotel tool) is just adding a
   node and an edge. The graph is easy to visualise and extend.

4. **Async support.** `graph.ainvoke()` runs the whole workflow
   asynchronously, which fits FastAPI.

### Overall architecture

```
                    ┌────────────────────────────────────────────┐
                    │               FastAPI (src/main.py)        │
                    │  /health  /status  /plan  /estimate-cost   │
                    └────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌────────────────────────────────────────────┐
                    │            LangGraph Workflow              │
                    │  input_parser → research → places →        │
                    │  weather → routing → cost_estimator →      │
                    │  planner → reflection                       │
                    └────────────────────────────────────────────┘
                                       │
             ┌───────────┬─────────────┼──────────────┬──────────┐
             ▼           ▼             ▼              ▼          ▼
       Tavily     Nominatim+    OpenWeather   OpenRouteService   (none)
       (Search)    Overpass        (Forecast)      (Routing)    Cost Estimator
                  (Geocoding & Places)                          (pure Python)
```

---

## 2. Folder Structure

```
travel-agent/
├── src/                        # All application source code
│   ├── main.py                 # FastAPI app entrypoint (starts the app)
│   ├── config.py               # Pydantic Settings (all config/env vars)
│   ├── agent/                  # LangGraph agent orchestration
│   │   ├── state.py            # AgentState TypedDict + UserInput schema
│   │   ├── nodes.py            # Workflow node implementations
│   │   ├── reflection.py       # Critic/reflection node
│   │   └── graph.py            # Builds + compiles the LangGraph
│   ├── tools/                  # Each external capability as a "tool"
│   │   ├── search_tool.py      # Destination research (Tavily)
│   │   ├── places_tool.py      # Attractions/restaurants (OSM/Overpass)
│   │   ├── weather_tool.py     # Day-by-day forecast (OpenWeatherMap)
│   │   ├── routing_tool.py     # Driving/walking distance (OpenRouteService)
│   │   └── cost_estimator.py   # Cost math + budget checks (no API)
│   ├── services/               # Shared infrastructure services
│   │   ├── api_clients.py      # Shared async httpx client singleton
│   │   ├── geocoding.py        # Shared destination→(lat,lon) lookup
│   │   └── retry.py            # Generic async retry decorator
│   ├── schemas/                # Shared Pydantic data contracts
│   │   ├── cost.py             # CostBreakdown / CostEstimateResult
│   │   └── plan.py             # PlanRequest / PlanResponse / CostEstimateRequest / Response
│   ├── utils/                  # Cross-cutting helpers
│   │   ├── logger.py           # Structured logging setup
│   │   ├── constants.py        # App version + node name constants
│   │   └── exceptions.py       # Custom tool exception hierarchy
│   ├── api/                    # HTTP layer
│   │   └── routes.py           # All route handlers
│   └── data/                   # Static data used by tools
│       └── transport_rates.json # Per-km transport pricing
├── tests/                      # pytest suite (10 test files)
├── docs/                       # Documentation
│   └── api_research.md         # Why each external API was chosen
├── frontend/                   # Static web frontend
│   └── index.html              # Single-page planner UI
├── requirements.txt            # Python dependencies
├── pytest.ini                  # pytest config
├── .gitignore                  # Git ignore rules
└── README.md                   # Quick-start guide
```

### Every important file explained

| File | Role | Kind |
|------|------|------|
| `src/main.py` | Builds the FastAPI app, wires routers + static files | **Startup** |
| `src/config.py` | Loads + validates all settings from `.env`/env vars | **Configuration** |
| `src/agent/graph.py` | Defines nodes, edges, entry point, compiles graph | **Business logic** |
| `src/agent/nodes.py` | Implements all 7 workflow node functions | **Business logic** |
| `src/agent/reflection.py` | Implements the reflection/critic node | **Business logic** |
| `src/agent/state.py` | Defines `AgentState` and `UserInput` | **Data contract** |
| `src/api/routes.py` | HTTP handlers for the 4 endpoints | **HTTP layer** |
| `src/tools/*.py` | Each external capability wrapper | **Business logic** |
| `src/services/*.py` | Shared HTTP client, geocoding, retry | **Infrastructure** |
| `src/schemas/*.py` | Request/response Pydantic models | **Data contract** |
| `src/utils/*.py` | Logging, constants, exceptions | **Infrastructure** |
| `src/data/transport_rates.json` | Per-km transport pricing data | **Data** |
| `frontend/index.html` | Browser UI that calls `/plan` | **Frontend** |

**Startup files:** `src/main.py` (the entrypoint executed by uvicorn).

**Configuration files:** `src/config.py`, `.env` (not committed), `.gitignore`,
`pytest.ini`, `requirements.txt`.

**Business logic:** `src/agent/*`, `src/tools/*`, `src/api/routes.py`,
`src/services/*`, `src/schemas/*`.

---

## 3. Application Startup

What happens when you run:

```bash
uvicorn src.main:app --reload
```

1. **`src/main.py` is loaded first.** This is the module uvicorn imports.

2. **Imports are resolved.** Importing `src.main` triggers a chain of
   imports:
   - `from src.api.routes import router` → imports `src.agent.graph`
     → imports `src.agent.nodes`, `src.agent.reflection`, `src.agent.state`
     → imports all the tools, services, schemas, config, utils.
   - `from src.config import settings` → runs `get_settings()` (cached)
     → reads `.env` / environment variables **once** at import time.
   - `from src.utils.constants import APP_VERSION`.
   - `from src.utils.logger import get_logger`.

3. **The LangGraph agent is initialized.** When `src.agent.graph` is
   imported, the module-level statement `graph = build_graph()` runs. This
   constructs a `StateGraph(AgentState)`, registers all 8 nodes, wires the
   edges, sets the entry point, and calls `.compile()`. So **the graph is
   compiled once at import time**, before the server starts accepting
   requests.

4. **The FastAPI app object is created.** `app = FastAPI(...)` with the
   app title, version, description, and a `lifespan` context manager.

5. **Routers are registered.** `app.include_router(system_router)` adds
   the routes from `src/api/routes.py`.

6. **Static files are mounted.** `app.mount("/frontend", StaticFiles(...))`
   serves the frontend HTML at `/frontend`.

7. **The lifespan context manager runs.** On startup it logs the app
   name, version, environment, and LLM model. On shutdown it logs a
   message.

8. **uvicorn starts serving.** The app is now live and listens for
   requests.

> **Key point:** The heavy lifting (graph compilation, settings loading,
> HTTP client creation) happens *lazily/at import*. The shared HTTP client
> is created the first time a tool calls `get_http_client()`.

---

## 4. Request Lifecycle

This is the most important flow in the project. Here is exactly what
happens when a user submits a plan.

```
User
  │  (fills the form, clicks "Plan Trip")
  ▼
Frontend (index.html)
  │  fetch('/plan', {method:'POST', body: JSON.stringify(payload)})
  ▼
FastAPI (uvicorn)
  │  routes request to the registered router
  ▼
Router (src/api/routes.py → plan_trip)
  │  validates body against PlanRequest (422 if invalid)
  │  builds initial AgentState
  ▼
LangGraph Agent (graph.ainvoke(initial_state))
  │
  ├──► input_parser     (validate user input)
  ├──► research         (Tavily → destination overview)
  ├──► places           (geocode + Overpass → attractions)
  ├──► weather          (geocode + OpenWeather → forecast)
  ├──► routing          (geocode origin + OpenRouteService → distance)
  ├──► cost_estimator   (pure Python → cost breakdown)
  ├──► planner          (LLM → day-by-day itinerary text)
  └──► reflection       (critic: budget, missing data, retries)
  │
  ▼
final_state
  │  extracts itinerary, reflection_notes, errors, tool_results
  ▼
PlanResponse (JSON)
  │
  ▼
Frontend
  │  renders itinerary + reflection notes
  ▼
User sees the final plan
```

By the end of the flow, the user receives a JSON object containing the
generated itinerary text, any reflection notes, any errors, and the raw
tool results.

---

## 5. LangGraph Explained

### The graph

`src/agent/graph.py` builds a `StateGraph`. A LangGraph workflow is
composed of:

- **State** — a typed dict (`AgentState`) that nodes read from and write to.
- **Nodes** — plain functions that take the state and return a partial
  update to merge back into the state.
- **Edges** — define the order nodes run in.
- **START / END** — special marker nodes for the entry and exit points.
- **Conditional edges** — edges that choose the next node based on state
  (not used here; the flow is linear except for the retry logic inside the
  reflection node).

### The node order

```
START
  │
  ▼
input_parser
  │
  ▼
research
  │
  ▼
places
  │
  ▼
weather
  │
  ▼
routing
  │
  ▼
cost_estimator
  │
  ▼
planner
  │
  ▼
reflection
  │
  ▼
END
```

### Why this order was chosen

1. **input_parser first** — validates/normalises the user's request before
   anything else runs.
2. **research** — gathers general destination knowledge (overview,
   culture, best season, tips).
3. **places** — needs the destination; discovers attractions/coordinates.
4. **weather** — needs the destination coordinates (urn from geocoding).
5. **routing** — needs the destination coordinates (from places) and the
   origin; computes distance.
6. **cost_estimator** — needs the routing distance (to compute transport
   cost) and the user's budget/travelers.
7. **planner** — needs *all* of the above (research, places, weather,
   routing, cost) to generate a smart, grounded itinerary via the LLM.
8. **reflection** — reviews the generated plan for budget issues, missing
   data, and tool failures, and decides whether to retry.

This is a **dependency-ordered pipeline**: each node consumes the outputs
of the nodes before it.

### The reflection node

The reflection node is the "critic". It:

1. Checks whether the planner succeeded (if not, it can schedule a retry,
   up to 3 times).
2. Checks the budget — if the cost estimate exceeds the budget, it records
   the overage and the cost estimator's recommendations.
3. Checks data quality — lists which tool results are missing.
4. Summarises any tool errors.
5. Appends a final note (e.g. "Plan looks good.").

It does **not** use a conditional graph edge to retry; instead it records
a `reflection_notes` entry and increments `retry_count`. The graph always
continues to `END`. (The retry mechanics are represented in state, ready
for future wiring.)

---

## 6. Every Agent Node

All node functions live in `src/agent/nodes.py` (except reflection, which
is in `src/agent/reflection.py`). They are registered in
`src/agent/graph.py`.

### 6.1 `input_parser_node`

- **Purpose:** Validate that user input exists.
- **Inputs:** `state["user_input"]`.
- **Outputs:** If input is missing, appends an error to `state["errors"]`.
  Otherwise logs the request and returns the `user_input`.
- **APIs called:** None (pure).
- **State modified:** `errors` (on failure).
- **Possible failures:** Missing `user_input`.
- **Retry strategy:** None.

### 6.2 `research_node`

- **Purpose:** Research the destination (overview, culture, best season,
  tips, attractions mentioned).
- **Inputs:** `state["user_input"].destination`.
- **Outputs:** `tool_results["search"]` (a `SearchResult` dict).
- **APIs called:** Tavily (3 queries).
- **State modified:** `tool_results["search"]`, `errors` (on failure).
- **Possible failures:** Missing API key, destination not found, Tavily
  unreachable/error.
- **Retry strategy:** `@async_retry` on the low-level `_tavily_query`.

### 6.3 `places_node`

- **Purpose:** Discover attractions, restaurants, museums, landmarks near
  the destination.
- **Inputs:** `state["user_input"].destination`.
- **Outputs:** `tool_results["places"]` (a `PlacesResult` dict).
- **APIs called:** Nominatim (geocode) → Overpass (POIs).
- **State modified:** `tool_results["places"]`, `errors` (on failure).
- **Possible failures:** Geocoding failure, no places found, OSM errors.
- **Retry strategy:** `@async_retry` on `_query_overpass`.

### 6.4 `weather_node`

- **Purpose:** Get a day-by-day forecast for the trip.
- **Inputs:** `state["user_input"].destination`, `.duration_days`.
- **Outputs:** `tool_results["weather"]` (a `WeatherResult` dict).
- **APIs called:** Nominatim (geocode) → OpenWeatherMap forecast.
- **State modified:** `tool_results["weather"]`, `errors` (on failure).
- **Possible failures:** Missing key, geocoding failure, no forecast data.
- **Retry strategy:** `@async_retry` on `_fetch_forecast`.

### 6.5 `routing_node`

- **Purpose:** Compute driving/walking distance & duration from origin to
  destination.
- **Inputs:** `state["user_input"].origin`, plus `tool_results["places"]`
  (for destination coordinates).
- **Outputs:** `tool_results["routing"]` (a `RouteResult` dict).
- **APIs called:** Nominatim (geocode origin) → OpenRouteService.
- **State modified:** `tool_results["routing"]`, `errors` (on failure).
- **Possible failures:** No places data, no origin, missing key, route
  failure.
- **Retry strategy:** `@async_retry` on `_fetch_route`. Walking failure
  degrades gracefully (recorded as a warning, not a hard error).

### 6.6 `cost_estimator_node`

- **Purpose:** Estimate the total trip cost and check budget.
- **Inputs:** `state["user_input"]` (budget, days, travelers) and the
  routing distance from `tool_results["routing"]`.
- **Outputs:** `tool_results["cost"]` (a `CostEstimateResult` dict).
- **APIs called:** None (pure Python).
- **State modified:** `tool_results["cost"]`, `errors` (on failure).
- **Possible failures:** Invalid inputs (budget ≤ 0, etc.).
- **Retry strategy:** None.

### 6.7 `planner_node`

- **Purpose:** Generate the final day-by-day itinerary using the LLM.
- **Inputs:** All `tool_results` (search, places, weather, routing, cost)
  plus the user's preferences and any errors.
- **Outputs:** `itinerary` (text) and appends messages to `state["messages"]`.
- **APIs called:** The LLM (OpenRouter/OpenAI-compatible `ChatOpenAI`).
- **State modified:** `itinerary`, `messages`, `errors` (on failure).
- **Possible failures:** LLM call failure (no key, network, rate limit).
- **Retry strategy:** None (a failure produces a fallback itinerary string
  and an error; the reflection node handles retry decisions).

### 6.8 `reflection_node`

- **Purpose:** Review the plan for quality.
- **Inputs:** `state["itinerary"]`, `state["tool_results"]`,
  `state["errors"]`, `state["retry_count"]`.
- **Outputs:** `reflection_notes`, `retry_count`.
- **APIs called:** None.
- **State modified:** `reflection_notes`, `retry_count`.
- **Possible failures:** None (pure logic).
- **Retry strategy:** If the itinerary failed, increments `retry_count`
  (up to 3) and records a retry note.

---

## 7. State Object

The `AgentState` (in `src/agent/state.py`) is a `TypedDict`. Conceptually:

```python
state = {
    "messages": [],            # LLM conversation history
    "user_input": {            # the validated user request
        "destination": "Murree",
        "budget": 50000,
        "duration_days": 5,
        "num_travelers": 2,
        "origin": "Lahore",
        "preferences": None,
    },
    "tool_results": {          # results from each tool
        "search":    {...},    #   set by research role
        "places":    {...},    #   set by places role
        "weather":   {...},    #   set by weather role
        "routing":   {...},    #   set by routing role
        "cost":      {...},    #   set by cost_estimator role
    },
    "itinerary": None,         # set by planner role
    "errors": [],              # tool/validation errors
    "retry_count": 0,          # incremented by reflection role
    "reflection_notes": [],    # critic output
}
```

(The `UserInput`, `ToolResults`, and `AgentState` schemas are defined in
`src/agent/state.py`.)

### How the state changes after each node

| Step | Node | New state written |
|------|------|-------------------|
| 1 | `input_parser` | confirms `user_input` (or adds `errors`) |
| 2 | `research` | `tool_results["search"]` |
| 3 | `places` | `tool_results["places"]` |
| 4 | `weather` | `tool_results["weather"]` |
| 5 | `routing` | `tool_results["routing"]` |
| 6 | `cost_estimator` | `tool_results["cost"]` |
| 7 | `planner` | `itinerary`, `messages` |
| 8 | `reflection` | `reflection_notes`, `retry_count` |

---

## 8. Tools

Each tool is a self-contained module under `src/tools/`. They share a
common design: an async public function returning a typed Pydantic result,
wrapped low-level API calls decorated with `@async_retry`, and custom
exception types on failure.

### 8.1 Search Tool (`search_tool.py`)

- **Why Tavily?** Tavily returns a synthesized `answer` field from multiple
  sources, so this tool stays a *pure data* tool (no extra LLM call to
  summarise results). Free tier: 1,000 searches/month, no credit card.
- **Inputs:** `destination` string.
- **Outputs:** `SearchResult` with `overview`, `culture`, `best_season`,
  `travel_tips`, `attractions_mentioned`, `sources`.
- **How it works:** Runs **three** targeted Tavily queries (travel guide,
  culture, best-time-to-visit) to get balanced fields, then splits the
  guidance into short `travel_tips`.
- **Error handling:** Raises `SearchToolError` for missing key, empty
  destination, no results, API errors, or unavailability.

### 8.2 Places Tool (`places_tool.py`)

- **Why OSM (Nominatim + Overpass)?** Zero API key, zero billing. Overpass
  queries multiple categories in one request.
- **Inputs:** `destination`, optional `radius_meters` (default 15,000 m).
- **Outputs:** `PlacesResult` with `destination`, `latitude`, `longitude`,
  and a list of `PlaceItem` (name, category, coordinates, address).
- **How it works:** geocodes the destination via Nominatim, then queries
  Overpass for tourism attractions, museums, viewpoints, historic sites,
  restaurants, and cafes around those coordinates.
- **Error handling:** Raises `PlacesToolError` for geocoding failure, no
  places found, or OSM errors.

### 8.3 Weather Tool (`weather_tool.py`)

- **Why OpenWeatherMap 5-day/3-hour forecast?** Free key, no credit card;
  returns `pop` (rain probability) directly.
- **Inputs:** `destination`, `duration_days`.
- **Outputs:** `WeatherResult` with `forecast` (list of `DailyWeather`:
  date, min/max temp, condition, rainfall probability, warnings) and
  `notes`.
- **How it works:** geocodes the destination, fetches the forecast,
  groups the 3-hour blocks by calendar day, computes daily min/max,
  dominant condition, and max rain probability, and adds warnings.
- **Error handling:** Raises `WeatherToolError` for missing key, no data,
  geocoding failure, or API errors. For trips > 5 days, a note explains
  the forecast only covers the first 5 days.

### 8.4 Routing Tool (`routing_tool.py`)

- **Why OpenRouteService?** Free tier (2,000 req/day), no credit card;
  supports driving-car and foot-walking profiles.
- **Inputs:** `origin` (lat, lon), `destination` (lat, lon).
- **Outputs:** `RouteResult` with driving/walking distance (km) and
  duration (min), plus warnings.
- **How it works:** calls ORS for `driving-car`, then `foot-walking`.
- **Error handling:** Raises `RoutingToolError` for missing key or driving
  failure. A failed *walking* route degrades gracefully into a warning.

### 8.5 Cost Estimator (`cost_estimator.py`)

- **Why no API?** Costing is pure math against local rates — no external
  service needed.
- **Inputs:** budget, duration, travelers, tier overrides, optional
  transport distance/mode, optional activity costs.
- **Outputs:** `CostEstimateResult` with a `CostBreakdown` (hotel,
  transport, food, activities, miscellaneous, total), budget-vs-total
  fields, `within_budget`, recommendations, and per-mode transport
  estimates.
- **How it works:** See [Cost Estimation Logic](#12-cost-estimation-logic).
- **Error handling:** Raises `CostEstimatorError` for invalid inputs
  (budget ≤ 0, duration ≤ 0, travelers ≤ 0, invalid tier).

---

## 9. Services Layer

The services layer exists to **share infrastructure** across tools so
that each tool does not duplicate HTTP-client management, geocoding, or
retry logic.

```
Agent
  │
  ▼
Weather Tool
  │
  ▼
Geocoding Service ──► HTTP Client ──► Nominatim / OpenWeather
                               │
                               └──► retry decorator wraps each call
```

### `api_clients.py`

A single shared `httpx.AsyncClient` singleton built with the configured
timeout. Reusing one client enables **connection pooling** when the
LangGraph workflow calls multiple tools back to back. `close_http_client()`
is meant to be called on app shutdown to release the pool.

### `geocoding.py`

A shared `geocode(destination)` that resolves a place name to
`(latitude, longitude)` via **Nominatim** (OpenStreetMap). It is reused
by the Places, Weather, and Routing tools. Raises `GeocodingError` if the
destination cannot be resolved.

### `retry.py`

A generic `@async_retry` decorator that retries transient network/HTTP
failures (`httpx.TimeoutException`, `httpx.ConnectError`,
`httpx.HTTPStatusError`) up to `settings.max_tool_retries` times with
growing backoff, logging each attempt. Every tool's low-level call is
wrapped with it.

---

## 10. Configuration

### `.env` and Pydantic Settings

All configuration lives in `src/config.py` using **Pydantic Settings**
(`BaseSettings`). It reads from a `.env` file first, then environment
variables (which always win), and validates/type-casts everything.

```python
class Settings(BaseSettings):
    app_name: str = "Autonomous Travel Planning Agent"
    app_env: str = "development"
    log_level: str = "INFO"
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-20b:free"
    search_api_key: str = ""
    places_api_key: str = ""
    weather_api_key: str = ""
    routing_api_key: str = ""
    currency_api_key: str = ""
    request_timeout_seconds: float = 15.0
    max_tool_retries: int = 3
    max_reflection_retries: int = 3
```

A cached `get_settings()` returns the same `Settings` object every call,
so the `.env` is read only **once per process**. The convenience singleton
`settings` is imported by most modules.

### Key configuration values

| Key | Purpose |
|-----|---------|
| `llm_api_key` / `llm_base_url` / `llm_model` | LLM provider (OpenRouter) and model |
| `search_api_key` | Tavily key |
| `weather_api_key` | OpenWeatherMap key |
| `routing_api_key` | OpenRouteService key |
| `request_timeout_seconds` | Shared HTTP timeout for all tools |
| `max_tool_retries` | Retry count for tool API calls |
| `max_reflection_retries` | Max retries for the reflection node |

> **Note:** `.env.example` is referenced in the README as the template for
> `.env`. The `.env` file is git-ignored (secrets are never committed).

---

## 11. API Reference

All endpoints are defined in `src/api/routes.py`. The schemas are in
`src/schemas/plan.py`.

### 11.1 `GET /health`

- **URL:** `/health`
- **Method:** `GET`
- **Purpose:** Liveness check.
- **Request model:** none.
- **Response:**

```json
{
  "status": "ok"
}
```

- **Possible HTTP errors:** none (always 200).

### 11.2 `GET /status`

- **URL:** `/status`
- **Method:** `GET`
- **Purpose:** Readiness / config presence check (never leaks secret values).
- **Request model:** none.
- **Response:**

```json
{
  "app_name": "Autonomous Travel Planning Agent",
  "version": "0.1.0",
  "environment": "development",
  "llm_model": "openai/gpt-oss-20b:free",
  "configured_keys": {
    "llm_api_key": false,
    "search_api_key": false,
    "places_api_key": false,
    "weather_api_key": false,
    "routing_api_key": false,
    "currency_api_key": false
  }
}
```

- **Possible HTTP errors:** none (always 200).

### 11.3 `POST /plan`

- **URL:** `/plan`
- **Method:** `POST`
- **Purpose:** Run the full LangGraph workflow and return an itinerary.
- **Request model (`PlanRequest`):**

```json
{
  "destination": "Hunza",
  "budget": 150000,
  "duration_days": 5,
  "num_travelers": 2,
  "origin": "Karachi",
  "preferences": "avoid hiking, prefer museums"
}
```

- **Validation rules:**
  - `destination`: required string.
  - `budget`: required, `> 0`.
  - `duration_days`: required, `> 0` and `<= 30`.
  - `num_travelers`: default `1`, `>= 1` and `<= 20`.
  - `origin`: optional string.
  - `preferences`: optional string.

- **Response model (`PlanResponse`):**

```json
{
  "destination": "Hunza",
  "budget": 150000,
  "duration_days": 5,
  "itinerary": "Day 1: ...",
  "reflection_notes": ["Plan looks good."],
  "errors": [],
  "tool_results": {
    "search": { ... },
    "places": { ... },
    "weather": { ... },
    "routing": { ... },
    "cost": { ... }
  }
}
```

- **Possible HTTP errors:**
  - `422 Unprocessable Entity` — validation failure (e.g. budget ≤ 0,
    duration out of range, missing destination).

### 11.4 `POST /estimate-cost`

- **URL:** `/estimate-cost`
- **Method:** `POST`
- **Purpose:** Run **only** the cost estimator (no agent/tools).
- **Request model (`CostEstimateRequest`):**

```json
{
  "budget": 100000,
  "duration_days": 3,
  "num_travelers": 2,
  "transport_distance_km": 200,
  "transport_mode": "driving",
  "hotel_tier": "mid-range",
  "food_tier": "mid-range",
  "activity_tier": "mid-range"
}
```

- **Validation rules:**
  - `budget`: required, `> 0`.
  - `duration_days`: required, `> 0`, `<= 30`.
  - `num_travelers`: default `1`, `>= 1`, `<= 20`.
  - `transport_distance_km`: optional, `>= 0`.
  - `transport_mode`: default `car`, must match
    `^(bus|car|bike|driving|public)$`.
  - `hotel_tier` / `food_tier` / `activity_tier`: default `mid-range`,
    must match `^(budget|mid-range|luxury)$`.

- **Response model (`CostEstimateResponse`):**

```json
{
  "currency": "PKR",
  "breakdown": {
    "hotel": 32000,
    "transport": 6000,
    "food": 15000,
    "activities": 12000,
    "miscellaneous": 6500,
    "total": 71500
  },
  "budget": 100000,
  "remaining_budget": 28500,
  "budget_percentage_used": 71.5,
  "within_budget": true,
  "recommendations": [],
  "transport_estimates": []
}
```

- **Possible HTTP errors:**
  - `422 Unprocessable Entity` — validation failure.

---

## 12. Cost Estimation Logic

The cost estimator (`src/tools/cost_estimator.py`) computes the total
trip cost from **local PKR/day heuristics** loaded from
`src/data/transport_rates.json`. It is pure Python (no external API).

### The formulas

```
nights = max(duration_days - 1, 0)
rooms  = ceil(num_travelers / 2)          # double occupancy

hotel_cost  = HOTEL_NIGHTLY_RATES[tier] * nights * rooms
food_cost   = FOOD_DAILY_RATES[tier] * duration_days * num_travelers
activities  = ACTIVITY_DAILY_RATES[tier] * duration_days   # or sum(activity_costs)
transport   = TRANSPORT_RATE_PER_KM[mode]["price_per_km"] * distance_km [* travelers for bus/public]

subtotal       = hotel_cost + food_cost + activities + transport
miscellaneous  = subtotal * 0.10          # DEFAULT_MISC_PERCENTAGE
total          = subtotal + miscellaneous

remaining_budget      = budget - total
budget_percentage_used = (total / budget) * 100
within_budget          = total <= budget
```

### Per-category rates (loaded from `transport_rates.json`)

| Mode | Price/km | Description |
|------|----------|-------------|
| bus | 6 | Public bus / coach |
| car | 35 | Private car (fuel + maintenance) |
| bike | 12 | Motorcycle / bike |
| driving | 30 | Legacy alias for car |
| public | 12 | Legacy alias |

### Daily/heuristic PKR rates (in code)

| Tier | Hotel/night | Food/day | Activities/day |
|------|------------|----------|----------------|
| budget | 3,500 | 1,200 | 800 |
| mid-range | 8,000 | 2,500 | 2,000 |
| luxury | 18,000 | 5,000 | 5,000 |

### Recommendations when over budget

If `total > budget`, the estimator builds concrete, quantified suggestions:

- **Hotel tier downgrade** — e.g. "Switch hotel tier from 'luxury' to
  'mid-range': ~X PKR saved over N night(s)". Only if the hotel used the
  tier default (not an explicit rate).
- **Food tier downgrade** — similar, only if the food tier default was used.
- **Trim activities** — reduce activities by 25% (fewer/cheaper paid
  attractions).
- **Switch transport to bus** — if using a private car and a bus is
  cheaper.
- **Shortfall notice** — if even all suggestions can't close the gap, it
  explicitly says by how much and suggests a shorter trip, a closer
  destination, or a higher budget.

### Transport estimates

When a `transport_distance_km` is provided, the estimator returns per-mode
costs for all available modes (bus, car, bike) so the planner can show
options.

---

## 13. Error Handling

The project has a layered error-handling strategy.

### Exception hierarchy (`src/utils/exceptions.py`)

```
ToolError (base)
 ├── SearchToolError
 ├── PlacesToolError
 ├── WeatherToolError
 ├── RoutingToolError
 └── CostEstimatorError
```

Tools wrap raw `httpx`/network exceptions in these types so the rest of
the app can distinguish "a tool cleanly failed and told us why" from an
unexpected bug.

### Network failures

- Low-level API calls are wrapped with `@async_retry`, which retries
  `httpx.TimeoutException`, `httpx.ConnectError`, and `httpx.HTTPStatusError`
  up to `max_tool_retries` times with growing backoff.
- After retries are exhausted, the tool raises its custom exception.

### Invalid city / no data

- **Geocoding** raises `GeocodingError` ("Could not geocode destination").
- **Places** raises `PlacesToolError` if no places are found.
- **Weather** raises `WeatherToolError` if no forecast data is returned.
- **Search** raises `SearchToolError` if no results are found.

### Missing API key

Each tool checks its key and raises its custom error if unset
(e.g. `SEARCH_API_KEY is not configured.`, `WEATHER_API_KEY is not
configured.`).

### Validation errors

- **FastAPI** returns `422` automatically when a request body fails
  Pydantic validation (e.g. budget ≤ 0, invalid tier, duration out of
  range).
- The **cost estimator** raises `CostEstimatorError` for invalid inputs.

### How tool failures propagate

Each LangGraph node wraps its tool call in a `try/except`, appends the
error to `state["errors"]`, and continues with the rest of the graph. The
**reflection node** then summarises these errors into `reflection_notes`
so the user sees what went wrong.

---

## 14. Frontend

The frontend is a **single static HTML page** at `frontend/index.html`,
served by FastAPI at `/frontend`.

### HTML

- A form with inputs: destination, budget, duration, travelers, origin
  (optional), preferences (optional), and a "Plan Trip" button.
- A hidden result card that displays the itinerary and reflection notes.

### JavaScript

- On form submit, `fetch('/plan', { method: 'POST', headers: { 'Content-Type':
  'application/json' }, body: JSON.stringify(payload) })` sends the form
  data as JSON.
- The payload maps form fields to the `PlanRequest` schema
  (`destination`, `budget`, `duration_days`, `num_travelers`, `origin`,
  `preferences`).
- On success, it renders `data.itinerary` (newlines → `<br>`) and lists any
  `data.reflection_notes`.
- On error, it shows an error message.
- The button is disabled and shows "Planning..." while the request is in
  flight, then re-enabled.

---

## 15. Testing

The test suite uses **pytest** with `pytest.ini` configuring
`asyncio_mode = auto` (so async tests run without decorators). Tests are
organised by component.

| Test file | What it tests | Type |
|-----------|---------------|------|
| `test_app_bootstrap.py` | Settings load with defaults; app builds; `/health`; `/status` shape | Integration/smoke |
| `test_routes.py` | `/plan` and `/estimate-cost` endpoints (success, over-budget, invalid payloads) | Integration |
| `test_plan_api.py` | `/plan` and `/estimate-cost` via TestClient, mocking `graph.ainvoke` | Integration + mock |
| `test_agent.py` | LangGraph nodes (input_parser, planner, reflection) and state | Unit |
| `test_cost.py` | Cost estimator formulas (tiers, travelers, transport, budget, suggestions) | Unit |
| `test_api_clients.py` | Shared HTTP client singleton behaviour | Unit |
| `test_geocoding.py` | `geocode()` success + not-found | Unit |
| `test_places.py` | `find_places()` (geocode + Overpass) | Unit (mocked) |
| `test_search.py` | `search_destination()` (Tavily) | Unit (mocked) |
| `test_weather.py` | `get_weather()` (geocode + OWM forecast) | Unit (mocked) |

### Mocking APIs

Tests that touch external APIs mock the HTTP layer. For example:
- `test_routes.py` mocks `httpx.AsyncClient.post` with an `AsyncMock` and
  injects canned responses.
- `test_plan_api.py` patches `src.api.routes.graph.ainvoke` to return a
  canned final state.
- `test_geocoding.py` mocks `httpx.AsyncClient.get`.

### Unit vs integration

- **Unit tests** (e.g. `test_cost.py`, `test_agent.py`) test pure logic
  with no network calls.
- **Integration tests** (e.g. `test_app_bootstrap.py`, `test_plan_api.py`)
  exercise the FastAPI app and endpoints, mocking only the external
  dependencies.

---

## 16. Sequence Diagram

```
Browser / User
   │
   │  POST /plan {destination, budget, duration_days, ...}
   ▼
FastAPI (uvicorn) ──► src/api/routes.py → plan_trip()
   │                 validates PlanRequest (422 if invalid)
   │                 builds initial AgentState
   ▼
LangGraph Agent (graph.ainvoke)
   │
   ├──► input_parser
   │        └── validates user_input
   ├──► research ─────────────► Tavily (3 queries) ──► SearchResult
   ├──► places ───────────────► Nominatim ──► Overpass ──► PlacesResult
   ├──► weather ──────────────► Nominatim ──► OpenWeather ──► WeatherResult
   ├──► routing ──────────────► Nominatim(origin) ──► OpenRouteService ──► RouteResult
   ├──► cost_estimator ───────► (pure Python) ──► CostEstimateResult
   ├──► planner ──────────────► LLM (OpenRouter) ──► itinerary text
   └──► reflection ───────────► criticism / retry decisions
   │
   ▼
final_state
   │  extract itinerary, reflection_notes, errors, tool_results
   ▼
PlanResponse (JSON)
   │
   ▼
Browser renders itinerary + notes
```

---

## 17. External APIs

| API | Used For | Returns | Why Used |
|-----|----------|---------|----------|
| **Tavily** | Search | Synthesized destination overview, culture, best season, tips | Returned `answer` keeps the tool a pure data tool; free tier |
| **OpenWeatherMap** | Weather | Day-by-day forecast + rain probability | Free key, no credit card; returns `pop` directly |
| **Nominatim** (OSM) | Geocoding | (lat, lon) for a place name | Free OpenStreetMap geocoder, no key |
| **Overpass** (OSM) | Places | Nearby attractions/restaurants/museums/landmarks | Rich OSM data, single multi-category query, no key |
| **OpenRouteService** | Routing | Driving/walking distance & duration | Free tier, supports driving/foot profiles, no credit card |
| **OpenRouter** | LLM | Itinerary generation | Single endpoint for many models incl. free tiers |

> Detailed justifications and trade-offs are in `docs/api_research.md`.

### Required keys

| Key (env var) | Provider | Required? |
|---------------|----------|-----------|
| `SEARCH_API_KEY` | Tavily | Yes for full search |
| `WEATHER_API_KEY` | OpenWeatherMap | Yes for weather |
| `ROUTING_API_KEY` | OpenRouteService | Yes for routing |
| `LLM_API_KEY` | OpenRouter | Yes for itinerary |
| `PLACES_API_KEY` | — | No (OSM is keyless) |
| `CURRENCY_API_KEY` | — | Not used yet |

---

## 18. Complete End-to-End Example

Let's follow a single request through the entire system.

### Request (from the frontend / `/plan`)

```json
{
  "destination": "Murree",
  "origin": "Lahore",
  "budget": 50000,
  "duration_days": 5,
  "num_travelers": 2
}
```

### Step 0 — FastAPI validation

`PlanRequest` validates: destination present, budget `50000 > 0`,
duration `5` in `1..30`, travelers `2` in `1..20`. Valid → proceed.

### Step 1 — `input_parser`

Logs the request and returns the `user_input`. No state change besides
confirming input.

### Step 2 — `research_node` (Tavily)

Runs 3 Tavily queries for Murree. Produces something like:

```json
{
  "destination": "Murree",
  "overview": "Murree is a hill station in Punjab, Pakistan, known for pine forests and colonial-era architecture.",
  "culture": "Local culture blends Punjabi and mountain traditions...",
  "best_season": "Summer (May–August) and winter for snow (December–February).",
  "travel_tips": ["Book accommodation in advance on weekends.", "Carry warm layers even in summer."],
  "attractions_mentioned": ["Mall Road", "Pindi Point", "Patriata Chairlift"],
  "sources": ["https://..."]
}
```

State now: `tool_results["search"] = {...}`.

### Step 3 — `places_node` (Nominatim + Overpass)

Geocodes "Murree" → `(33.9067, 73.3903)`, then queries Overpass for POIs.
Produces:

```json
{
  "destination": "Murree",
  "latitude": 33.9067,
  "longitude": 73.3903,
  "places": [
    {"name": "Mall Road", "category": "attraction", "latitude": 33.9067, "longitude": 73.3903},
    {"name": "Pindi Point", "category": "viewpoint", "latitude": ..., "longitude": ...},
    {"name": "Some Restaurant", "category": "restaurant", "latitude": ..., "longitude": ...}
  ]
}
```

State now: `tool_results["places"] = {...}`.

### Step 4 — `weather_node` (Nominatim + OpenWeather)

Geocodes Murree (again) and fetches the 5-day forecast. Produces:

```json
{
  "destination": "Murree",
  "forecast": [
    {"date": "2026-01-01", "temperature_min_c": 4.0, "temperature_max_c": 12.0,
     "condition": "Clouds", "rainfall_probability": 0.3, "warnings": []},
    ...
  ],
  "notes": []
}
```

State now: `tool_results["weather"] = {...}`.

### Step 5 — `routing_node` (Nominatim + OpenRouteService)

Geocodes origin "Lahore" → `(31.5204, 74.3587)`, then queries ORS for the
Lahore→Murree route. Produces:

```json
{
  "origin": {"lat": 31.5204, "lon": 74.3587},
  "destination": {"lat": 33.9067, "lon": 73.3903},
  "driving_distance_km": 56.0,
  "driving_duration_min": 90.0,
  "walking_distance_km": null,
  "walking_duration_min": null,
  "warnings": []
}
```

State now: `tool_results["routing"] = {...}`.

### Step 6 — `cost_estimator` (pure Python)

Uses budget 50,000, 5 days, 2 travelers, driving distance 56 km.

```
nights = 4, rooms = 1
hotel   = 8000 * 4 * 1 = 32000
food    = 2500 * 5 * 2 = 25000
activ   = 2000 * 5     = 10000
transport = 35 * 56   = 1960
subtotal         = 32000 + 25000 + 10000 + 1960 = 68960
misc (10%)       = 6896
total            = 75856
budget_used      = (75856/50000)*100 = 151.7%
within_budget    = false
remaining        = -25856
```

Over budget → recommendations generated (hotel downgrade to budget saves
`(8000-3500)*4 = 18000`, food downgrade, trim activities, etc.).

State now: `tool_results["cost"] = {...}`.

### Step 7 — `planner_node` (LLM)

Builds a prompt combining search, places, weather, routing, cost, and
errors, then calls the LLM. Produces a day-by-day itinerary text, e.g.:

```
Day 1: Arrive in Murree, check in, evening walk on Mall Road.
Day 2: Morning at Pindi Point (weather is cloudy, ~12°C), afternoon...
...
```

State now: `itinerary = "Day 1: ..."`, `messages` updated.

### Step 8 — `reflection_node` (critic)

- Itinerary exists → not a failure.
- Cost is over budget → adds a note with the overage and recommendations.
- Checks missing data → none missing.
- No tool errors → OK.

State now: `reflection_notes = [ "Original cost estimate exceeds budget by 25,856 PKR. To stay within budget: Switch hotel tier...", ... ]`.

### Step 9 — Response

The router extracts the fields and returns:

```json
{
  "destination": "Murree",
  "budget": 50000,
  "duration_days": 5,
  "itinerary": "Day 1: ...",
  "reflection_notes": [
    "Original cost estimate exceeds budget by 25,856 PKR. To stay within budget: ..."
  ],
  "errors": [],
  "tool_results": { "search": {...}, "places": {...}, "weather": {...}, "routing": {...}, "cost": {...} }
}
```

### Step 10 — Frontend

The frontend renders the itinerary text and lists the reflection notes,
so the user sees the plan and why the budget is over.

---

## Running the project

```bash
# 1. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env from .env.example and fill in real keys
cp .env.example .env

# 4. Run the server
uvicorn src.main:app --reload
```

Then open:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/status
- http://127.0.0.1:8000/docs (interactive Swagger UI)
- http://127.0.0.1:8000/frontend/index.html (the web UI)

### Tests

```bash
pytest -v
pytest --cov=src --cov-report=html
```

---

*End of documentation.*
