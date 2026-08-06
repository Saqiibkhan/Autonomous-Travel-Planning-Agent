
# Autonomous Travel Planning Agent

> Status: **Complete** -- FastAPI backend with LangGraph workflow,
> reflection node, `/plan` and `/estimate-cost` endpoints, tests, and a
> simple web frontend.

## What this is

An agent that plans a full multi-day trip -- given a destination, budget,
and duration -- by researching attractions, checking weather, calculating
routes, estimating costs, and generating a day-by-day itinerary with a
stated reason behind every recommendation. Built for the AI Summer
Internship 2026 travel-agent task.

## Project structure

```
travel-agent/
├── src/
│   ├── main.py            # FastAPI app entrypoint
│   ├── config.py          # Pydantic Settings (env-var driven config)
│   ├── agent/
│   │   ├── state.py       # LangGraph state schema
│   │   ├── nodes.py       # Workflow nodes (research, places, weather, routing, cost, planner)
│   │   ├── reflection.py  # Critic node (budget checks, data quality, retry decisions)
│   │   └── graph.py       # Compiled LangGraph workflow
│   ├── tools/
│   │   ├── search_tool.py   # destination overview/culture/season/tips (Tavily)
│   │   ├── places_tool.py   # attractions/restaurants/museums/landmarks (OSM)
│   │   ├── weather_tool.py  # day-by-day forecast (OpenWeatherMap)
│   │   ├── routing_tool.py  # driving/walking distance+duration (OpenRouteService)
│   │   └── cost_estimator.py  # hotel/transport/food/activities cost + budget check (no API)
│   ├── services/
│   │   ├── api_clients.py   # shared async httpx client
│   │   ├── geocoding.py     # shared destination -> (lat, lon) lookup (Nominatim)
│   │   └── retry.py         # generic retry decorator w/ logging + backoff
│   ├── schemas/
│   │   ├── cost.py           # CostBreakdown / CostEstimateResult
│   │   └── plan.py           # PlanRequest / PlanResponse / CostEstimateRequest / CostEstimateResponse
│   ├── utils/
│   │   ├── logger.py        # structured logging setup
│   │   ├── constants.py
│   │   └── exceptions.py
│   └── api/
│       └── routes.py        # /health, /status, /plan, /estimate-cost
├── tests/
│   ├── test_app_bootstrap.py
│   ├── test_cost.py
│   ├── test_places.py
│   ├── test_routes.py
│   ├── test_search.py
│   ├── test_weather.py
│   ├── test_plan_api.py
│   └── test_agent.py
├── docs/
│   └── api_research.md
├── frontend/
│   └── index.html
├── .env.example
├── requirements.txt
├── .gitignore
├── pytest.ini
└── README.md
```

## Setup

```bash
cd travel-agent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in real keys
```

## Running locally

```bash
uvicorn src.main:app --reload
```

Then check:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/status
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/frontend/index.html

## Testing

```bash
pytest -v
pytest --cov=src --cov-report=html
```

## Environment variables

See `.env.example` for the full list.
