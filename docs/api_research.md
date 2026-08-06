
# API Research

This document explains why each external API was chosen, what its trade-offs are,
and how to obtain credentials.

## Search: Tavily

**Why:** Tavily returns a synthesized `answer` field built from multiple sources,
so the Search Tool can stay a pure data tool with no extra LLM call.  
**Alternatives considered:** Serper.dev, DuckDuckGo.  
**Trade-off:** Requires a free API key (1,000 searches/month).  
**Key:** https://tavily.com

## Places: OpenStreetMap (Nominatim + Overpass)

**Why:** Zero API key, zero billing setup. Overpass lets us query multiple
tourism categories in one request.  
**Alternatives considered:** Google Places (requires billing).  
**Trade-off:** No star ratings or photos; data completeness varies by region.  
**Key:** None required.

## Weather: OpenWeatherMap 5 Day / 3 Hour Forecast

**Why:** Free tier needs only a key, no credit card. Returns `pop` (rain
probability) directly.  
**Alternatives considered:** OWM One Call 3.0 (now requires payment method).  
**Trade-off:** Only forecasts 5 days out.  
**Key:** https://openweathermap.org/api

## Routing: OpenRouteService Directions

**Why:** Free tier (2,000 requests/day), no credit card. Supports driving-car
and foot-walking profiles.  
**Alternatives considered:** Google Directions (requires billing).  
**Trade-off:** No live traffic data.  
**Key:** https://openrouteservice.org/dev/#/signup

## LLM: OpenRouter (OpenAI-compatible)

**Why:** Single endpoint for many models, including free-tier options.  
**Alternatives considered:** Direct OpenAI API (paid).  
**Trade-off:** Free models can be slower or rate-limited.  
**Key:** https://openrouter.ai
