import requests
from ddgs import DDGS  # pip install ddgs
from .retriever import retrieve as _retrieve
from .config import TOP_K

def search_knowledge_base(query: str, top_k: int = TOP_K) -> list[dict]:
    """Search the user's local documents for relevant information."""
    results = _retrieve(query, top_k=top_k)
    return [{"source": r["source"], "page": r["page"], "text": r["text"]} for r in results]

def get_weather(latitude: float, longitude: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["current"]

def web_search(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search the user's own documents for information relevant to their question. Use this for anything that could be answered from their uploaded files, unless the answer is already clear from earlier in this conversation.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                "description": "A standalone search query -- resolve any pronouns or references from the conversation into an explicit query"}},
            "required": ["query"]},
    }},    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a location by latitude/longitude",
        "parameters": {"type": "object", "properties": {
            "latitude": {"type": "number"}, "longitude": {"type": "number"}},
            "required": ["latitude", "longitude"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live web for current info not in the local documents",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
            "required": ["query"]},
    }},
]


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "web_search": web_search,
    "search_knowledge_base": search_knowledge_base,
}