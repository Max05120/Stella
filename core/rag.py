"""
rag.py

Primary reasoning + tool execution pipeline for Stella.

Optimized for low-latency conversational use.
"""

import json
import time

import ollama

from core.config import (
    HISTORY_LIMIT,
    LLM_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_OPTIONS,
    SYSTEM_PROMPT,
    USER_CONTEXT,
)
from core.memory import ConversationMemory
from core.tools import AVAILABLE_FUNCTIONS, TOOL_SCHEMAS
from enum import Enum

SYSTEM_MESSAGE = (
    SYSTEM_PROMPT
    + "\n\n"
    + USER_CONTEXT
)

class Route(str, Enum):
    CHAT = "chat"
    KNOWLEDGE = "knowledge"
    WEATHER = "weather"
    WEB = "web"
    CALENDAR = "calendar"


def route_question(question: str) -> Route:
    """
    Fast deterministic router.

    This deliberately handles only obvious intents.
    Ambiguous/general requests fall back to normal CHAT instead
    of unnecessarily exposing tools to the LLM.
    """

    q = question.lower().strip()

    # --------------------------------------------------------------
    # Explicit knowledge-base requests
    # --------------------------------------------------------------

    knowledge_phrases = (
        "knowledge base",
        "my documents",
        "my document",
        "my files say",
        "search my files",
        "search my docs",
        "according to my documents",
    )

    if any(phrase in q for phrase in knowledge_phrases):
        return Route.KNOWLEDGE

    # --------------------------------------------------------------
    # Weather
    # --------------------------------------------------------------

    weather_words = (
        "weather",
        "temperature",
        "forecast",
        "rain today",
        "will it rain",
    )

    if any(word in q for word in weather_words):
        return Route.WEATHER

    # --------------------------------------------------------------
    # Calendar
    # --------------------------------------------------------------

    calendar_words = (
        "calendar",
        "my schedule",
        "my meetings",
        "meeting today",
        "meetings today",
        "meeting tomorrow",
        "meetings tomorrow",
        "appointment",
    )

    if any(word in q for word in calendar_words):
        return Route.CALENDAR

    # --------------------------------------------------------------
    # Explicit web/current-information request
    # --------------------------------------------------------------

    web_phrases = (
        "search the web",
        "search online",
        "look online",
        "look it up online",
        "google",
        "latest news",
        "recent news",
        "news today",
        "current news",
    )

    if any(phrase in q for phrase in web_phrases):
        return Route.WEB

    # --------------------------------------------------------------
    # Default
    #
    # Normal conversation, explanations, jokes, follow-ups,
    # reasoning and memory should NOT receive tools.
    # --------------------------------------------------------------

    return Route.CHAT


def _chat(
    messages: list[dict],
    *,
    tools=None,
    num_predict: int | None = None,
):

    options = dict(OLLAMA_OPTIONS)

    if num_predict is not None:
        options["num_predict"] = num_predict

    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": options,
    }

    if tools:
        kwargs["tools"] = tools

    return ollama.chat(**kwargs)


def _compact_json(value) -> str:
    """
    Compact tool results before feeding them back to the LLM.

    Whitespace inside JSON contributes nothing useful but still
    increases prompt size.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )
def _tool_name(schema: dict) -> str | None:
    """
    Supports Ollama-style tool schemas.
    """

    if not isinstance(schema, dict):
        return None

    function = schema.get("function", {})

    if isinstance(function, dict):
        return function.get("name")

    return None


def _tools_for_route(route: Route) -> list[dict]:
    """
    Return only the tools relevant to this request.
    """

    allowed = {
        Route.KNOWLEDGE: {"search_knowledge_base"},
        Route.WEATHER: {"get_weather"},
        Route.WEB: {"web_search"},

        # Change this name if your calendar function uses
        # a different function name.
        Route.CALENDAR: {
            "get_calendar_events",
            "calendar_events",
            "get_calendar",
        },
    }.get(route, set())

    if not allowed:
        return []

    return [
        schema
        for schema in TOOL_SCHEMAS
        if _tool_name(schema) in allowed
    ]

def ask(
    question: str,
    memory: ConversationMemory,
    top_k: int | None = None,
):
    """
    Main Stella conversation path.

    Returns:
        answer
        retrieved results
        search query
        tools used
        timing information
    """

    started = time.perf_counter()

    # --------------------------------------------------------------
    # Memory
    # --------------------------------------------------------------

    memory_started = time.perf_counter()

    route_started = time.perf_counter()

    route = route_question(question)
    selected_tools = _tools_for_route(route)

    route_ms = (
        time.perf_counter() - route_started
    ) * 1000

    print(
        f"[STELLA ROUTER] "
        f"route={route.value} | "
        f"tools={[ _tool_name(t) for t in selected_tools ]}"
    )

    history = memory.get_messages(
        limit=HISTORY_LIMIT
    )

    memory_read_ms = (
        time.perf_counter() - memory_started
    ) * 1000

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        *history,
        {
            "role": "user",
            "content": question,
        },
    ]

    # --------------------------------------------------------------
    # First LLM pass
    #
    # Stella either:
    #   1. answers directly
    #   2. requests one or more tools
    # --------------------------------------------------------------

    llm1_started = time.perf_counter()

    if selected_tools:

        response = _chat(
        messages,
        tools=selected_tools,
        num_predict=96,
    )

    else:

        response = _chat(
        messages,
        num_predict=160,
    )

    llm1_ms = (
        time.perf_counter() - llm1_started
    ) * 1000

    message = response["message"]

    results = []
    search_query = question
    tools_used = []

    tool_ms = 0.0
    llm2_ms = 0.0

    # --------------------------------------------------------------
    # Tool execution
    # --------------------------------------------------------------

    tool_calls = message.get("tool_calls") or []

    if tool_calls:
        print(
        "[STELLA TOOLS]",
        [
            {
                "name": call["function"]["name"],
                "args": call["function"]["arguments"],
            }
            for call in tool_calls
        ],
    )
        messages.append(message)

        tool_started = time.perf_counter()
        
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"]["arguments"]

            tools_used.append(fn_name)

            fn = AVAILABLE_FUNCTIONS.get(fn_name)

            if fn is None:
                result = {
                    "error": f"Unknown tool: {fn_name}"
                }

            else:
                try:
                    result = fn(**fn_args)
                except Exception as exc:
                    result = {
                        "error": f"{fn_name} failed: {exc}"
                    }

            if fn_name == "search_knowledge_base":
                results = result

                if isinstance(fn_args, dict):
                    search_query = fn_args.get(
                        "query",
                        question,
                    )

            messages.append(
                {
                    "role": "tool",
                    "content": _compact_json(result),
                }
            )

        tool_ms = (
            time.perf_counter() - tool_started
        ) * 1000

        # ----------------------------------------------------------
        # Second LLM pass only when tools were actually used
        # ----------------------------------------------------------

        llm2_started = time.perf_counter()

        response = _chat(
            messages,
            num_predict=160,
        )

        llm2_ms = (
            time.perf_counter() - llm2_started
        ) * 1000

        message = response["message"]

    # --------------------------------------------------------------
    # Answer
    # --------------------------------------------------------------

    answer = message.get("content", "").strip()

    if not answer:
        answer = "I couldn't generate a response."

    # --------------------------------------------------------------
    # Persist conversation
    # --------------------------------------------------------------

    memory_write_started = time.perf_counter()

    memory.add_user_message(question)

    memory.add_assistant_message(
        answer,
        sources=results,
        tools_used=tools_used,
    )

    memory_write_ms = (
        time.perf_counter() - memory_write_started
    ) * 1000

    total_ms = (
        time.perf_counter() - started
    ) * 1000

    timings = {
        "memory_read_ms": round(memory_read_ms, 1),
        "llm_1_ms": round(llm1_ms, 1),
        "tool_ms": round(tool_ms, 1),
        "llm_2_ms": round(llm2_ms, 1),
        "memory_write_ms": round(memory_write_ms, 1),
        "total_ms": round(total_ms, 1),
    }

    print(
        "[STELLA LATENCY] "
        f"memory={timings['memory_read_ms']}ms | "
        f"llm1={timings['llm_1_ms']}ms | "
        f"tools={timings['tool_ms']}ms | "
        f"llm2={timings['llm_2_ms']}ms | "
        f"write={timings['memory_write_ms']}ms | "
        f"TOTAL={timings['total_ms']}ms"
    )

    return (
        answer,
        results,
        search_query,
        tools_used,
        timings,
    )