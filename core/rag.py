"""
rag.py

Primary reasoning + tool execution pipeline for Stella.

Optimized for low-latency conversational use.
"""

import json
import time

import ollama
import re
from core.config import (
    HISTORY_LIMIT,
    HISTORY_CHAR_BUDGET,
    SOURCE_CHAR_BUDGET,
    PREFERRED_NAME,
    TOP_K,
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
        and callable(
            AVAILABLE_FUNCTIONS.get(
                _tool_name(schema)
            )
        )
    ]

def _valid_sources(value) -> list[dict]:
    """
    Keep sources compatible with the API's SourceModel.
    """
    if not isinstance(value, list):
        return []

    sources = []

    for item in value:
        if not isinstance(item, dict):
            continue

        if not isinstance(item.get("source"), str):
            continue

        page = item.get("page")
        text = item.get("text")

        if page is not None:
            if (
                not isinstance(page, int)
                or isinstance(page, bool)
            ):
                continue

        if text is not None and not isinstance(text, str):
            continue

        sources.append(
            {
                "source": item["source"],
                "page": page,
                "text": text,
            }
        )

    return sources

def _is_followup(question: str) -> bool:
    q = question.lower().strip()

    if q.startswith(
        (
            "new topic",
            "unrelated",
            "changing topics",
        )
    ):
        return False

    reference = re.search(
        r"\b(it|its|they|them|their|that|those|these|"
        r"the first one|the second one)\b",
        q,
    )

    continuation = q.startswith(
        (
            "what about ",
            "and tomorrow",
            "and in ",
            "tell me more",
            "go on",
        )
    )

    return bool(reference or continuation)


def _reuse_evidence(question: str) -> bool:
    """
    Recognize transformations of the previous answer,
    including common spoken lead-ins.
    """
    q = question.lower().strip()

    # Example:
    # "Hm. Alright, summarize it."
    # becomes:
    # "summarize it."
    q = re.sub(
        r"^(?:(?:h+m+|um+|uh+|okay|ok|alright|all right|well)"
        r"[\s,.;:!?]+)+",
        "",
        q,
    )

    return bool(
        re.fullmatch(
            r"(?:please )?"
            r"(?:summari[sz]e|simplify|shorten|rephrase|explain) "
            r"(?:it|that|this|your (?:last |previous )?answer)"
            r"(?: please)?[.!?]*",
            q,
        )
    )

def _context_route(
    question: str,
    history: list[dict],
):
    previous = Route.CHAT
    anchor = ""
    last_user = ""

    tool_routes = {
        "search_knowledge_base": Route.KNOWLEDGE,
        "get_weather": Route.WEATHER,
        "web_search": Route.WEB,
    }

    for item in history:
        if item["role"] == "assistant":
            used = item.get("tools_used") or []

            observed_route = next(
                (
                    tool_routes[name]
                    for name in used
                    if isinstance(name, str)
                    and name in tool_routes
                ),
                None,
            )

            if _valid_sources(item.get("sources")):
                observed_route = Route.KNOWLEDGE

            if observed_route is not None:
                previous = observed_route
                anchor = anchor or last_user

        if item["role"] == "user":
            last_user = item["content"]

            explicit = route_question(
                item["content"]
            )

            if explicit != Route.CHAT:
                previous = explicit
                anchor = item["content"]

            elif not (
                _is_followup(item["content"])
                or _reuse_evidence(item["content"])
            ):
                previous = Route.CHAT
                anchor = ""

    explicit = route_question(question)

    followup = (
        _is_followup(question)
        or _reuse_evidence(question)
    )

    route = (
        previous
        if explicit == Route.CHAT and followup
        else explicit
    )

    last = (
        history[-1]
        if history
        and history[-1]["role"] == "assistant"
        else {}
    )

    evidence = (
        _valid_sources(last.get("sources"))
        if followup and route == Route.KNOWLEDGE
        else []
    )

    return (
        route,
        anchor if followup else "",
        evidence,
    )

def _bounded_history(
    history: list[dict],
) -> list[dict]:
    remaining = HISTORY_CHAR_BUDGET
    kept = []

    for item in reversed(history):
        if remaining <= 0:
            break

        content = item["content"][:remaining]

        kept.append(
            {
                "role": item["role"],
                "content": content,
            }
        )

        remaining -= len(content)

    return list(reversed(kept))


def _evidence_message(
    sources: list[dict],
) -> dict:
    remaining = SOURCE_CHAR_BUDGET
    passages = []

    for source in sources[:TOP_K]:
        if remaining <= 0:
            break

        text = (
            source.get("text") or ""
        )[:remaining]

        remaining -= len(text)

        passages.append(
            {
                **source,
                "text": text,
            }
        )

    return {
        "role": "tool",
        "tool_name": "search_knowledge_base",
        "content": _compact_json(passages),
    }

def _model_metrics(
    response,
    prefix: str,
) -> dict:
    metrics = {}

    duration_fields = (
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    )

    for key in duration_fields:
        value = response.get(key)

        if isinstance(value, (float, int)):
            metrics[
                f"{prefix}_{key}_ms"
            ] = round(
                value / 1_000_000,
                1,
            )

    count_fields = (
        "prompt_eval_count",
        "eval_count",
    )

    for key in count_fields:
        value = response.get(key)

        if isinstance(value, int):
            metrics[
                f"{prefix}_{key}"
            ] = value

    return metrics

def ask(
    question: str,
    memory: ConversationMemory,
    top_k: int | None = None,
    *,
    action_context: str | None = None,
):
    """
    Return:
        answer
        sources
        search_query
        tools_used
        timings
    """
    started = time.perf_counter()

    # ---------------------------------------------------------
    # 1. Read conversation and select a route
    # ---------------------------------------------------------

    mark = time.perf_counter()

    history = memory.get_messages(
        limit=HISTORY_LIMIT,
        include_metadata=True,
    )

    timings = {
        "memory_read_ms": (
            time.perf_counter() - mark
        ) * 1000,
        "llm_1_ms": 0.0,
        "tool_ms": 0.0,
        "llm_2_ms": 0.0,
    }

    mark = time.perf_counter()

    route, anchor, evidence = _context_route(
        question,
        history,
    )

    selected_tools = _tools_for_route(route)

    allowed = {
        _tool_name(tool)
        for tool in selected_tools
    }

    timings["route_ms"] = (
        time.perf_counter() - mark
    ) * 1000

    print(
        f"[STELLA ROUTER] "
        f"route={route.value} | "
        f"tools={sorted(allowed)}"
    )

    # ---------------------------------------------------------
    # 2. Build the prompt
    # ---------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        }
    ]

    if action_context:
        messages.append(
            {
                "role": "system",
                "content": action_context,
            }
        )

    if _reuse_evidence(question):
        messages.append(
            {
                "role": "system",
                "content": (
                    "The user wants a concise transformation of "
                    "your previous answer. Respond in at most two "
                    "short sentences, aiming for 45 words or fewer. "
                    "Preserve the key point and essential caveats. "
                    "Do not introduce new topics or repeat a long list."
                ),
            }
        )
        
    messages.extend(
        _bounded_history(history)
    )

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    results = []
    tools_used = []
    search_query = question
    answer = None
    model_calls = 0

    # ---------------------------------------------------------
    # 3. Shared generation helper with timing
    # ---------------------------------------------------------

    def generate(
        *,
        tools=None,
        stage="llm_1",
        num_predict=160,
    ):
        nonlocal model_calls

        mark = time.perf_counter()

        response = _chat(
            messages,
            tools=tools,
            num_predict=num_predict,
        )

        timings[stage + "_ms"] += (
            time.perf_counter() - mark
        ) * 1000

        timings.update(
            _model_metrics(
                response,
                stage,
            )
        )

        model_calls += 1

        return response["message"]

    # ---------------------------------------------------------
    # 4. Handle simple identity/unavailable-tool requests
    # ---------------------------------------------------------

    normalized = (
        question.lower()
        .strip()
        .rstrip(".!?")
        .replace("’", "'")
    )

    name_questions = {
        "what's my name",
        "what is my name",
        "do you remember my name",
    }

    if normalized in name_questions:
        answer = (
            f"Your name is {PREFERRED_NAME}."
        )

    elif (
        route != Route.CHAT
        and not selected_tools
    ):
        if route == Route.CALENDAR:
            answer = (
                "I don't currently have access to your calendar, "
                "so I can't check your events or change them."
            )
        else:
            answer = (
                f"The {route.value} tool "
                "is currently unavailable."
            )

    # ---------------------------------------------------------
    # 5. Document requests: retrieve, then generate once
    # ---------------------------------------------------------

    elif route == Route.KNOWLEDGE:
        if evidence and _reuse_evidence(question):
            results = evidence

        else:
            search_query = (
                f"Previous document question: {anchor[:1500]}\n"
                f"Follow-up: {question}"
                if anchor
                else question
            )

            mark = time.perf_counter()

            tools_used.append(
                "search_knowledge_base"
            )

            try:
                raw = AVAILABLE_FUNCTIONS[
                    "search_knowledge_base"
                ](
                    query=search_query,
                    top_k=(
                        top_k
                        if top_k is not None
                        else TOP_K
                    ),
                )

                results = _valid_sources(raw)

                if (
                    not isinstance(raw, list)
                    or (raw and not results)
                ):
                    answer = (
                        "I couldn't retrieve usable document "
                        "passages. Please try again."
                    )

                elif not results:
                    answer = (
                        "I couldn't find relevant passages "
                        "in your documents."
                    )

            except Exception:
                answer = (
                    "I couldn't search your documents "
                    "right now. Please try again."
                )

            timings["tool_ms"] = (
                time.perf_counter() - mark
            ) * 1000

        if answer is None:
            # Include the tool call and its evidence together.
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": {
                                    "query": search_query,
                                },
                            }
                        }
                    ],
                }
            )

            messages.append(
                _evidence_message(results)
            )

            message = generate()

            answer = message.get(
                "content",
                "",
            ).strip()

    # ---------------------------------------------------------
    # 6. Ordinary chat and model-selected weather/web tools
    # ---------------------------------------------------------

    else:
        message = generate(
            tools=selected_tools or None,
            num_predict=(
                96
                if selected_tools
                else 160
            ),
        )

        calls = (
            message.get("tool_calls")
            or []
        )

        if calls:
            messages.append(message)

            mark = time.perf_counter()

            for call in calls:
                function = call.get(
                    "function",
                    {},
                )

                name = function.get("name")
                args = function.get("arguments")

                fn = (
                    AVAILABLE_FUNCTIONS.get(name)
                    if isinstance(name, str)
                    else None
                )

                # Schema selection is not enough:
                # enforce permission again before execution.
                if (
                    not isinstance(name, str)
                    or name not in allowed
                    or not callable(fn)
                ):
                    result = {
                        "error": (
                            "Tool not allowed for this route: "
                            f"{name}"
                        )
                    }

                elif not isinstance(args, dict):
                    result = {
                        "error": (
                            "Tool arguments must be an object."
                        )
                    }

                else:
                    tools_used.append(name)

                    try:
                        result = fn(**args)

                    except Exception as exc:
                        result = {
                            "error": (
                                f"{name} failed: {exc}"
                            )
                        }

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": (
                            name
                            if isinstance(name, str)
                            else "unknown"
                        ),
                        "content": _compact_json(result),
                    }
                )

            timings["tool_ms"] = (
                time.perf_counter() - mark
            ) * 1000

            message = generate(
                stage="llm_2"
            )

        answer = message.get(
            "content",
            "",
        ).strip()

    # ---------------------------------------------------------
    # 7. Save the turn and return the existing response contract
    # ---------------------------------------------------------

    answer = (
        answer
        or "I couldn't generate a response."
    )

    tools_used = list(
        dict.fromkeys(tools_used)
    )

    mark = time.perf_counter()

    memory.add_turn(
        question,
        answer,
        sources=results,
        tools_used=tools_used,
    )

    timings["memory_write_ms"] = (
        time.perf_counter() - mark
    ) * 1000

    timings["total_ms"] = (
        time.perf_counter() - started
    ) * 1000

    timings = {
        key: round(value, 1)
        for key, value in timings.items()
    }

    timings["model_calls"] = model_calls

    print(
        "[STELLA LATENCY] "
        + _compact_json(timings)
    )

    return (
        answer,
        results,
        search_query,
        tools_used,
        timings,
    )