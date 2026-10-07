"""Conservative pre-execution routing. No model calls or side effects."""
from dataclasses import dataclass
import re
from core.actions.planner import plan_action


@dataclass(frozen=True)
class RouteDecision:
    route: str
    message: str
    reason: str


def _device_snapshot_question(text):
    """Recognize a bounded set of read-only questions, never quoted commands.

    Full matching is deliberate: tutorials, conditionals and extra instructions
    must not become device actions just because they mention volume or an app.
    """
    question = text.lower().strip().rstrip(".?!").strip()
    question = re.sub(r"^(?:(?:please|can you|could you|would you)\s+)+", "", question)
    question = re.sub(r"^(?:tell me|show me|check)\s+", "", question)
    question = re.sub(r"\s+(?:right now|currently|now)$", "", question)
    return any(re.fullmatch(pattern, question) for pattern in (
        r"how loud (?:my|the) (?:mac|computer)(?: is)?",
        r"how loud is (?:my|the) (?:mac|computer)",
        r"(?:what is|what's) (?:my |the )?(?:current )?(?:(?:system|mac|computer)(?:'s)? )?volume(?: level)?",
        r"(?:the |my )?(?:current )?(?:system |mac )?volume(?: level)?",
        r"(?:what|which) (?:app|application) is (?:frontmost|in front|active)",
        r"(?:what is|what's) the (?:frontmost|active) (?:app|application)",
    ))


def route_request(message, mode="auto"):
    text = message.strip()
    if mode not in {"auto", "chat", "agent"}:
        raise ValueError("mode must be auto, chat, or agent")
    for prefix, forced in (("/agent ", "agent"), ("/chat ", "chat")):
        if text.lower().startswith(prefix):
            mode, text = forced, text[len(prefix):].strip()
            break
    if not text:
        return RouteDecision("chat", text, "empty request")
    if mode != "auto":
        return RouteDecision(mode, text, "explicit route")
    if _device_snapshot_question(text):
        return RouteDecision("agent", text, "current device inspection question")
    # Questions/explanations must never execute their quoted/example commands.
    normalized = re.sub(r"^(?:please\s+|(?:can|could|would) you\s+)", "", text, flags=re.I)
    if re.match(r"^(?:what|why|how|when|where|who|explain|describe|tell me|teach|do you|is |are )\b", normalized, re.I):
        return RouteDecision("chat", text, "conversation or information request")
    action = re.match(r"^(?:get|inspect|focus|toggle|increase|decrease|activate|switch|scroll|open|launch|close|quit|create|make|move|copy|rename|delete|remove|trash|list|find|show|reveal|set|turn|mute|unmute|organize|organise|arrange|resize|minimize|maximize|click|press|type|paste|search)\b", normalized, re.I)
    # Remove quoted targets before looking for sequencing words.
    structure = re.sub(r'''"[^"]*"|'[^']*' ''', "", normalized, flags=re.X)
    compound = re.search(r"\b(?:and|then|after|before|followed by)\b|[;\n]", structure, re.I)
    if action and compound:
        return RouteDecision("agent", text, "multi-step action goal")
    try:
        planning = plan_action(normalized.rstrip(".,!?;:"))
    except Exception:
        return RouteDecision("chat", text, "action parser unavailable; nothing executed")
    if planning.understood and planning.request is not None:
        return RouteDecision("direct", normalized, "deterministic single action")
    if action and not re.match(r"^(?:search|find|show|list)\b.*\b(?:docs|documents|weather|news|web)\b", normalized, re.I):
        return RouteDecision("agent", text, "action goal needs planning")
    return RouteDecision("chat", text, "conversation/RAG fallback")