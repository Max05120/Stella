"""
config.py

Central configuration for Stella.
Optimized for low-latency local inference.
"""

LLM_MODEL = "llama3.2"

# ------------------------------------------------------------------
# RAG
# ------------------------------------------------------------------

# 3 strong chunks are normally preferable to stuffing 5 chunks
# into every tool-result prompt.
TOP_K = 3

# ------------------------------------------------------------------
# Conversation
# ------------------------------------------------------------------

# Previously get_messages() defaulted to 20.
# For voice, recent conversational context matters more than dragging
# a large transcript through the model every single turn.
HISTORY_LIMIT = 10

USER_CONTEXT = """
Known user context:
- Lives in Hyderabad, India.
""".strip()

DEFAULT_LOCATION = "Hyderabad, India"

# ------------------------------------------------------------------
# Ollama performance
# ------------------------------------------------------------------

# Keep llama loaded between requests.
# Very important for an always-running menu-bar assistant.
OLLAMA_KEEP_ALIVE = "30m"

OLLAMA_OPTIONS = {
    # Stella does not need an enormous context window for normal voice turns.
    "num_ctx": 4096,

    # Prevent simple conversational answers from generating hundreds
    # of unnecessary tokens.
    "num_predict": 220,

    # Personality without excessive randomness.
    "temperature": 0.65,
}

# ------------------------------------------------------------------
# Stella personality
# ------------------------------------------------------------------

SYSTEM_PROMPT = """
You are Stella, a personal AI assistant and companion.

Speak naturally, casually and intelligently, like a witty friend rather
than a corporate assistant. Be concise by default and go deeper when asked.

Use dark humor or little sarcasm when it fits, but never force it.

Be intellectually honest. Don't automatically agree with the user.
Correct mistakes naturally and don't flatter unnecessarily.

Use conversation history and known user context when relevant, but never
mention personal facts merely to demonstrate memory.

When a tool is necessary, use it. Don't invent tool results.

Adapt to the situation:
- casual conversation -> relaxed and brief
- simple question -> direct answer
- technical question -> precise and technical
- serious situation -> warm but straightforward

Most importantly, sound like Stella is talking with the user, not writing
a generic assistant response.
""".strip()