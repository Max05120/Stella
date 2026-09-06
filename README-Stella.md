# ✦ Stella

> My personal assistant and a witty companion.

Stella is a local-first AI assistant built around **Ollama**, **ChromaDB**, and **retrieval-augmented generation (RAG)**. It answers questions grounded in your own documents, calls out to the web or a weather API when needed, remembers conversations, and includes an early framework for letting Stella actually *do* things on macOS rather than just talk about them.

This is the **backend** — a Python engine exposed through a Streamlit chat UI and a FastAPI HTTP API. It's designed to be driven directly, or from the companion menu-bar app, [Stella-app](https://github.com/Max05120/Stella-app).

## Features

- **Local RAG over your own documents** — ingest PDFs, Word docs, and text files, chunk them, embed them with Ollama, and store them in a persistent ChromaDB collection.
- **Conversational memory** — every conversation is persisted to SQLite, keyed by `conversation_id`, so history survives restarts.
- **Fast, deterministic intent routing** — a lightweight keyword router decides whether a message needs the knowledge base, the weather, the web, or just a plain chat reply, and only exposes the relevant tool to the model. This keeps normal conversation fast and avoids the model reaching for tools it doesn't need.
- **Tool calling** — `search_knowledge_base`, `get_weather`, and `web_search` are wired up as native Ollama tool calls.
- **Query rewriting** — conversational, pronoun-heavy questions ("what about the second one?") are rewritten into standalone queries before retrieval.
- **A distinct macOS knowledge pipeline** — a second, separately-curated RAG collection (`mac_knowledge`) built from scraped Apple/AppleScript/JXA documentation, used to reason about what's technically possible on a Mac.
- **An experimental capability & action system** — a registry of things Stella can *actually execute* (starting with opening applications), a planner that turns free text into a structured action, a resolver that matches intent to a registered capability, and a gap analyzer that uses the Mac knowledge base to judge whether an unimplemented request is plausible, blocked, or unknown.
- **Two front ends** — a Streamlit chat window for quick local testing, and a FastAPI service for driving Stella from another app (like the menu-bar client).
- **Personality with guardrails** — a system prompt that aims for a witty, direct "friend" tone rather than a generic corporate assistant, without inventing tool results or blind agreement.

## How it works

### Document ingestion pipeline

```
data/raw/*.pdf,.docx,.txt
        │  ingest.py
        ▼
data/processed/documents.json
        │  chunk.py
        ▼
data/processed/chunks.json
        │  embed.py  (nomic-embed-text via Ollama)
        ▼
ChromaDB collection: "stella"  (data/chroma_db)
```

### Mac knowledge pipeline (separate, curated)

```
fetch.py  (scrapes curated macOS/AppleScript/JXA docs via trafilatura)
        ▼
data/mac_raw/
        │  ingest_mac.py
        ▼
data/mac_processed/documents.json
        │  chunk_mac.py
        ▼
data/mac_processed/chunks.json
        │  embed_mac.py
        ▼
ChromaDB collection: "mac_knowledge"
```

### Conversation pipeline (`core/rag.py: ask()`)

1. **Route** the message deterministically (`Route.CHAT / KNOWLEDGE / WEATHER / WEB / CALENDAR`) based on keyword matches.
2. **First LLM pass** — the model either answers directly, or requests a tool call (only tools relevant to the detected route are offered).
3. **Tool execution** — if a tool was requested, Stella runs it (`core/tools.py`) and feeds the result back as a tool message.
4. **Second LLM pass** — the model produces a final answer grounded in the tool result.
5. **Persist** — the exchange, including sources and which tools were used, is written to `ConversationMemory` (SQLite). Per-stage timings are logged to help keep responses snappy for voice use.

### Capability / action system (`core/capabilities`, `core/actions`)

This layer is separate from the chat/RAG path above and is aimed at letting Stella take real actions on the Mac:

| Piece | Answers |
|---|---|
| `CapabilityRegistry` | "What can Stella actually execute right now?" |
| `Resolver` | "Which registered capability best matches this request?" (fuzzy match) |
| `Planner` (`core/actions`) | "What structured action is the user asking for?" |
| `Decision` | Classifies a request as `AVAILABLE`, `IMPLEMENTABLE`, or `UNKNOWN` |
| `GapAnalyzer` | For unimplemented actions, uses the `mac_knowledge` RAG collection to judge whether it's technically possible, blocked, or unclear |
| `Executor` | The only code path allowed to actually run a capability, tagged with a risk level (`low` / `medium` / `high` / `destructive`) |

At the moment only one low-risk capability ships out of the box: opening an application (`core/capabilities/workspace.py`).

## Project structure

```
Stella/
├── api.py                     # FastAPI app (chat, conversations, memory)
├── app.py                     # Streamlit chat UI
├── requirements.txt
├── core/
│   ├── engine.py               # StellaEngine: conversation/session orchestration
│   ├── rag.py                  # Router + tool-calling + LLM pipeline
│   ├── tools.py                 # Tool implementations + schemas
│   ├── memory.py                # SQLite-backed conversation memory
│   ├── config.py                # Models, prompts, personality, tuning knobs
│   ├── query_rewriter.py        # Standalone-question rewriting
│   ├── ingest.py / chunk.py / embed.py / retriever.py / search.py
│   │                             # General document RAG pipeline
│   ├── ingest_mac.py / chunk_mac.py / embed_mac.py / mac_retriever.py / fetch.py
│   │                             # Curated macOS knowledge RAG pipeline
│   ├── capabilities/            # Capability registry, resolver, executor, gap analyzer
│   └── actions/                 # Natural language → structured action planner/runner
└── test/
    └── test.py
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally, with these models pulled:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

## Setup

```bash
git clone https://github.com/Max05120/Stella.git
cd Stella
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` currently covers the ingestion/embedding stack (`pypdf`, `python-docx`, `langchain-text-splitters`, `ollama`, `chromadb`, `requests`, `trafilatura`). To run the two front ends and web search tool you'll also need:
> ```bash
> pip install streamlit fastapi "uvicorn[standard]" ddgs
> ```

## Building your knowledge base

Drop your PDFs, `.docx`, or `.txt` files into `data/raw/`, then run the pipeline in order:

```bash
python -m core.ingest
python -m core.chunk
python -m core.embed
```

Sanity-check retrieval at any point with:

```bash
python -m core.search "your question here"
```

(The macOS knowledge base under `data/mac_raw/` is built the same way, via `fetch.py → ingest_mac.py → chunk_mac.py → embed_mac.py`.)

## Running Stella

**Chat UI (Streamlit):**

```bash
streamlit run app.py
```

**HTTP API (FastAPI), for driving Stella from another app:**

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

**Command line:**

```bash
python -m core.engine
```

### API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/chat` | `{message, conversation_id}` → `{answer, search_query, sources, tools_used}` |
| `POST` | `/memory/clear?conversation_id=...` | Clears memory for a conversation |
| `GET` | `/conversations` | List conversation summaries |
| `POST` | `/conversations` | Create a new conversation |
| `GET` | `/conversations/{conversation_id}/history` | Full message history |

## Configuration

Most tuning lives in `core/config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Chat model served by Ollama |
| `TOP_K` | `3` | Chunks retrieved per knowledge-base query |
| `HISTORY_LIMIT` | `10` | Messages of prior context sent to the model |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keeps the model loaded between requests |
| `USER_CONTEXT` / `DEFAULT_LOCATION` | Hyderabad, India | Default location context — update for your own city |
| `SYSTEM_PROMPT` | — | Stella's personality and behavioral rules |

## Testing

```bash
pytest
```

Covers the action planner, action runner, and capability system (`core/test_action_planner.py`, `core/test_action_runner.py`, `core/test_capabilities.py`).

## Known limitations

- `Route.CALENDAR` is wired into the router, but no calendar tool is registered in `core/tools.py` yet — calendar questions currently fall through without a matching function.
- The capability system currently only ships one executable, low-risk capability (opening an application).
- `requirements.txt` doesn't yet list `streamlit`, `fastapi`, `uvicorn`, or `ddgs` — see Setup above.

## Related project

[**Stella-app**](https://github.com/Max05120/Stella-app) — the native macOS menu-bar and voice front end that talks to this backend over `localhost:8000`.

## License

Personal project — no license has been specified yet.
