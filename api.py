from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Literal

from core.engine import StellaEngine

app = FastAPI(title="Stella")
stella = StellaEngine()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    mode: Literal["auto", "chat", "agent"] = "auto"
    run_id: str | None = None
    confirmation_id: str | None = None
    resume_token: str | None = None


class SourceModel(BaseModel):
    source: str
    page: int | None = None
    text: str | None = None


class ChatResponse(BaseModel):
    answer: str
    search_query: str
    sources: list[SourceModel]
    tools_used: list[str]
    route: str = "chat"
    agent: dict[str, Any] | None = None
    agent_error: str | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str


@app.get("/health")
def health():
    return {"status": "ok", "assistant": "Stella"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return stella.chat(
            request.message, request.conversation_id, mode=request.mode,
            run_id=request.run_id, confirmation_id=request.confirmation_id,
            resume_token=request.resume_token,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear")
def clear_memory(conversation_id: str):
    stella.clear_memory(conversation_id)
    return {"status": "cleared"}


@app.get("/conversations", response_model=list[ConversationSummary])
def list_conversations():
    return stella.list_conversations()


@app.post("/conversations", response_model=ConversationSummary)
def create_conversation():
    conversation_id = stella.create_conversation()
    return next(c for c in stella.list_conversations() if c["id"] == conversation_id)


@app.get("/conversations/{conversation_id}/history")
def get_history(conversation_id: str):
    return stella.get_history(conversation_id)