"""
rag.py

Core Retrieval-Augmented Generation pipeline.

Flow:

User question
    ↓
Embed question
    ↓
Retrieve relevant chunks
    ↓
Build context
    ↓
Send context + question to Ollama
    ↓
Return grounded answer
"""

import ollama
import json
from .tools import TOOL_SCHEMAS, AVAILABLE_FUNCTIONS
from .config import (
    LLM_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
)

from .retriever import retrieve
from .memory import ConversationMemory
from .query_rewriter import rewrite_query


def build_context(results: list[dict]) -> str:
    """
    Convert retrieved chunks into a context block
    that can be passed to the LLM.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        source = result["source"]
        page = result["page"]
        text = result["text"]

        context_parts.append(
            f"""
                [Source {i}]
                File: {source}
                Page: {page}
                {text}
                """
        )

    return "\n".join(context_parts)


def generate_answer(question: str, results: list[dict]) -> str:
    context = build_context(results)
    user_prompt = f"""..."""  # unchanged — keep exactly what you have

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = ollama.chat(model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMAS)
    message = response["message"]

    if message.get("tool_calls"):
        messages.append(message)
        for call in message["tool_calls"]:
            fn = AVAILABLE_FUNCTIONS[call["function"]["name"]]
            result = fn(**call["function"]["arguments"])
            messages.append({"role": "tool", "content": json.dumps(result)})
        response = ollama.chat(model=LLM_MODEL, messages=messages)
        message = response["message"]

    return message["content"]


def ask(question: str, memory: ConversationMemory, top_k: int = None):
    history = memory.get_messages()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": question},
    ]

    response = ollama.chat(model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMAS)
    message = response["message"]

    results = []
    search_query = question

    if message.get("tool_calls"):
        messages.append(message)
        for call in message["tool_calls"]:
            fn_name = call["function"]["name"]
            fn_args = call["function"]["arguments"]
            result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)

            if fn_name == "search_knowledge_base":
                results = result
                search_query = fn_args.get("query", question)

            messages.append({"role": "tool", "content": json.dumps(result)})

        response = ollama.chat(model=LLM_MODEL, messages=messages)
        message = response["message"]

    answer = message["content"]
    memory.add_user_message(question)
    memory.add_assistant_message(answer)

    return answer, results, search_query


if __name__ == "__main__":

    memory = ConversationMemory()

    print("STELLA")
    print("======")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        answer, results, search_query = ask(
            question,
            memory,
        )

        print("\nStella:")
        print(answer)

        print("\n[Search query]")
        print(search_query)

        print("\n[Sources]")

        for result in results:
            print(
                f"- {result['source']} "
                f"(page {result['page']}) "
                f"[distance={result['distance']:.4f}]"
            )

        print("\n" + "-" * 60 + "\n")