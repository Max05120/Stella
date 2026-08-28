"""
query_rewriter.py

Converts conversational questions into standalone
questions suitable for semantic retrieval.
"""

import ollama

from .config import LLM_MODEL


QUERY_REWRITE_PROMPT = """
You rewrite conversational questions into standalone
questions for a semantic search system.

The user may refer to things using words like:
- it
- they
- them
- that
- this
- the first one
- the second one
- the previous answer

Use the conversation history to resolve those references.

Rules:

1. Preserve the user's actual intent.
2. Do not answer the question.
3. Do not add information that isn't supported by the conversation.
4. Make the rewritten question self-contained.
5. If the question is already self-contained, return it unchanged.
6. Return ONLY the rewritten question.
"""


def rewrite_query(
    question: str,
    conversation_history: list[dict],
) -> str:

    if not conversation_history:
        return question

    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in conversation_history
    )

    prompt = f"""
{QUERY_REWRITE_PROMPT}

CONVERSATION HISTORY
====================

{history_text}

====================

CURRENT QUESTION
=================

{question}

=================

STANDALONE QUESTION:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    rewritten = response["message"]["content"].strip()

    return rewritten

# if __name__ == "__main__":

#     history = [
#         {
#             "role": "user",
#             "content": "What are the three models?"
#         },
#         {
#             "role": "assistant",
#             "content": "The document discusses Llama, Mistral, and Gemma."
#         },
#     ]

#     question = "Which one is the smallest?"

#     rewritten = rewrite_query(
#         question,
#         history,
#     )

#     print("\nOriginal:")
#     print(question)

#     print("\nRewritten:")
#     print(rewritten)