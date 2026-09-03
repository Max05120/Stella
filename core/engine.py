"""
engine.py

Main application interface for Stella.
"""

from collections import OrderedDict

from core.memory import ConversationMemory
from core.rag import ask


class StellaEngine:

    MAX_MEMORY_CACHE = 32

    def __init__(self):
        self._memories: OrderedDict[
            str,
            ConversationMemory
        ] = OrderedDict()

    def _get_memory(
        self,
        conversation_id: str,
    ) -> ConversationMemory:

        memory = self._memories.get(conversation_id)

        if memory is not None:
            self._memories.move_to_end(
                conversation_id
            )
            return memory

        memory = ConversationMemory(
            conversation_id
        )

        self._memories[
            conversation_id
        ] = memory

        # Prevent unbounded cache growth.
        if (
            len(self._memories)
            > self.MAX_MEMORY_CACHE
        ):
            self._memories.popitem(
                last=False
            )

        return memory

    def chat(
        self,
        message: str,
        conversation_id: str,
    ):

        memory = self._get_memory(
            conversation_id
        )

        (
            answer,
            results,
            search_query,
            tools_used,
            timings,
        ) = ask(
            message,
            memory,
        )

        return {
            "answer": answer,
            "sources": results,
            "search_query": search_query,
            "tools_used": tools_used,
        }

    def clear_memory(
        self,
        conversation_id: str,
    ):
        memory = self._get_memory(
            conversation_id
        )

        memory.clear()

    def get_history(
        self,
        conversation_id: str,
    ):
        return self._get_memory(
            conversation_id
        ).get_full_history()

    def list_conversations(self):
        return ConversationMemory.list_all()

    def create_conversation(self) -> str:
        conversation_id = (
            ConversationMemory.create_new()
        )

        # Cache it immediately.
        self._memories[
            conversation_id
        ] = ConversationMemory(
            conversation_id
        )

        return conversation_id


if __name__ == "__main__":

    stella = StellaEngine()

    conversation_id = (
        stella.create_conversation()
    )

    print("STELLA\n======\n")

    while True:

        message = input("You: ").strip()

        if message.lower() == "exit":
            break

        response = stella.chat(
            message,
            conversation_id,
        )

        print(
            "\nStella:",
            response["answer"],
            "\n",
        )