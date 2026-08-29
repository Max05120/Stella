"""
engine.py

Main application interface for Stella.

The rest of the application should communicate
with Stella through this class rather than directly
calling RAG, memory, or tools.
"""
from core.memory import ConversationMemory
from core.rag import ask


class StellaEngine:

    def chat(self, message: str, conversation_id: str):
        memory = ConversationMemory(conversation_id)
        answer, results, search_query, tools_used = ask(message, memory)
        return {
            "answer": answer,
            "sources": results,
            "search_query": search_query,
            "tools_used": tools_used,
        }

    def clear_memory(self, conversation_id: str):
        ConversationMemory(conversation_id).clear()

    def get_history(self, conversation_id: str):
        return ConversationMemory(conversation_id).get_full_history()

    def list_conversations(self):
        return ConversationMemory.list_all()

    def create_conversation(self) -> str:
        return ConversationMemory.create_new()


if __name__ == "__main__":
    stella = StellaEngine()
    conversation_id = stella.create_conversation()
    print("STELLA\n======\n")
    while True:
        message = input("You: ").strip()
        if message.lower() == "exit":
            break
        response = stella.chat(message, conversation_id)
        print("\nStella:", response["answer"], "\n")