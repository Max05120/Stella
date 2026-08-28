"""
memory.py

Conversation memory for Stella.
"""


class ConversationMemory:

    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_assistant_message(self, message: str):
        self.messages.append({
            "role": "assistant",
            "content": message
        })

    def get_messages(self):
        return self.messages.copy()

    def clear(self):
        self.messages.clear()

    def is_empty(self):
        return len(self.messages) == 0