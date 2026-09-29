"""
memory.py

Conversation memory for Stella, persisted to SQLite, scoped by
conversation_id so multiple conversations can exist side by side.
"""

import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("data/memory.db")

class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()

def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
                DB_PATH,
                factory=_ClosingConnection,
            )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            tools_used TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute(
            """
            CREATE INDEX IF NOT EXISTS messages_conversation_id
            ON messages(conversation_id, id)
            """
        )
    return conn


class ConversationMemory:

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        with _connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
                (conversation_id, None, datetime.now(timezone.utc).isoformat()),
            )

    def add_user_message(self, message: str):
        self._add("user", message)

    def add_assistant_message(self, message: str, sources=None, tools_used=None):
        self._add("assistant", message, sources=sources, tools_used=tools_used)
        self._maybe_set_title()

    def _add(self, role: str, content: str, sources=None, tools_used=None):
        with _connect() as conn:
            conn.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, sources, tools_used, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.conversation_id, role, content,
                 json.dumps(sources) if sources else None,
                 json.dumps(tools_used) if tools_used else None,
                 datetime.now(timezone.utc).isoformat()),
            )

    def add_turn(
            self,
            user_message: str,
            answer: str,
            sources=None,
            tools_used=None,
        ):
            """
            Save one complete conversational turn in one transaction.
            """
            now = datetime.now(timezone.utc).isoformat()

            with _connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO messages (
                        conversation_id,
                        role,
                        content,
                        sources,
                        tools_used,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            self.conversation_id,
                            "user",
                            user_message,
                            None,
                            None,
                            now,
                        ),
                        (
                            self.conversation_id,
                            "assistant",
                            answer,
                            json.dumps(sources) if sources else None,
                            json.dumps(tools_used) if tools_used else None,
                            now,
                        ),
                    ],
                )

                conn.execute(
                    """
                    UPDATE conversations
                    SET title = ?
                    WHERE id = ? AND title IS NULL
                    """,
                    (
                        user_message[:40],
                        self.conversation_id,
                    ),
                )

    def get_messages(
            self,
            limit: int = 20,
            *,
            include_metadata: bool = False,
        ) -> list[dict]:
            """
            Read recent messages, optionally including evidence/tool metadata.
            """
            with _connect() as conn:
                conn.row_factory = sqlite3.Row

                rows = conn.execute(
                    """
                    SELECT role, content, sources, tools_used
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (
                        self.conversation_id,
                        limit,
                    ),
                ).fetchall()

            messages = []

            for row in reversed(rows):
                message = {
                    "role": row["role"],
                    "content": row["content"],
                }

                if include_metadata:
                    for key in ("sources", "tools_used"):
                        try:
                            value = (
                                json.loads(row[key])
                                if row[key]
                                else []
                            )
                        except (ValueError, TypeError):
                            value = []

                        # Older failed searches could store an error dictionary.
                        message[key] = (
                            value
                            if isinstance(value, list)
                            else []
                        )

                messages.append(message)

            return messages

    def get_full_history(self) -> list[dict]:
        """Every message with sources/tools_used, for the UI to render on load."""
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT role, content, sources, tools_used FROM messages
                   WHERE conversation_id = ? ORDER BY id ASC""",
                (self.conversation_id,),
            ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "sources": json.loads(r["sources"]) if r["sources"] else [],
                "tools_used": json.loads(r["tools_used"]) if r["tools_used"] else [],
            }
            for r in rows
        ]

    def clear(self):
        with _connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (self.conversation_id,))

    def is_empty(self) -> bool:
        with _connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (self.conversation_id,)
            ).fetchone()[0]
        return count == 0

    def _maybe_set_title(self):
        """First user message becomes the conversation's title, once."""
        with _connect() as conn:
            row = conn.execute(
                "SELECT title FROM conversations WHERE id = ?", (self.conversation_id,)
            ).fetchone()
            if row and row[0] is None:
                first_user = conn.execute(
                    """SELECT content FROM messages WHERE conversation_id = ? AND role = 'user'
                       ORDER BY id ASC LIMIT 1""",
                    (self.conversation_id,),
                ).fetchone()
                if first_user:
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (first_user[0][:40], self.conversation_id),
                    )

    @staticmethod
    def create_new() -> str:
        conversation_id = str(uuid.uuid4())
        with _connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
                (conversation_id, None, datetime.now(timezone.utc).isoformat()),
            )
        return conversation_id

    @staticmethod
    def list_all() -> list[dict]:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"id": r["id"], "title": r["title"] or "New conversation", "created_at": r["created_at"]}
            for r in rows
        ]