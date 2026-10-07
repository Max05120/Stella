"""Versioned, lossless state snapshots and local SQLite storage (macOS/POSIX)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from core.agent import models as M, budgets as B


class AgentStorageError(RuntimeError):
    pass


class AgentBusyError(AgentStorageError):
    pass


_TYPES = {cls.__name__: cls for cls in (
    M.AgentState, M.AgentGoal, M.AgentStep, M.AgentDecision,
    M.AgentObservation, M.AgentConfirmation, M.AgentStatus,
    M.AgentDecisionType, M.AgentTerminationReason,
)}


def encode(value):
    # Tag every container: observation dictionaries cannot impersonate types.
    if isinstance(value, Enum):
        return {"kind": "enum", "type": type(value).__name__, "value": value.value}
    if is_dataclass(value) and type(value).__name__ in _TYPES:
        return {"kind": "record", "type": type(value).__name__, "value": {
            f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, dict):
        return {"kind": "dict", "value": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {"kind": type(value).__name__, "value": [encode(v) for v in value]}
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise AgentStorageError(f"Unsupported persistent value: {type(value).__name__}")


def decode(value):
    if not isinstance(value, dict):
        return value
    kind, body = value["kind"], value["value"]
    if kind == "enum":
        return _TYPES[value["type"]](body)
    if kind == "record":
        cls = _TYPES[value["type"]]
        if set(body) != {f.name for f in fields(cls)}:
            raise AgentStorageError("Snapshot fields do not match this runtime version")
        return cls(**{k: decode(v) for k, v in body.items()})
    if kind == "dict":
        return {decode(k): decode(v) for k, v in body}
    if kind == "path":
        return Path(body)
    if kind in {"list", "tuple", "set"}:
        return {"list": list, "tuple": tuple, "set": set}[kind](decode(v) for v in body)
    raise AgentStorageError("Unknown snapshot value type")


def dumps_state(state):
    now = B.monotonic()
    elapsed = state.elapsed_seconds
    if state.started_monotonic is not None and not state.is_terminal:
        elapsed = max(elapsed, now - state.started_monotonic)
    return json.dumps({
        "version": 2, "saved_wall": time.time(), "elapsed": elapsed,
        "runtime_remaining": None if state.deadline_monotonic is None else max(0, state.deadline_monotonic - now),
        "confirmation_remaining": None if state.confirmation is None else max(0, state.confirmation.expires_monotonic - now),
        "state": encode(state),
    }, allow_nan=False, ensure_ascii=False)


def loads_state(payload):
    try:
        def invalid_constant(value):
            raise AgentStorageError("Non-finite number in snapshot")
        data = json.loads(payload, parse_constant=invalid_constant)
        for name in ("saved_wall", "elapsed", "runtime_remaining", "confirmation_remaining"):
            value = data[name]
            if value is None and name.endswith("remaining"):
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise AgentStorageError("Invalid snapshot clock")
            if name != "saved_wall" and value < 0:
                raise AgentStorageError("Negative snapshot duration")
        if data["version"] not in {1, 2}:
            raise AgentStorageError("Unsupported agent snapshot version")
        if data["version"] == 1:
            tagged = data["state"]
            if tagged.get("kind") != "record" or tagged.get("type") != "AgentState":
                raise AgentStorageError("Legacy snapshot does not contain AgentState")
            # Only schema 1 permits these known missing fields.
            tagged["value"].setdefault("conversation_context", {"kind": "list", "value": []})
            tagged["value"].setdefault("resume_token", None)
        state = decode(data["state"])
        if not isinstance(state, M.AgentState):
            raise AgentStorageError("Snapshot does not contain AgentState")
        if state.is_terminal:
            return state
        downtime = time.time() - data["saved_wall"]
        # Clock rollback cannot grant extra approval time. Expire conservatively.
        rollback = downtime < 0
        downtime = max(0, downtime)
        now = B.monotonic()
        elapsed = data["elapsed"] + downtime
        if state.started_monotonic is not None:
            state.started_monotonic = now - elapsed
            remaining = 0 if rollback else max(0, data["runtime_remaining"] - downtime)
            state.deadline_monotonic = now + remaining
            state.elapsed_seconds = elapsed
        if state.confirmation is not None:
            remaining = 0 if rollback else max(0, data["confirmation_remaining"] - downtime)
            age = max(0, state.confirmation.expires_monotonic - state.confirmation.created_monotonic - remaining)
            state.confirmation = replace(state.confirmation,
                created_monotonic=now-age, expires_monotonic=now+remaining)
        return state
    except AgentStorageError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise AgentStorageError("Invalid agent snapshot; no action was resumed") from exc


class AgentStore:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.directory / "runs.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (conversation TEXT PRIMARY KEY, phase TEXT NOT NULL, payload TEXT NOT NULL)")
        self.database.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def locked(self, conversation):
        digest = hashlib.sha256(conversation.encode()).hexdigest()
        with (self.directory / (digest + ".lock")).open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AgentBusyError("This conversation already has an agent request in progress.") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self, conversation):
        try:
            with self.connect() as db:
                row = db.execute("SELECT phase, payload FROM runs WHERE conversation=?", (conversation,)).fetchone()
            if row is not None and row[0] not in {"idle", "busy"}:
                raise AgentStorageError("Invalid saved agent lifecycle phase")
            return None if row is None else (row[0], loads_state(row[1]))
        except sqlite3.Error as exc:
            raise AgentStorageError("Could not read durable agent state") from exc

    def save(self, conversation, state, phase):
        if phase not in {"busy", "idle"}:
            raise ValueError("Invalid persistence phase")
        payload = dumps_state(state)
        try:
            with self.connect() as db:
                db.execute("INSERT INTO runs VALUES (?,?,?) ON CONFLICT(conversation) DO UPDATE SET phase=excluded.phase, payload=excluded.payload",
                           (conversation, phase, payload))
        except sqlite3.Error as exc:
            raise AgentStorageError("Could not save durable agent state") from exc