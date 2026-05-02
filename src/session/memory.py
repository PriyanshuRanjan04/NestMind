"""
In-memory session store.

Design decision (documented in README):
  We use a simple in-memory dict keyed by session_id. Each session stores
  only the user-turn strings — we do not persist assistant responses or
  metadata to keep the context prompt compact and cheap.

  Trade-off: sessions are lost on process restart. This is acceptable for
  the assignment demo and explicitly called out in the README. A production
  system would back this with Redis or a DB with TTL expiry.

  The store is intentionally NOT async-locked — FastAPI runs in a single
  event loop, so dict access is safe without explicit locks in this model.
  A multi-process deployment (Gunicorn workers) would need a shared backend,
  which is exactly why DATABASE_URL / REDIS_URL are in .env.example.
"""
from __future__ import annotations

from collections import defaultdict
from threading import Lock

# Max number of turns stored per session. Older turns are dropped.
# 20 turns gives ~2000 tokens of history budget (generous for intent context).
_MAX_TURNS_PER_SESSION = 20


class SessionMemory:
    """
    Thread-safe in-memory store for conversation turn history.
    """

    def __init__(self, max_turns: int = _MAX_TURNS_PER_SESSION) -> None:
        self._max_turns = max_turns
        self._store: dict[str, list[str]] = defaultdict(list)
        self._lock = Lock()

    def get_history(self, session_id: str) -> list[str]:
        """Return the stored user turns for this session (oldest first)."""
        with self._lock:
            return list(self._store[session_id])

    def append_turn(self, session_id: str, user_turn: str) -> None:
        """Append a user turn and trim to max_turns."""
        with self._lock:
            turns = self._store[session_id]
            turns.append(user_turn)
            if len(turns) > self._max_turns:
                self._store[session_id] = turns[-self._max_turns :]

    def clear(self, session_id: str) -> None:
        """Clear history for a session (e.g. explicit reset)."""
        with self._lock:
            self._store.pop(session_id, None)


# Module-level singleton — shared across the FastAPI app lifetime
memory = SessionMemory()
