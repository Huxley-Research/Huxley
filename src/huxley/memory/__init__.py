"""Memory and state management."""

from huxley.memory.store import MemoryStore, InMemoryStore
from huxley.memory.conversation import ConversationMemory
from huxley.memory.factory import (
    get_memory_store,
    get_conversation_memory,
    get_database_connection,
    save_research_session,
    save_molecule,
    log_tool_execution,
)

__all__ = [
    "ConversationMemory",
    "InMemoryStore",
    "MemoryStore",
    "get_memory_store",
    "get_conversation_memory",
    "get_database_connection",
    "save_research_session",
    "save_molecule",
    "log_tool_execution",
]

# Lazy imports for optional stores
def __getattr__(name):
    if name == "PostgresStore":
        from huxley.memory.postgres_store import PostgresStore
        return PostgresStore
    if name == "SQLiteStore":
        from huxley.memory.sqlite_store import SQLiteStore
        return SQLiteStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
