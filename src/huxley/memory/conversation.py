"""
Conversation memory management.

Handles storage and retrieval of conversation history,
with support for:
- Conversation persistence
- Message windowing
- Token budget management
- Context summarization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from huxley.core.types import Message, MessageRole, generate_id
from huxley.memory.store import InMemoryStore, MemoryStore


@dataclass
class Conversation:
    """A conversation with metadata."""

    id: str = field(default_factory=generate_id)
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def get_messages(
        self,
        *,
        last_n: int | None = None,
        roles: list[MessageRole] | None = None,
    ) -> list[Message]:
        """
        Get messages from the conversation.

        Args:
            last_n: Return only the last N messages
            roles: Filter by message roles

        Returns:
            Filtered list of messages
        """
        messages = self.messages

        if roles:
            messages = [m for m in messages if m.role in roles]

        if last_n:
            messages = messages[-last_n:]

        return messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role.value,
                    "content": m.content,
                    "name": m.name,
                    "tool_call_id": m.tool_call_id,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        messages = []
        for m in data.get("messages", []):
            messages.append(
                Message(
                    id=m["id"],
                    role=MessageRole(m["role"]),
                    content=m.get("content"),
                    name=m.get("name"),
                    tool_call_id=m.get("tool_call_id"),
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                    metadata=m.get("metadata", {}),
                )
            )

        return cls(
            id=data["id"],
            messages=messages,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


class ConversationMemory:
    """
    Manages conversation history with persistence.

    Provides:
    - Conversation CRUD operations
    - Message windowing for context management
    - Automatic summarization (optional)
    """

    def __init__(
        self,
        store: MemoryStore[dict[str, Any]] | None = None,
        *,
        max_messages: int = 100,
        key_prefix: str = "conversation:",
    ) -> None:
        """
        Initialize conversation memory.

        Args:
            store: Backend store (in-memory if not provided)
            max_messages: Maximum messages to keep per conversation
            key_prefix: Prefix for storage keys
        """
        self._store = store or InMemoryStore()
        self._max_messages = max_messages
        self._key_prefix = key_prefix

    def _key(self, conversation_id: str) -> str:
        """Generate storage key for a conversation."""
        return f"{self._key_prefix}{conversation_id}"

    async def create(
        self,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            system_prompt: Optional system message to start with
            metadata: Optional metadata

        Returns:
            New Conversation instance
        """
        conversation = Conversation(metadata=metadata or {})

        if system_prompt:
            conversation.add_message(
                Message(role=MessageRole.SYSTEM, content=system_prompt)
            )

        await self._store.set(self._key(conversation.id), conversation.to_dict())
        return conversation

    async def get(self, conversation_id: str) -> Conversation | None:
        """
        Retrieve a conversation by ID.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation or None if not found
        """
        data = await self._store.get(self._key(conversation_id))
        if data is None:
            return None
        return Conversation.from_dict(data)

    async def save(self, conversation: Conversation) -> None:
        """
        Save a conversation.

        Automatically trims to max_messages if exceeded.

        Args:
            conversation: Conversation to save
        """
        # Trim if needed
        if len(conversation.messages) > self._max_messages:
            # Keep system message if present
            system_messages = [
                m for m in conversation.messages if m.role == MessageRole.SYSTEM
            ]
            other_messages = [
                m for m in conversation.messages if m.role != MessageRole.SYSTEM
            ]

            # Keep system messages + last N other messages
            keep_count = self._max_messages - len(system_messages)
            conversation.messages = system_messages + other_messages[-keep_count:]

        await self._store.set(self._key(conversation.id), conversation.to_dict())

    async def delete(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if deleted
        """
        return await self._store.delete(self._key(conversation_id))

    async def add_message(
        self,
        conversation_id: str,
        message: Message,
    ) -> Conversation | None:
        """
        Add a message to an existing conversation.

        Args:
            conversation_id: Conversation identifier
            message: Message to add

        Returns:
            Updated conversation or None if not found
        """
        conversation = await self.get(conversation_id)
        if conversation is None:
            return None

        conversation.add_message(message)
        await self.save(conversation)
        return conversation

    async def get_messages(
        self,
        conversation_id: str,
        *,
        last_n: int | None = None,
        include_system: bool = True,
    ) -> list[Message]:
        """
        Get messages from a conversation.

        Args:
            conversation_id: Conversation identifier
            last_n: Return only last N messages
            include_system: Include system messages

        Returns:
            List of messages
        """
        conversation = await self.get(conversation_id)
        if conversation is None:
            return []

        messages = conversation.messages

        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if last_n:
            # Always include system message if present
            if include_system:
                system = [m for m in messages if m.role == MessageRole.SYSTEM]
                others = [m for m in messages if m.role != MessageRole.SYSTEM]
                return system + others[-last_n:]
            else:
                return messages[-last_n:]

        return messages

    async def list_conversations(
        self,
        *,
        limit: int = 100,
    ) -> list[str]:
        """
        List all conversation IDs.

        Args:
            limit: Maximum number to return

        Returns:
            List of conversation IDs
        """
        keys = await self._store.list_keys(f"{self._key_prefix}*")
        ids = [k.replace(self._key_prefix, "") for k in keys]
        return ids[:limit]

    async def get_context_window(
        self,
        conversation_id: str,
        *,
        max_tokens: int = 4000,
        token_counter: Any | None = None,
    ) -> list[Message]:
        """
        Get messages that fit within a token budget.

        This is useful for preparing context for LLM calls
        while respecting context window limits.

        Args:
            conversation_id: Conversation identifier
            max_tokens: Maximum token budget
            token_counter: Optional function to count tokens

        Returns:
            Messages that fit within budget
        """
        conversation = await self.get(conversation_id)
        if conversation is None:
            return []

        if token_counter is None:
            # Rough estimate: 4 chars per token
            def estimate_tokens(msg: Message) -> int:
                content = msg.content or ""
                return len(content) // 4 + 10  # +10 for message overhead

            token_counter = estimate_tokens

        # Always include system message
        result = []
        token_count = 0

        system_messages = [
            m for m in conversation.messages if m.role == MessageRole.SYSTEM
        ]
        for msg in system_messages:
            tokens = token_counter(msg)
            result.append(msg)
            token_count += tokens

        # Add messages from newest to oldest until budget exhausted
        other_messages = [
            m for m in conversation.messages if m.role != MessageRole.SYSTEM
        ]
        for msg in reversed(other_messages):
            tokens = token_counter(msg)
            if token_count + tokens > max_tokens:
                break
            result.insert(len(system_messages), msg)  # Insert after system
            token_count += tokens

        return result
