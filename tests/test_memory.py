"""Tests for memory system."""

import pytest

from huxley.memory.store import InMemoryStore
from huxley.memory.conversation import Conversation, ConversationMemory
from huxley.core.types import Message, MessageRole


class TestInMemoryStore:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        store: InMemoryStore[str] = InMemoryStore()
        await store.set("key1", "value1")
        result = await store.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        store: InMemoryStore[str] = InMemoryStore()
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        store: InMemoryStore[str] = InMemoryStore()
        await store.set("key1", "value1")
        deleted = await store.delete("key1")
        assert deleted is True
        result = await store.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_exists(self):
        store: InMemoryStore[str] = InMemoryStore()
        await store.set("key1", "value1")
        assert await store.exists("key1") is True
        assert await store.exists("key2") is False

    @pytest.mark.asyncio
    async def test_list_keys(self):
        store: InMemoryStore[str] = InMemoryStore()
        await store.set("prefix:1", "a")
        await store.set("prefix:2", "b")
        await store.set("other:1", "c")

        all_keys = await store.list_keys()
        assert len(all_keys) == 3

        prefix_keys = await store.list_keys("prefix:*")
        assert len(prefix_keys) == 2


class TestConversation:
    def test_create(self):
        conv = Conversation()
        assert conv.id is not None
        assert len(conv.messages) == 0

    def test_add_message(self):
        conv = Conversation()
        msg = Message(role=MessageRole.USER, content="Hello")
        conv.add_message(msg)
        assert len(conv.messages) == 1

    def test_get_messages_filtered(self):
        conv = Conversation()
        conv.add_message(Message(role=MessageRole.SYSTEM, content="System"))
        conv.add_message(Message(role=MessageRole.USER, content="User"))
        conv.add_message(Message(role=MessageRole.ASSISTANT, content="Assistant"))

        user_msgs = conv.get_messages(roles=[MessageRole.USER])
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "User"

    def test_serialization(self):
        conv = Conversation()
        conv.add_message(Message(role=MessageRole.USER, content="Test"))

        data = conv.to_dict()
        restored = Conversation.from_dict(data)

        assert restored.id == conv.id
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Test"


class TestConversationMemory:
    @pytest.mark.asyncio
    async def test_create_and_get(self):
        memory = ConversationMemory()
        conv = await memory.create(system_prompt="You are helpful.")

        retrieved = await memory.get(conv.id)
        assert retrieved is not None
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].role == MessageRole.SYSTEM

    @pytest.mark.asyncio
    async def test_add_message(self):
        memory = ConversationMemory()
        conv = await memory.create()

        msg = Message(role=MessageRole.USER, content="Hello")
        updated = await memory.add_message(conv.id, msg)

        assert updated is not None
        assert len(updated.messages) == 1

    @pytest.mark.asyncio
    async def test_delete(self):
        memory = ConversationMemory()
        conv = await memory.create()

        deleted = await memory.delete(conv.id)
        assert deleted is True

        retrieved = await memory.get(conv.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_message_limit(self):
        memory = ConversationMemory(max_messages=5)
        conv = await memory.create()

        for i in range(10):
            await memory.add_message(
                conv.id,
                Message(role=MessageRole.USER, content=f"Message {i}"),
            )

        retrieved = await memory.get(conv.id)
        assert retrieved is not None
        assert len(retrieved.messages) <= 5
