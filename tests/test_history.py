from __future__ import annotations

from pathlib import Path

import pytest

from tgllm.db import Database
from tgllm.history import build_context, estimate_tokens, trim_to_budget
from tgllm.store import SettingsStore


def test_estimate_tokens_cyrillic_is_denser() -> None:
    assert estimate_tokens("привет") > estimate_tokens("hello")
    assert estimate_tokens("") == 0


def test_trim_keeps_last_message() -> None:
    messages = [
        {"role": "user", "content": "старое " * 100},
        {"role": "assistant", "content": "ответ " * 100},
        {"role": "user", "content": "новое"},
    ]
    trimmed = trim_to_budget(messages, 20)
    assert trimmed[-1]["content"] == "новое"
    assert trimmed[0]["role"] == "user"


def test_trim_to_zero_keeps_last() -> None:
    messages = [{"role": "user", "content": "x" * 500}]
    assert trim_to_budget(messages, 1) == messages


@pytest.mark.asyncio
async def test_history_roundtrip(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.sqlite3")
    await db.connect()
    try:
        await db.add_message(1, "user", "привет")
        await db.add_message(1, "assistant", "здравствуй")
        await db.add_message(1, "user", "как дела?")
        recent = await db.recent_messages(1, 2)
        assert recent == [("assistant", "здравствуй"), ("user", "как дела?")]

        context = await build_context(db, 1, "система", 10, 6000)
        assert context[0] == {"role": "system", "content": "система"}
        assert context[1]["role"] == "user"

        assert await db.clear_messages(1) == 3
        assert await db.recent_messages(1, 10) == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_settings_store(tmp_path: Path) -> None:
    db = Database(tmp_path / "settings.sqlite3")
    await db.connect()
    try:
        store = SettingsStore(db, {"temperature": 0.5, "model": ""})
        await store.init()
        assert store.get_float("temperature") == 0.5
        await store.set("temperature", 0.9)
        assert store.get_float("temperature") == 0.9
        # Значение переживает повторную инициализацию.
        second = SettingsStore(db, {"temperature": 0.5, "model": ""})
        await second.init()
        assert second.get_float("temperature") == 0.9
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_allowlist(tmp_path: Path) -> None:
    db = Database(tmp_path / "access.sqlite3")
    await db.connect()
    try:
        assert not await db.is_allowed(42)
        await db.allow(42)
        assert await db.is_allowed(42)
        await db.touch_user(42, "user", "User")
        assert await db.allowed_users() == [(42, "user")]
        await db.deny(42)
        assert not await db.is_allowed(42)
    finally:
        await db.close()
