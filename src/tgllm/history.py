"""История диалога и оценка токенов."""

from __future__ import annotations

from typing import Any

from tgllm.db import Database


def estimate_tokens(text: str) -> int:
    """Грубая оценка: кириллица ~2.2 символа на токен, латиница ~3.8."""
    if not text:
        return 0
    cyrillic = sum(1 for char in text if "\u0400" <= char <= "\u04ff")
    other = len(text) - cyrillic
    return int(cyrillic / 2.2 + other / 3.8) + 2


def trim_to_budget(messages: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Убирает самые старые сообщения, но всегда оставляет последнее."""
    if not messages:
        return []
    total = sum(estimate_tokens(str(item.get("content", ""))) for item in messages)
    index = 0
    while index < len(messages) - 1 and total > max_tokens:
        total -= estimate_tokens(str(messages[index].get("content", "")))
        index += 1
    trimmed = [dict(item) for item in messages[index:]]
    while len(trimmed) > 1 and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    return trimmed


async def build_context(
    db: Database,
    user_id: int,
    system_prompt: str,
    max_messages: int,
    max_tokens: int,
) -> list[dict[str, str]]:
    rows = await db.recent_messages(user_id, max_messages)
    messages: list[dict[str, Any]] = [
        {"role": role, "content": content} for role, content in rows
    ]
    budget = max(128, max_tokens - estimate_tokens(system_prompt) - 32)
    trimmed = trim_to_budget(messages, budget)
    return [{"role": "system", "content": system_prompt}, *trimmed]
