"""Стриминг ответа в Telegram: аккуратные правки без флуда."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest

logger = logging.getLogger(__name__)

TELEGRAM_LIMIT = 3900
EDIT_INTERVAL = 1.2


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < int(limit * 0.6):
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


class StreamRenderer:
    def __init__(self, bot: Bot, chat_id: int) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.text = ""
        self._ids: list[int] = []
        self._rendered: dict[int, str] = {}
        self._last_flush = 0.0

    async def update(self, text: str, force: bool = False) -> None:
        self.text = text
        now = asyncio.get_running_loop().time()
        if not force and now - self._last_flush < EDIT_INTERVAL:
            return
        await self.flush()

    async def flush(self, fallback: str = "…") -> None:
        chunks = split_text(self.text) or [fallback]
        while len(self._ids) < len(chunks):
            message = await self.bot.send_message(self.chat_id, fallback)
            self._ids.append(message.message_id)
        for index, chunk in enumerate(chunks):
            if self._rendered.get(index) == chunk:
                continue
            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self._ids[index],
                    text=chunk,
                )
            except TelegramBadRequest as exc:
                if "not modified" not in str(exc).lower():
                    logger.warning("Не удалось обновить сообщение: %s", exc)
            self._rendered[index] = chunk
        self._last_flush = asyncio.get_running_loop().time()

    async def finish(self, text: str | None = None, fallback: str = "…") -> str:
        if text is not None:
            self.text = text
        await self.flush(fallback=fallback)
        return self.text
