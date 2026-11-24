"""Проверка доступа: владелец и allowlist."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

if TYPE_CHECKING:
    from tgllm.telegram.bot import AppState

logger = logging.getLogger(__name__)


class AccessMiddleware(BaseMiddleware):
    def __init__(self, state: AppState) -> None:
        self.state = state

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is None:
            return await handler(event, data)

        allowed = user.id == self.state.config.owner_id or await self.state.db.is_allowed(
            user.id
        )
        if not allowed:
            text = (
                "⛔ Доступ закрыт.\n"
                f"Ваш Telegram ID: {user.id}\n\n"
                "Передайте этот ID владельцу бота — он добавит вас командой /allow."
            )
            if isinstance(event, CallbackQuery):
                await event.answer("Нет доступа", show_alert=True)
            elif isinstance(event, Message):
                try:
                    await event.answer(text)
                except Exception:  # noqa: BLE001 - доступ закрыт, ответ не критичен
                    logger.debug("Не удалось отправить отказ пользователю %s", user.id)
            return None

        await self.state.db.touch_user(user.id, user.username, user.first_name)
        return await handler(event, data)
