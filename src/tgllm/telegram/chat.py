"""Обычный чат: доступен всем пользователям из allowlist."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from tgllm.history import build_context
from tgllm.llm import LLMError
from tgllm.render import StreamRenderer

if TYPE_CHECKING:
    from tgllm.telegram.bot import AppState

logger = logging.getLogger(__name__)

HELP = (
    "💬 Просто напиши сообщение — я отвечу.\n\n"
    "/new — очистить историю диалога\n"
    "/stop — остановить генерацию\n"
    "/help — эта справка"
)


def build_chat_router(state: AppState) -> Router:
    router = Router(name="chat")

    @router.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        model = state.resolve_model()
        name = model.short_name if model else "модель не выбрана"
        await message.answer(
            "👋 Привет! Я локальный ассистент на llama.cpp.\n"
            f"Модель: {name}\n\n{HELP}"
        )

    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(HELP)

    @router.message(Command("new"))
    async def cmd_new(message: Message) -> None:
        user_id = message.from_user.id
        task = state.tasks.get(user_id)
        if task is not None and not task.done():
            task.cancel()
        count = await state.db.clear_messages(user_id)
        await message.answer(f"🧹 История очищена ({count} сообщений).")

    @router.message(Command("stop"))
    async def cmd_stop(message: Message) -> None:
        task = state.tasks.get(message.from_user.id)
        if task is None or task.done():
            await message.answer("Нечего останавливать.")
            return
        task.cancel()
        await message.answer("⏹ Останавливаю…")

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        text = (message.text or "").strip()
        if not text:
            return
        if text.startswith("/"):
            await message.answer("Неизвестная команда. /help")
            return
        await _handle_chat(state, message, text)

    return router


async def _handle_chat(state: AppState, message: Message, text: str) -> None:
    user = message.from_user
    user_id = user.id
    if len(text) > 20000:
        await message.answer("Слишком длинное сообщение. Максимум 20 000 символов.")
        return

    status = await state.server.status()
    if not status.healthy:
        if status.pid is not None:
            await message.answer(
                "⏳ Модель ещё загружается. Подожди немного и отправь сообщение снова."
            )
        else:
            await message.answer(
                "❌ llama-server не запущен.\n"
                "Владелец может запустить его: /settings → 🖥 Сервер."
            )
        return

    current = state.tasks.get(user_id)
    if current is not None and not current.done():
        await message.answer("⏳ Я ещё отвечаю на прошлое сообщение. /stop — прервать.")
        return

    await state.db.add_message(user_id, "user", text)
    messages = await build_context(
        state.db,
        user_id,
        state.store.get_str("system_prompt"),
        state.store.get_int("max_messages"),
        state.store.get_int("history_max_tokens"),
    )
    renderer = StreamRenderer(message.bot, message.chat.id)
    task = asyncio.current_task()
    state.tasks[user_id] = task  # type: ignore[assignment]

    queued_notice = False
    if state.gate.locked():
        await message.answer("⏳ В очереди: сначала отвечу другому пользователю.")
        queued_notice = True

    chunks: list[str] = []
    try:
        async with state.gate:
            if not await state.server.health():
                raise LLMError("llama-server не отвечает. Проверь /settings → Сервер.")
            await message.bot.send_chat_action(message.chat.id, "typing")
            async for delta in state.llm.stream_chat(
                messages,
                temperature=state.store.get_float("temperature"),
                top_p=state.store.get_float("top_p"),
                top_k=state.store.get_int("top_k"),
                repeat_penalty=state.store.get_float("repeat_penalty"),
                max_tokens=state.store.get_int("max_tokens"),
                reasoning=state.store.get_str("reasoning") == "on",
            ):
                chunks.append(delta)
                await renderer.update("".join(chunks))
        answer = "".join(chunks).strip()
        await renderer.finish(answer or "…")
        if answer:
            await state.db.add_message(user_id, "assistant", answer)
    except asyncio.CancelledError:
        partial = renderer.text.strip()
        try:
            await renderer.finish(
                (partial + "\n\n⏹ Остановлено") if partial else "⏹ Остановлено"
            )
        finally:
            if partial:
                await state.db.add_message(user_id, "assistant", partial)
    except LLMError as exc:
        logger.warning("Ошибка LLM у %s: %s", user_id, exc)
        await message.answer(f"❌ {exc}")
    except Exception:  # noqa: BLE001 - пользователю нужен внятный ответ
        logger.exception("Неожиданная ошибка в чате у %s", user_id)
        await message.answer("❌ Внутренняя ошибка. Подробности в логе бота.")
    finally:
        if state.tasks.get(user_id) is task:
            state.tasks.pop(user_id, None)
        if queued_notice:
            logger.debug("Очередь обработана для %s", user_id)
