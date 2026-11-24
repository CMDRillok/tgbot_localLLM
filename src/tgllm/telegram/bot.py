"""Сборка приложения: bot, dispatcher, состояние."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from dataclasses import dataclass, field

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramRetryAfter, TelegramUnauthorizedError
from aiogram.fsm.storage.memory import MemoryStorage

from tgllm.config import AppConfig, load_defaults
from tgllm.db import Database
from tgllm.llm import LLMClient
from tgllm.models import ModelInfo, choose_model, scan_models
from tgllm.server import LlamaServer
from tgllm.store import SettingsStore
from tgllm.telegram.access import AccessMiddleware
from tgllm.telegram.chat import build_chat_router
from tgllm.telegram.owner import build_owner_router

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    config: AppConfig
    bot: Bot
    db: Database
    store: SettingsStore
    models: list[ModelInfo]
    server: LlamaServer
    llm: LLMClient
    gate: asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks: dict[int, asyncio.Task] = field(default_factory=dict)  # type: ignore[type-arg]

    def resolve_model(self) -> ModelInfo | None:
        return choose_model(self.models, self.store.get_str("model"))


def setup_logging(config: AppConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.log_level, logging.INFO))
    root.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file = logging.FileHandler(config.log_file, encoding="utf-8")
    file.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file)
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def notify_owner(state: AppState, text: str) -> None:
    try:
        await state.bot.send_message(state.config.owner_id, text[:4000])
    except Exception:  # noqa: BLE001 - уведомление не должно ронять бота
        logger.exception("Не удалось отправить уведомление владельцу")


async def wait_for_bot(bot: Bot, attempts: int = 4, max_delay: float = 45.0) -> None:
    """Дожидается Telegram, если он временно отвечает flood control."""
    for attempt in range(1, attempts + 1):
        try:
            info = await bot.me()
            logger.info("Telegram: @%s (%s)", info.username, info.id)
            return
        except TelegramRetryAfter as exc:
            delay = max(1.0, min(float(exc.retry_after) + 1.0, max_delay))
            logger.warning(
                "Telegram flood control (попытка %d/%d), ждём %.0f с",
                attempt,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)
        except TelegramUnauthorizedError as exc:
            raise RuntimeError(
                "Telegram отклонил BOT_TOKEN. Проверь токен в .env."
            ) from exc
    raise RuntimeError(
        "Telegram слишком долго отвечает 429 Too Many Requests. "
        "Попробуй запустить позже или проверь, не запущен ли бот где-то ещё."
    )


async def _bootstrap(state: AppState) -> None:
    await asyncio.sleep(1.5)
    model = state.resolve_model()
    if model is None:
        await notify_owner(
            state,
            "⚠️ Не найдено ни одной .gguf-модели.\n"
            f"Проверь MODELS_DIRS: {', '.join(str(path) for path in state.config.models_dirs)}\n"
            "Положи модель и перезапусти бота.",
        )
        return
    if state.store.get_str("model") != str(model.path):
        await state.store.set("model", str(model.path))
    try:
        async with state.gate:
            restarted = await state.server.ensure_running(model.path)
    except Exception as exc:  # noqa: BLE001 - показываем причину владельцу
        logger.exception("Модель не запустилась при старте")
        await notify_owner(state, f"⚠️ Модель не запустилась:\n{str(exc)[:3500]}")
        return
    suffix = "запущена" if restarted else "уже работает"
    await notify_owner(
        state,
        f"✅ Бот онлайн. Модель {model.short_name} {suffix}.\n"
        f"Порт {state.config.llm_port}, /settings — настройки.",
    )


async def run_bot(config: AppConfig) -> None:
    setup_logging(config)
    logger.info("Запуск бота: %s", config.llm_base_url)

    db = Database(config.db_path)
    await db.connect()
    store = SettingsStore(db, load_defaults())
    await store.init()

    models = scan_models(config.models_dirs)
    logger.info("Найдено моделей: %d", len(models))

    server = LlamaServer(config, store)
    llm = LLMClient(config.llm_base_url)
    bot = Bot(config.bot_token, default=DefaultBotProperties(parse_mode=None))
    state = AppState(
        config=config,
        bot=bot,
        db=db,
        store=store,
        models=models,
        server=server,
        llm=llm,
    )

    dispatcher = Dispatcher(storage=MemoryStorage())
    access = AccessMiddleware(state)
    dispatcher.message.outer_middleware(access)
    dispatcher.callback_query.outer_middleware(access)
    dispatcher.include_router(build_owner_router(state))
    dispatcher.include_router(build_chat_router(state))

    bootstrap: asyncio.Task[None] | None = None
    try:
        await wait_for_bot(bot)
        bootstrap = asyncio.create_task(_bootstrap(state), name="bootstrap")
        await dispatcher.start_polling(
            bot, allowed_updates=dispatcher.resolve_used_update_types()
        )
    finally:
        if bootstrap is not None:
            bootstrap.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bootstrap
        if config.stop_server_on_exit:
            await server.stop()
        await llm.aclose()
        await server.aclose()
        await db.close()
        await bot.session.close()
        logger.info("Бот остановлен")
