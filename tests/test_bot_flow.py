"""Интеграционный тест телеграм-слоя без сети: фейковая сессия Bot API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.base import BaseSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import TelegramMethod
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from tgllm.config import AppConfig, load_defaults
from tgllm.db import Database
from tgllm.models import ModelInfo, scan_models
from tgllm.server import ServerStatus
from tgllm.store import SettingsStore
from tgllm.telegram.access import AccessMiddleware
from tgllm.telegram.bot import AppState
from tgllm.telegram.chat import build_chat_router
from tgllm.telegram.owner import build_owner_router

TOKEN = "123456:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
OWNER = 5147569136
STRANGER = 777
FRIEND = 888


class FakeSession(BaseSession):
    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.calls: list[str] = []
        self.edits: list[str] = []
        self._message_id = 100
        self._callback_answers: list[str] = []

    async def close(self) -> None:
        return None

    async def stream_content(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        yield b""

    async def make_request(
        self, bot: Bot, method: TelegramMethod, timeout: int | None = None
    ) -> Any:
        name = type(method).__name__
        self.calls.append(name)
        if name == "GetMe":
            return User(id=bot.id, is_bot=True, first_name="Test", username="test_bot")
        if name in {"SendMessage", "EditMessageText"}:
            text = str(getattr(method, "text", ""))
            self.texts.append(text)
            if name == "EditMessageText":
                self.edits.append(text)
            self._message_id += 1
            chat_id = int(method.chat_id)  # type: ignore[attr-defined]
            return Message(
                message_id=self._message_id,
                date=datetime.now(),  # noqa: DTZ005
                chat=Chat(id=chat_id, type="private"),
                text=text,
            )
        if name == "AnswerCallbackQuery":
            self._callback_answers.append(str(getattr(method, "text", "")))
            return True
        return True


class FakeServer:
    def __init__(self, models: list[ModelInfo]) -> None:
        self.models = models
        self.started: list[Path] = []
        self.stopped = 0

    async def health(self, timeout: float = 2.0) -> bool:
        return True

    async def status(self) -> ServerStatus:
        return ServerStatus(pid=123, healthy=True, port=9932)

    async def current_model(self) -> Path:
        return self.models[0].path

    async def start(self, path: Path) -> None:
        self.started.append(path)

    async def stop(self) -> None:
        self.stopped += 1

    def log_tail(self, chars: int = 3500) -> str:
        return "fake log"


class FakeLLM:
    def __init__(self) -> None:
        self.requests: list[list[dict[str, str]]] = []

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any):  # noqa: ANN201
        self.requests.append(messages)
        for part in ("Привет", ", это", " тест."):
            yield part

    async def aclose(self) -> None:
        return None


def _config(tmp_path: Path) -> AppConfig:
    data = tmp_path / "data"
    return AppConfig(
        bot_token=TOKEN,
        owner_id=OWNER,
        models_dirs=(tmp_path,),
        data_dir=data,
        db_path=data / "test.sqlite3",
        log_file=data / "bot.log",
        llama_binary="llama-server",
        llm_port=9932,
        llm_base_url="http://127.0.0.1:9932",
        start_timeout=1,
        stop_server_on_exit=False,
        log_level="INFO",
    )


class Harness:
    def __init__(self) -> None:
        self.update_id = 0

    def message_update(self, user_id: int, text: str) -> Update:
        self.update_id += 1
        return Update(
            update_id=self.update_id,
            message=Message(
                message_id=self.update_id,
                date=datetime.now(),  # noqa: DTZ005
                chat=Chat(id=user_id, type="private"),
                from_user=User(id=user_id, is_bot=False, first_name=f"user{user_id}"),
                text=text,
            ),
        )

    def callback_update(self, user_id: int, data: str) -> Update:
        self.update_id += 1
        user = User(id=user_id, is_bot=False, first_name=f"user{user_id}")
        return Update(
            update_id=self.update_id,
            callback_query=CallbackQuery(
                id=str(self.update_id),
                from_user=user,
                chat_instance="test",
                data=data,
                message=Message(
                    message_id=self.update_id,
                    date=datetime.now(),  # noqa: DTZ005
                    chat=Chat(id=user_id, type="private"),
                    text="menu",
                ),
            ),
        )


@pytest.fixture
async def app(tmp_path: Path):  # noqa: ANN201
    (tmp_path / "model-a-Q4_K_M.gguf").write_bytes(b"GGUF")
    (tmp_path / "model-b-Q6_K.gguf").write_bytes(b"GGUF")
    models = scan_models((tmp_path,))
    assert len(models) == 2

    config = _config(tmp_path)
    db = Database(config.db_path)
    await db.connect()
    store = SettingsStore(db, load_defaults())
    await store.init()
    await store.set("model", str(models[0].path))

    session = FakeSession()
    bot = Bot(TOKEN, session=session, default=DefaultBotProperties(parse_mode=None))
    state = AppState(
        config=config,
        bot=bot,
        db=db,
        store=store,
        models=models,
        server=FakeServer(models),  # type: ignore[arg-type]
        llm=FakeLLM(),  # type: ignore[arg-type]
    )

    dispatcher = Dispatcher(storage=MemoryStorage())
    access = AccessMiddleware(state)
    dispatcher.message.outer_middleware(access)
    dispatcher.callback_query.outer_middleware(access)
    dispatcher.include_router(build_owner_router(state))
    dispatcher.include_router(build_chat_router(state))

    harness = Harness()
    try:
        yield state, dispatcher, session, harness
    finally:
        await db.close()
        await bot.session.close()


@pytest.mark.asyncio
async def test_owner_start_and_settings(app) -> None:  # noqa: ANN001
    state, dispatcher, session, harness = app
    await dispatcher.feed_update(
        state.bot, harness.message_update(OWNER, "/start")
    )
    assert any("владелец" in text.lower() for text in session.texts)

    await dispatcher.feed_update(
        state.bot, harness.message_update(OWNER, "/settings")
    )
    assert any("Управление ботом" in text for text in session.texts)


@pytest.mark.asyncio
async def test_stranger_is_denied(app) -> None:  # noqa: ANN001
    state, dispatcher, session, harness = app
    await dispatcher.feed_update(
        state.bot, harness.message_update(STRANGER, "привет")
    )
    assert any("Доступ закрыт" in text for text in session.texts)
    assert await state.db.recent_messages(STRANGER, 10) == []


@pytest.mark.asyncio
async def test_allow_and_chat(app) -> None:  # noqa: ANN001
    state, dispatcher, session, harness = app
    await dispatcher.feed_update(
        state.bot, harness.message_update(OWNER, f"/allow {FRIEND}")
    )
    await dispatcher.feed_update(
        state.bot, harness.message_update(FRIEND, "привет")
    )
    messages = await state.db.recent_messages(FRIEND, 10)
    assert messages == [("user", "привет"), ("assistant", "Привет, это тест.")]
    assert any("Привет, это тест." in text for text in session.texts)
    assert any("system" in [item["role"] for item in request] for request in state.llm.requests)


@pytest.mark.asyncio
async def test_owner_changes_temperature(app) -> None:  # noqa: ANN001
    state, dispatcher, session, harness = app
    await dispatcher.feed_update(
        state.bot, harness.callback_update(OWNER, "cfg:temperature")
    )
    assert any("Сейчас: 0.7" in text for text in session.texts)

    await dispatcher.feed_update(
        state.bot, harness.message_update(OWNER, "0,5")
    )
    assert state.store.get_float("temperature") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_owner_switches_model(app) -> None:  # noqa: ANN001
    state, dispatcher, session, harness = app
    target = state.models[1]
    await dispatcher.feed_update(
        state.bot, harness.callback_update(OWNER, f"mdl:{target.id}")
    )
    assert state.store.get_str("model") == str(target.path)
    assert target.path in state.server.started  # type: ignore[attr-defined]
    assert any("Готово" in text for text in session.texts)


@pytest.mark.asyncio
async def test_denied_callback(app) -> None:  # noqa: ANN001
    state, dispatcher, session, harness = app
    await dispatcher.feed_update(
        state.bot, harness.callback_update(STRANGER, "menu:main")
    )
    assert "Нет доступа" in session._callback_answers
