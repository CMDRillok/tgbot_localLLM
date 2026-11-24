"""Панель владельца: модели и параметры llama.cpp."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import BaseFilter, Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message, TelegramObject

from tgllm.models import ModelInfo, choose_model, scan_models
from tgllm.specs import SECTIONS, SPECS
from tgllm.telegram.keyboards import (
    access_menu,
    main_menu,
    model_menu,
    restart_keyboard,
    section_menu,
    server_menu,
)

if TYPE_CHECKING:
    from tgllm.telegram.bot import AppState

logger = logging.getLogger(__name__)


class OwnerFlow(StatesGroup):
    value = State()


class IsOwner(BaseFilter):
    def __init__(self, state: AppState) -> None:
        self.state = state

    async def __call__(self, event: TelegramObject, **kwargs: object) -> bool:
        user = getattr(event, "from_user", None)
        return bool(user and user.id == self.state.config.owner_id)


async def _send_or_edit(
    state: AppState,
    chat_id: int,
    message_id: int | None,
    text: str,
    markup=None,  # noqa: ANN001 - InlineKeyboardMarkup | None
) -> None:
    if message_id is None:
        await state.bot.send_message(chat_id, text, reply_markup=markup)
        return
    try:
        await state.bot.edit_message_text(
            text=text, chat_id=chat_id, message_id=message_id, reply_markup=markup
        )
    except TelegramBadRequest as exc:
        if "not modified" in str(exc).lower():
            return
        await state.bot.send_message(chat_id, text, reply_markup=markup)


async def show_main(state: AppState, chat_id: int, message_id: int | None = None) -> None:
    model = state.resolve_model()
    status = await state.server.status()
    if status.healthy:
        icon = "✅"
    elif status.pid is not None:
        icon = "⏳"
    else:
        icon = "❌"
    text = (
        "⚙️ Управление ботом\n\n"
        f"Модель: {model.short_name if model else '—'}\n"
        f"Сервер: {icon} · порт {state.config.llm_port}\n\n"
        "Выбери раздел:"
    )
    await _send_or_edit(state, chat_id, message_id, text, main_menu())


async def show_section(
    state: AppState, chat_id: int, message_id: int | None, section_key: str
) -> None:
    if section_key == "access":
        await show_access(state, chat_id, message_id)
        return
    section = SECTIONS.get(section_key)
    if section is None:
        await show_main(state, chat_id, message_id)
        return
    values = state.store.snapshot()
    restart_note = (
        "\n⚠️ Изменения применятся после перезапуска сервера."
        if any(spec.restart for spec in section.specs)
        else ""
    )
    text = (
        f"{section.title}\n\n"
        f"{section.hint}{restart_note}\n\n"
        "Нажми на параметр, чтобы изменить значение. "
        "Для параметров с вариантами нажатие переключает значение."
    )
    await _send_or_edit(state, chat_id, message_id, text, section_menu(section, values))


async def show_models(
    state: AppState, chat_id: int, message_id: int | None = None, rescan: bool = False
) -> None:
    state.models = scan_models(state.config.models_dirs)
    current = state.store.get_str("model")
    if rescan:
        model = choose_model(state.models, current)
        if model is not None and not any(
            str(item.path) == current for item in state.models
        ):
            await state.store.set("model", str(model.path))
            current = str(model.path)
    current_model = next(
        (item for item in state.models if str(item.path) == current), None
    )
    dirs = "\n".join(f"• {path}" for path in state.config.models_dirs)
    context = (
        f"\nКонтекст модели: {current_model.context_length}"
        if current_model and current_model.context_length
        else ""
    )
    text = (
        "🧠 Модель\n\n"
        f"Текущая: {current_model.short_name if current_model else '—'}"
        f"{context}\n"
        f"Файл: {current_model.path if current_model else '—'}\n\n"
        f"Каталоги поиска:\n{dirs}\n\n"
        "Нажми на модель, чтобы переключить. Сервер перезапустится."
    )
    await _send_or_edit(
        state, chat_id, message_id, text, model_menu(state.models, current)
    )


async def show_access(
    state: AppState, chat_id: int, message_id: int | None = None
) -> None:
    allowed = await state.db.allowed_users()
    lines = [
        "👥 Доступ",
        "",
        f"Владелец: {state.config.owner_id}",
        f"Разрешённых пользователей: {len(allowed)}",
        "",
        "Добавить: /allow <id>",
        "Удалить: /deny <id> или нажми на пользователя ниже.",
        "",
        "ID пользователя виден ему в сообщении об отказе.",
    ]
    await _send_or_edit(
        state, chat_id, message_id, "\n".join(lines), access_menu(allowed, state.config.owner_id)
    )


async def show_server(
    state: AppState, chat_id: int, message_id: int | None = None
) -> None:
    status = await state.server.status()
    model = state.resolve_model()
    if status.healthy:
        icon = "✅ работает"
    elif status.pid is not None:
        icon = f"⏳ запускается (pid {status.pid})"
    else:
        icon = "❌ остановлен"
    loaded = await state.server.current_model() if status.healthy else None
    values = state.store.snapshot()
    lines = [
        "🖥 llama-server",
        "",
        f"Статус: {icon}",
        f"Порт: {state.config.llm_port}",
        f"Модель в настройках: {model.short_name if model else '—'}",
    ]
    if loaded is not None:
        lines.append(f"Загружено сейчас: {loaded.name}")
    lines += [
        f"ctx-size: {values['ctx_size']} · ngl: {values['n_gpu_layers']} · fit: {values['fit']}",
        f"flash-attn: {values['flash_attn']} · "
        f"KV: {values['cache_type_k']}/{values['cache_type_v']}",
        f"Слотов: {values['parallel']} · threads: {values['threads']}",
    ]
    await _send_or_edit(state, chat_id, message_id, "\n".join(lines), server_menu())


async def restart_flow(
    state: AppState,
    chat_id: int,
    message_id: int | None,
    model: ModelInfo | None = None,
    reason: str = "Перезапускаю сервер",
) -> None:
    model = model or state.resolve_model()
    if model is None:
        await _send_or_edit(
            state,
            chat_id,
            message_id,
            "❌ Не найдено ни одной .gguf-модели.\n"
            "Положи файл в один из каталогов MODELS_DIRS и обнови список в /model.",
        )
        return
    await _send_or_edit(state, chat_id, message_id, f"⏳ {reason}…")
    if state.gate.locked():
        await _send_or_edit(
            state, chat_id, message_id, "⏳ Жду завершения текущего ответа…"
        )
    try:
        async with state.gate:
            await _send_or_edit(
                state,
                chat_id,
                message_id,
                f"⏳ Запускаю {model.short_name}…\nЭто может занять до минуты.",
            )
            await state.server.start(model.path)
        await _send_or_edit(
            state,
            chat_id,
            message_id,
            f"✅ Готово. Модель: {model.short_name}\n"
            f"Порт: {state.config.llm_port} · ctx-size: {state.store.get_int('ctx_size')}",
            server_menu(),
        )
    except Exception as exc:  # noqa: BLE001 - показываем причину владельцу
        logger.exception("Не удалось запустить llama-server")
        await _send_or_edit(
            state,
            chat_id,
            message_id,
            f"❌ Не удалось запустить сервер:\n{str(exc)[:3200]}",
            server_menu(),
        )


def build_owner_router(state: AppState) -> Router:
    router = Router(name="owner")
    owner = IsOwner(state)

    @router.message(owner, CommandStart())
    async def cmd_start(message: Message) -> None:
        model = state.resolve_model()
        await message.answer(
            "👋 Привет, владелец!\n"
            f"Модель: {model.short_name if model else '—'}\n\n"
            "Просто пиши сообщения — я отвечу.\n"
            "/settings — модель, llama.cpp, генерация и история\n"
            "/model — список моделей\n"
            "/server — состояние и перезапуск\n"
            "/allow <id>, /deny <id> — доступ пользователей\n"
            "/new — очистить историю, /stop — прервать ответ"
        )

    @router.message(owner, Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(
            "Команды владельца:\n"
            "/settings — настройки и модель\n"
            "/model — переключить модель\n"
            "/server — llama-server: статус, лог, рестарт\n"
            "/allow <id> — выдать доступ\n"
            "/deny <id> — забрать доступ\n"
            "/new — очистить историю\n"
            "/stop — остановить генерацию\n"
            "/cancel — отменить ввод значения"
        )

    @router.message(owner, Command("settings", "panel"))
    async def cmd_settings(message: Message, fsm: FSMContext) -> None:
        await fsm.clear()
        await show_main(state, message.chat.id)

    @router.message(owner, Command("model"))
    async def cmd_model(message: Message, fsm: FSMContext) -> None:
        await fsm.clear()
        await show_models(state, message.chat.id)

    @router.message(owner, Command("server"))
    async def cmd_server(message: Message, fsm: FSMContext) -> None:
        await fsm.clear()
        await show_server(state, message.chat.id)

    @router.message(owner, Command("cancel"))
    async def cmd_cancel(message: Message, fsm: FSMContext) -> None:
        await fsm.clear()
        await message.answer("Отменено.")

    @router.message(owner, Command("allow"))
    async def cmd_allow(message: Message) -> None:
        parts = (message.text or "").split()
        if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
            await message.answer("Использование: /allow <telegram_id>")
            return
        user_id = int(parts[1])
        if user_id == state.config.owner_id:
            await message.answer("Это ваш ID — владелец и так имеет доступ.")
            return
        await state.db.allow(user_id)
        await message.answer(f"✅ Пользователь {user_id} добавлен в allowlist.")

    @router.message(owner, Command("deny"))
    async def cmd_deny(message: Message) -> None:
        parts = (message.text or "").split()
        if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
            await message.answer("Использование: /deny <telegram_id>")
            return
        user_id = int(parts[1])
        if user_id == state.config.owner_id:
            await message.answer("Нельзя удалить владельца.")
            return
        await state.db.deny(user_id)
        await message.answer(f"🗑 Пользователь {user_id} удалён из allowlist.")

    # --- ввод значения ---

    @router.message(owner, StateFilter(OwnerFlow.value), F.text)
    async def on_value(message: Message, fsm: FSMContext) -> None:
        data = await fsm.get_data()
        key = data.get("key")
        spec = SPECS.get(str(key))
        if spec is None:
            await fsm.clear()
            await message.answer("Не понял, какой параметр менять. Открываю /settings.")
            await show_main(state, message.chat.id)
            return
        try:
            value = spec.parse(message.text or "")
        except ValueError as exc:
            await message.answer(
                f"❌ {exc}\n\n{spec.label}: {spec.range_hint()}\n"
                "Отправь значение ещё раз или /cancel."
            )
            return
        await state.store.set(key, value)
        await fsm.clear()
        note = (
            "\n\n⚠️ Применится после перезапуска сервера."
            if spec.restart
            else "\n\n✅ Применится к следующему ответу."
        )
        await message.answer(
            f"✅ {spec.label} = {spec.format(value)}{note}",
            reply_markup=restart_keyboard() if spec.restart else None,
        )
        if spec.section in SECTIONS:
            await show_section(state, message.chat.id, None, spec.section)

    # --- колбэки меню ---

    @router.callback_query(owner, F.data == "menu:main")
    async def cb_main(callback: CallbackQuery, fsm: FSMContext) -> None:
        await callback.answer()
        await fsm.clear()
        await show_main(state, callback.message.chat.id, callback.message.message_id)

    @router.callback_query(owner, F.data.startswith("sec:"))
    async def cb_section(callback: CallbackQuery, fsm: FSMContext) -> None:
        await callback.answer()
        await fsm.clear()
        section_key = (callback.data or "").split(":", 1)[1]
        await show_section(
            state, callback.message.chat.id, callback.message.message_id, section_key
        )

    @router.callback_query(owner, F.data.startswith("cfg:"))
    async def cb_cfg(callback: CallbackQuery, fsm: FSMContext) -> None:
        key = (callback.data or "").split(":", 1)[1]
        spec = SPECS.get(key)
        if spec is None:
            await callback.answer("Неизвестный параметр", show_alert=True)
            return
        await callback.answer()
        await fsm.set_state(OwnerFlow.value)
        await fsm.update_data(key=key)
        current = state.store.get(key)
        await callback.message.edit_text(
            f"{spec.label}\n\n"
            f"Сейчас: {spec.format(current)}\n"
            f"Допустимо: {spec.range_hint()}\n\n"
            f"{spec.hint}\n\n"
            "Отправь новое значение сообщением или /cancel."
        )

    @router.callback_query(owner, F.data.startswith("cyc:"))
    async def cb_cycle(callback: CallbackQuery, fsm: FSMContext) -> None:
        key = (callback.data or "").split(":", 1)[1]
        spec = SPECS.get(key)
        if spec is None or spec.kind != "choice":
            await callback.answer("Неизвестный параметр", show_alert=True)
            return
        await fsm.clear()
        current = state.store.get_str(key)
        try:
            index = spec.choices.index(current)
        except ValueError:
            index = -1
        value = spec.choices[(index + 1) % len(spec.choices)]
        await state.store.set(key, value)
        await callback.answer(f"{spec.label}: {value}")
        await show_section(
            state, callback.message.chat.id, callback.message.message_id, spec.section
        )

    @router.callback_query(owner, F.data == "mdl:menu")
    async def cb_models(callback: CallbackQuery, fsm: FSMContext) -> None:
        await callback.answer()
        await fsm.clear()
        await show_models(state, callback.message.chat.id, callback.message.message_id)

    @router.callback_query(owner, F.data == "mdl:rescan")
    async def cb_rescan(callback: CallbackQuery, fsm: FSMContext) -> None:
        await fsm.clear()
        await show_models(
            state, callback.message.chat.id, callback.message.message_id, rescan=True
        )
        await callback.answer(f"Найдено моделей: {len(state.models)}")

    @router.callback_query(owner, F.data.startswith("mdl:"))
    async def cb_model(callback: CallbackQuery, fsm: FSMContext) -> None:
        model_id = (callback.data or "").split(":", 1)[1]
        model = next((item for item in state.models if item.id == model_id), None)
        if model is None:
            await callback.answer("Модель не найдена. Обнови список.", show_alert=True)
            return
        await callback.answer("Переключаю…")
        await fsm.clear()
        await state.store.set("model", str(model.path))
        await restart_flow(
            state,
            callback.message.chat.id,
            callback.message.message_id,
            model=model,
            reason=f"Переключаю модель на {model.short_name}",
        )

    @router.callback_query(owner, F.data == "srv:menu")
    async def cb_server(callback: CallbackQuery, fsm: FSMContext) -> None:
        await callback.answer()
        await fsm.clear()
        await show_server(state, callback.message.chat.id, callback.message.message_id)

    @router.callback_query(owner, F.data == "srv:refresh")
    async def cb_refresh(callback: CallbackQuery) -> None:
        await callback.answer()
        await show_server(state, callback.message.chat.id, callback.message.message_id)

    @router.callback_query(owner, F.data == "srv:restart")
    async def cb_restart(callback: CallbackQuery) -> None:
        await callback.answer("Перезапускаю…")
        await restart_flow(
            state, callback.message.chat.id, callback.message.message_id, reason="Перезапускаю"
        )

    @router.callback_query(owner, F.data == "srv:start")
    async def cb_start(callback: CallbackQuery) -> None:
        if await state.server.health():
            await callback.answer("Сервер уже работает", show_alert=True)
            return
        await callback.answer("Запускаю…")
        await restart_flow(
            state, callback.message.chat.id, callback.message.message_id, reason="Запускаю сервер"
        )

    @router.callback_query(owner, F.data == "srv:stop")
    async def cb_stop(callback: CallbackQuery) -> None:
        await callback.answer("Останавливаю…")
        await _send_or_edit(
            state, callback.message.chat.id, callback.message.message_id, "⏳ Останавливаю…"
        )
        if state.gate.locked():
            await _send_or_edit(
                state,
                callback.message.chat.id,
                callback.message.message_id,
                "⏳ Жду завершения текущего ответа…",
            )
        async with state.gate:
            await state.server.stop()
        await show_server(state, callback.message.chat.id, callback.message.message_id)

    @router.callback_query(owner, F.data == "srv:logs")
    async def cb_logs(callback: CallbackQuery) -> None:
        await callback.answer()
        tail = await _read_log(state)
        await state.bot.send_message(callback.message.chat.id, tail)

    @router.callback_query(owner, F.data == "ctx:clear")
    async def cb_clear(callback: CallbackQuery, fsm: FSMContext) -> None:
        await fsm.clear()
        task = state.tasks.get(state.config.owner_id)
        if task is not None and not task.done():
            task.cancel()
        count = await state.db.clear_messages(state.config.owner_id)
        await callback.answer(f"Удалено: {count}", show_alert=True)

    @router.callback_query(owner, F.data.startswith("deny:"))
    async def cb_deny(callback: CallbackQuery) -> None:
        raw = (callback.data or "").split(":", 1)[1]
        if not raw.lstrip("-").isdigit():
            await callback.answer("Некорректный ID", show_alert=True)
            return
        user_id = int(raw)
        if user_id == state.config.owner_id:
            await callback.answer("Нельзя удалить владельца", show_alert=True)
            return
        await state.db.deny(user_id)
        await callback.answer(f"{user_id} удалён")
        await show_access(state, callback.message.chat.id, callback.message.message_id)

    return router


async def _read_log(state: AppState) -> str:
    tail = state.server.log_tail(3600)
    if len(tail) > 3900:
        tail = tail[-3900:]
    return "📜 Последние строки llama-server.log:\n\n" + tail
