"""Инлайн-клавиатуры бота."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from tgllm.models import ModelInfo
from tgllm.specs import Section, Spec


def _markup(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=data) for text, data in row]
            for row in rows
        ]
    )


def main_menu() -> InlineKeyboardMarkup:
    return _markup(
        [
            [("🧠 Модель", "mdl:menu"), ("⚙️ llama.cpp", "sec:llama")],
            [("🎛 Генерация", "sec:gen"), ("💬 Контекст", "sec:ctx")],
            [("🖥 Сервер", "srv:menu"), ("👥 Доступ", "sec:access")],
        ]
    )


def section_menu(section: Section, values: dict[str, object]) -> InlineKeyboardMarkup:
    rows: list[list[tuple[str, str]]] = []
    for spec in section.specs:
        data = f"cyc:{spec.key}" if spec.kind == "choice" else f"cfg:{spec.key}"
        value = spec.format(values.get(spec.key, "—"))
        rows.append([(f"{spec.label}: {value}", data)])
    if section.key == "ctx":
        rows.append([("🧹 Очистить мою историю", "ctx:clear")])
    rows.append([("⬅️ Назад", "menu:main")])
    return _markup(rows)


def model_menu(models: list[ModelInfo], current_path: str) -> InlineKeyboardMarkup:
    rows: list[list[tuple[str, str]]] = []
    for model in models:
        mark = "✅ " if str(model.path) == current_path else ""
        label = (
            f"{mark}{model.short_name} · {model.size_gb:.1f} ГБ · {model.quant}"
        )
        rows.append([(label, f"mdl:{model.id}")])
    rows.append([("🔄 Обновить список", "mdl:rescan")])
    rows.append([("⬅️ Назад", "menu:main")])
    return _markup(rows)


def server_menu() -> InlineKeyboardMarkup:
    return _markup(
        [
            [("🔄 Перезапустить", "srv:restart"), ("📜 Лог", "srv:logs")],
            [("▶️ Запустить", "srv:start"), ("⏹ Остановить", "srv:stop")],
            [("🔄 Обновить", "srv:refresh"), ("⬅️ Назад", "menu:main")],
        ]
    )


def access_menu(allowed: list[tuple[int, str | None]], owner_id: int) -> InlineKeyboardMarkup:
    rows: list[list[tuple[str, str]]] = []
    for user_id, username in allowed:
        label = f"❌ {user_id}" + (f" @{username}" if username else "")
        rows.append([(label, f"deny:{user_id}")])
    rows.append([("⬅️ Назад", "menu:main")])
    return _markup(rows)


def restart_keyboard() -> InlineKeyboardMarkup:
    return _markup([[("🔄 Перезапустить сейчас", "srv:restart")], [("⬅️ В меню", "menu:main")]])


def specs_hint(specs: tuple[Spec, ...]) -> str:
    lines = []
    for spec in specs:
        flag = " (нужен рестарт)" if spec.restart else ""
        lines.append(f"• {spec.label}: {spec.hint}{flag}")
    return "\n".join(lines)
