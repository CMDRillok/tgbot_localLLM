"""Описание параметров, которые владелец меняет в боте."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["int", "float", "str", "choice"]


@dataclass(frozen=True, slots=True)
class Spec:
    key: str
    label: str
    section: str
    kind: Kind
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    restart: bool = False
    hint: str = ""
    multiline: bool = False
    unit: str = ""

    def parse(self, raw: str) -> int | float | str:
        text = raw.strip()
        if not text:
            raise ValueError("Пустое значение")
        if self.kind == "int":
            try:
                value: int | float = int(text.replace(" ", ""))
            except ValueError as exc:
                raise ValueError("Нужно целое число") from exc
        elif self.kind == "float":
            try:
                value = float(text.replace(",", "."))
            except ValueError as exc:
                raise ValueError("Нужно число, например 0.7") from exc
        elif self.kind == "choice":
            value = text.lower()
            if value not in self.choices:
                raise ValueError("Допустимо: " + ", ".join(self.choices))
            return value
        else:
            value = text
            if len(value) > 8000:
                raise ValueError("Слишком длинный текст (максимум 8000 символов)")
            return value
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"Минимум: {self.minimum:g}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"Максимум: {self.maximum:g}")
        return value

    def format(self, value: Any) -> str:
        if self.kind == "float":
            return f"{float(value):g}"
        if self.kind == "str" and self.multiline:
            return f"{len(str(value))} симв."
        return f"{value}{self.unit}"

    def range_hint(self) -> str:
        if self.kind == "choice":
            return "Одно из: " + ", ".join(self.choices)
        if self.kind == "str":
            return "Текст до 8000 символов"
        parts = []
        if self.minimum is not None:
            parts.append(f"от {self.minimum:g}")
        if self.maximum is not None:
            parts.append(f"до {self.maximum:g}")
        return " ".join(parts) or "любое"


@dataclass(frozen=True, slots=True)
class Section:
    key: str
    title: str
    hint: str = ""
    specs: tuple[Spec, ...] = field(default_factory=tuple)


LLAMA_SPECS = (
    Spec("ctx_size", "ctx-size", "llama", "int", 512, 131072, restart=True,
         hint="Размер контекста в токенах. Больше — длиннее память, но больше VRAM."),
    Spec("n_gpu_layers", "n-gpu-layers", "llama", "int", 0, 999, restart=True,
         hint="Сколько слоёв на GPU. 999 = авто: --fit подберёт под видеопамять."),
    Spec("threads", "threads", "llama", "int", 1, 64, restart=True,
         hint="Потоки CPU. На 12 ядрах обычно 8–10."),
    Spec("parallel", "parallel", "llama", "int", 1, 4, restart=True,
         hint="Слоты сервера. 1 достаточно: бот и так отвечает по одному."),
    Spec("fit", "fit", "llama", "choice", choices=("on", "off"), restart=True,
         hint="Автоподбор параметров под VRAM. Оставь on."),
    Spec("fit_target", "fit-target", "llama", "int", 0, 8192, restart=True,
         hint="Запас VRAM в МиБ для --fit. 256–1024."),
    Spec("flash_attn", "flash-attn", "llama", "choice",
         choices=("auto", "on", "off"), restart=True,
         hint="Flash Attention: быстрее и экономнее. На RTX 3070 — on."),
    Spec("cache_type_k", "cache-type-k", "llama", "choice",
         choices=("f16", "q8_0", "q4_0"), restart=True,
         hint="Тип KV-кэша ключей. q8_0 экономит память почти без потерь."),
    Spec("cache_type_v", "cache-type-v", "llama", "choice",
         choices=("f16", "q8_0", "q4_0"), restart=True,
         hint="Тип KV-кэша значений."),
)

GEN_SPECS = (
    Spec("temperature", "temperature", "gen", "float", 0.0, 2.0,
         hint="0 — строго и предсказуемо, 1+ — креативно."),
    Spec("top_p", "top-p", "gen", "float", 0.01, 1.0, hint="Ядерная выборка."),
    Spec("top_k", "top-k", "gen", "int", 0, 200, hint="0 = выключено."),
    Spec("repeat_penalty", "repeat-penalty", "gen", "float", 0.8, 2.0,
         hint="Штраф за повторы. 1.0 = выключен."),
    Spec("max_tokens", "max-tokens", "gen", "int", 64, 16384,
         hint="Максимум токенов в ответе."),
    Spec("reasoning", "reasoning", "gen", "choice", choices=("off", "on"),
         hint="Размышления Qwen3. Off — быстрее и дешевле по контексту."),
    Spec("system_prompt", "system prompt", "gen", "str", multiline=True,
         hint="Характер и правила ассистента. Применяется сразу."),
)

CTX_SPECS = (
    Spec("max_messages", "сообщений в контексте", "ctx", "int", 2, 400,
         hint="Сколько последних сообщений отправлять модели. Старые хранятся, но не мешают."),
    Spec("history_max_tokens", "лимит токенов истории", "ctx", "int", 512, 60000,
         hint="Обрезать историю, если она не влезает в этот бюджет."),
)

SECTIONS: dict[str, Section] = {
    "llama": Section("llama", "⚙️ llama.cpp", "Параметры сервера. Нужен перезапуск.", LLAMA_SPECS),
    "gen": Section("gen", "🎛 Генерация", "Применяются к следующему ответу.", GEN_SPECS),
    "ctx": Section("ctx", "💬 Контекст", "История хранится в SQLite.", CTX_SPECS),
}

SPECS: dict[str, Spec] = {
    spec.key: spec for section in SECTIONS.values() for spec in section.specs
}
