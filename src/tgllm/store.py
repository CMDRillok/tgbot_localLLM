"""Хранилище настроек: дефолты из YAML + значения из SQLite."""

from __future__ import annotations

from typing import Any

from tgllm.db import Database

MODEL_KEY = "model"


class SettingsStore:
    def __init__(self, db: Database, defaults: dict[str, Any]) -> None:
        self.db = db
        self.defaults = dict(defaults)
        self._values: dict[str, Any] = dict(defaults)

    async def init(self) -> None:
        stored = await self.db.load_settings()
        for key, value in self.defaults.items():
            if key not in stored:
                await self.db.save_setting(key, value)
        self._values = {**self.defaults, **stored}

    def snapshot(self) -> dict[str, Any]:
        return dict(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, self.defaults.get(key, default))

    def get_int(self, key: str) -> int:
        return int(self.get(key))

    def get_float(self, key: str) -> float:
        return float(self.get(key))

    def get_str(self, key: str) -> str:
        return str(self.get(key, ""))

    async def set(self, key: str, value: Any) -> None:
        await self.db.save_setting(key, value)
        self._values[key] = value
