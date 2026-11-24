"""Загрузка конфигурации из .env и config/defaults.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULTS_PATH = PROJECT_ROOT / "config" / "defaults.yaml"


class ConfigError(RuntimeError):
    """Понятная человеку ошибка конфигурации."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    bot_token: str
    owner_id: int
    models_dirs: tuple[Path, ...]
    data_dir: Path
    db_path: Path
    log_file: Path
    llama_binary: str
    llm_port: int
    llm_base_url: str
    start_timeout: float
    stop_server_on_exit: bool
    log_level: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on", "да"}


def load_config(env_path: Path | None = None) -> AppConfig:
    """Читает .env. Не требует токен — CLI сервера работает и без бота."""
    load_dotenv(env_path or ENV_PATH, override=False)

    token = os.getenv("BOT_TOKEN", "").strip()
    raw_owner = os.getenv("OWNER_ID", "").strip()
    try:
        owner_id = int(raw_owner) if raw_owner else 0
    except ValueError as exc:
        raise ConfigError(f"OWNER_ID должен быть числом, получено: {raw_owner!r}") from exc

    raw_dirs = os.getenv("MODELS_DIRS", "").strip()
    if not raw_dirs:
        dirs = (Path.home() / "Downloads", PROJECT_ROOT / "models")
    else:
        dirs = tuple(
            Path(part).expanduser() for part in raw_dirs.split(os.pathsep) if part.strip()
        )

    raw_port = os.getenv("LLAMA_PORT", "9932").strip()
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ConfigError(f"LLAMA_PORT должен быть числом, получено: {raw_port!r}") from exc
    if not 1 <= port <= 65535:
        raise ConfigError("LLAMA_PORT должен быть в диапазоне 1..65535")

    raw_timeout = os.getenv("LLAMA_START_TIMEOUT", "300").strip()
    try:
        start_timeout = float(raw_timeout)
    except ValueError as exc:
        raise ConfigError("LLAMA_START_TIMEOUT должен быть числом") from exc

    data_dir = PROJECT_ROOT / "data"
    return AppConfig(
        bot_token=token,
        owner_id=owner_id,
        models_dirs=dirs,
        data_dir=data_dir,
        db_path=data_dir / "bot.sqlite3",
        log_file=data_dir / "bot.log",
        llama_binary=os.getenv("LLAMA_BINARY", "llama-server").strip() or "llama-server",
        llm_port=port,
        llm_base_url=f"http://127.0.0.1:{port}",
        start_timeout=start_timeout,
        stop_server_on_exit=_env_bool("STOP_SERVER_ON_EXIT", True),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
    )


def load_defaults(path: Path | None = None) -> dict[str, Any]:
    """Плоский словарь дефолтов из YAML."""
    config_path = path or DEFAULTS_PATH
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Не найден файл дефолтов: {config_path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Ожидался словарь в {config_path}")
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update({str(inner): inner_value for inner, inner_value in value.items()})
        else:
            flat[str(key)] = value
    return flat
