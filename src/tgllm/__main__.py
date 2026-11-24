"""Точка входа Telegram-бота."""

from __future__ import annotations

import asyncio
import sys

from aiogram.exceptions import TelegramUnauthorizedError

from tgllm.config import ConfigError, load_config
from tgllm.telegram.bot import run_bot


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2
    if not config.bot_token:
        print(
            "BOT_TOKEN не задан. Скопируй .env.example в .env и вставь токен от @BotFather.",
            file=sys.stderr,
        )
        return 2
    if not config.owner_id:
        print("OWNER_ID не задан. Укажи свой Telegram ID в .env.", file=sys.stderr)
        return 2
    try:
        asyncio.run(run_bot(config))
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, TelegramUnauthorizedError) as exc:
        print(f"Не удалось запустить бота: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
