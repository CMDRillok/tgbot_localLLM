"""CLI для ручного управления llama-server (без запуска бота)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgllm.config import AppConfig, ConfigError, load_config, load_defaults  # noqa: E402
from tgllm.db import Database  # noqa: E402
from tgllm.models import choose_model, scan_models  # noqa: E402
from tgllm.server import LlamaServer, ServerError  # noqa: E402
from tgllm.store import SettingsStore  # noqa: E402


def _tail_lines(text: str, lines: int) -> str:
    rows = text.splitlines()
    return "\n".join(rows[-lines:])


async def _run(args: argparse.Namespace, config: AppConfig) -> int:
    db = Database(config.db_path)
    await db.connect()
    try:
        store = SettingsStore(db, load_defaults())
        await store.init()
        server = LlamaServer(config, store)
        try:
            if args.command == "logs":
                raw = server.log_tail(20000)
                print(_tail_lines(raw, args.lines))
                return 0
            if args.command == "models":
                models = scan_models(config.models_dirs)
                if not models:
                    print("Модели не найдены. Проверь MODELS_DIRS.")
                for model in models:
                    ctx = f", ctx {model.context_length}" if model.context_length else ""
                    print(
                        f"{model.short_name:40s} {model.size_gb:5.1f} ГБ  "
                        f"{model.quant}{ctx}\n  {model.path}"
                    )
                return 0
            if args.command == "stop":
                await server.stop()
                print("llama-server остановлен.")
                return 0

            models = scan_models(config.models_dirs)
            model = choose_model(models, store.get_str("model"))
            if args.command in {"start", "restart"}:
                if model is None:
                    print("Не найдено ни одной .gguf-модели в MODELS_DIRS.", file=sys.stderr)
                    return 1
                if args.command == "restart":
                    await server.stop()
                await server.start(model.path)
                print(f"llama-server запущен: {model.short_name}")
                print(f"URL: {config.llm_base_url}")
                return 0

            status = await server.status()
            loaded = await server.current_model() if status.healthy else None
            icon = "работает" if status.healthy else "остановлен"
            print(f"Статус: {icon}")
            print(f"URL: {config.llm_base_url}")
            print(f"PID: {status.pid or '—'}")
            print(f"Модель в настройках: {model.short_name if model else '—'}")
            if loaded is not None:
                print(f"Загружено сейчас: {loaded}")
            return 0 if status.healthy else 1
        finally:
            await server.aclose()
    finally:
        await db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tgllm-server", description="Управление llama-server для Telegram-бота"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=("start", "stop", "restart", "status", "logs", "models"),
    )
    parser.add_argument("-n", "--lines", type=int, default=40, help="строк лога для logs")
    args = parser.parse_args(argv)
    try:
        config = load_config()
        return asyncio.run(_run(args, config))
    except (ConfigError, ServerError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
