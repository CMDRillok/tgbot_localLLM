"""Управление процессом llama-server: запуск, остановка, статус, логи."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from tgllm.config import PROJECT_ROOT, AppConfig
from tgllm.store import SettingsStore

logger = logging.getLogger(__name__)

STOP_TIMEOUT = 30.0


class ServerError(RuntimeError):
    """llama-server не запустился или упал."""


@dataclass(frozen=True, slots=True)
class ServerParams:
    binary: str
    model: Path
    alias: str
    host: str
    port: int
    ctx_size: int
    n_gpu_layers: int
    threads: int
    parallel: int
    flash_attn: str
    cache_type_k: str
    cache_type_v: str
    fit: str
    fit_target: int
    reasoning: str


@dataclass(frozen=True, slots=True)
class ServerStatus:
    pid: int | None
    healthy: bool
    port: int

    @property
    def running(self) -> bool:
        return self.healthy or self.pid is not None


def build_command(params: ServerParams) -> list[str]:
    command = [
        params.binary,
        "--model",
        str(params.model),
        "--alias",
        params.alias,
        "--host",
        params.host,
        "--port",
        str(params.port),
        "--ctx-size",
        str(params.ctx_size),
        "--parallel",
        str(params.parallel),
        "--threads",
        str(params.threads),
        "--flash-attn",
        params.flash_attn,
        "--cache-type-k",
        params.cache_type_k,
        "--cache-type-v",
        params.cache_type_v,
        "--fit",
        params.fit,
        "--fit-target",
        str(params.fit_target),
        "--reasoning",
        params.reasoning,
        "--jinja",
        "--no-webui",
    ]
    if params.n_gpu_layers < 999:
        command.extend(["--n-gpu-layers", str(params.n_gpu_layers)])
    elif params.fit == "off":
        command.extend(["--n-gpu-layers", "99"])
    return command


def _alias_for(path: Path) -> str:
    alias = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-")
    return (alias or "local-model")[:64]


class LlamaServer:
    def __init__(self, config: AppConfig, store: SettingsStore) -> None:
        self.config = config
        self.store = store
        self.data_dir = config.data_dir
        self.pid_path = self.data_dir / "llama-server.pid"
        self.log_path = self.data_dir / "llama-server.log"
        self._proc: subprocess.Popen[bytes] | None = None
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=3.0),
            trust_env=False,
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --- состояние ---

    def _read_pid(self) -> int | None:
        try:
            raw = self.pid_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return int(raw) if raw.isdigit() else None

    def _write_pid(self, pid: int) -> None:
        self.pid_path.write_text(f"{pid}\n", encoding="utf-8")

    def _remove_pid(self) -> None:
        with contextlib.suppress(OSError):
            self.pid_path.unlink()

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _is_our_process(self, pid: int) -> bool:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False
        args = [part for part in raw.split(b"\x00") if part]
        has_binary = any(b"llama-server" in arg for arg in args)
        has_port = str(self.config.llm_port).encode() in args
        return has_binary and has_port

    async def health(self, timeout: float = 2.0) -> bool:
        try:
            response = await self._client.get(
                f"{self.config.llm_base_url}/health", timeout=timeout
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def status(self) -> ServerStatus:
        pid = self._read_pid()
        if pid is not None and not self._alive(pid):
            pid = None
            self._remove_pid()
        return ServerStatus(pid=pid, healthy=await self.health(), port=self.config.llm_port)

    async def current_model(self) -> Path | None:
        try:
            response = await self._client.get(f"{self.config.llm_base_url}/props", timeout=3.0)
            if response.status_code != 200:
                return None
            raw = response.json().get("model_path")
            return Path(raw).resolve() if raw else None
        except (httpx.HTTPError, ValueError, OSError):
            return None

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- параметры ---

    async def params(self, model_path: Path) -> ServerParams:
        values = self.store.snapshot()
        return ServerParams(
            binary=self.config.llama_binary,
            model=Path(model_path).resolve(),
            alias=_alias_for(Path(model_path)),
            host="127.0.0.1",
            port=self.config.llm_port,
            ctx_size=int(values["ctx_size"]),
            n_gpu_layers=int(values["n_gpu_layers"]),
            threads=int(values["threads"]),
            parallel=int(values["parallel"]),
            flash_attn=str(values["flash_attn"]),
            cache_type_k=str(values["cache_type_k"]),
            cache_type_v=str(values["cache_type_v"]),
            fit=str(values["fit"]),
            fit_target=int(values["fit_target"]),
            reasoning=str(values["reasoning"]),
        )

    # --- жизненный цикл ---

    async def ensure_running(self, model_path: Path) -> bool:
        """Переиспользует работающий сервер с той же моделью. True — был рестарт."""
        if await self.health():
            current = await self.current_model()
            if current is None or current == Path(model_path).resolve():
                return False
        await self.start(model_path)
        return True

    async def start(self, model_path: Path) -> None:
        await self.stop()
        params = await self.params(model_path)
        command = build_command(params)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        header = (
            f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"$ {shlex.join(command)}\n"
        )
        try:
            log = self.log_path.open("ab", buffering=0)
        except OSError as exc:
            raise ServerError(f"Не удалось открыть лог {self.log_path}: {exc}") from exc
        try:
            log.write(header.encode("utf-8"))
            try:
                proc = subprocess.Popen(  # noqa: S603 - команда собирается нами
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=str(PROJECT_ROOT),
                )
            except FileNotFoundError as exc:
                raise ServerError(f"Не найден бинарник {params.binary!r}") from exc
            except OSError as exc:
                raise ServerError(f"Не удалось запустить {params.binary!r}: {exc}") from exc
        finally:
            log.close()

        self._proc = proc
        self._write_pid(proc.pid)
        logger.info("llama-server запущен (pid %s): %s", proc.pid, params.model)

        deadline = time.monotonic() + self.config.start_timeout
        while time.monotonic() < deadline:
            code = proc.poll()
            if code is not None:
                self._remove_pid()
                self._proc = None
                raise ServerError(
                    f"llama-server завершился с кодом {code}.\n{self.log_tail(1800)}"
                )
            if await self.health(timeout=2.0):
                logger.info("llama-server готов (pid %s)", proc.pid)
                return
            await asyncio.sleep(0.5)

        self._remove_pid()
        self._proc = None
        with contextlib.suppress(OSError):
            proc.terminate()
        raise ServerError(
            f"llama-server не поднялся за {int(self.config.start_timeout)} с.\n"
            f"{self.log_tail(1800)}"
        )

    async def stop(self) -> None:
        pid = self._read_pid()
        if pid is not None and self._alive(pid) and self._is_our_process(pid):
            logger.info("Останавливаю llama-server (pid %s)", pid)
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + STOP_TIMEOUT
            while time.monotonic() < deadline and self._alive(pid):
                await asyncio.sleep(0.2)
            if self._alive(pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
        self._proc = None
        self._remove_pid()

    async def restart(self, model_path: Path) -> None:
        await self.start(model_path)

    def log_tail(self, chars: int = 3500) -> str:
        try:
            data = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "Лог пуст."
        return data[-chars:].strip() or "Лог пуст."
