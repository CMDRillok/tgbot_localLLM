"""Клиент OpenAI-совместимого API llama-server со стримингом."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Модель недоступна или вернула ошибку."""


class ThinkingFilter:
    """Убирает блок размышлений, даже если теги разорваны между чанками."""

    OPEN = "<think>"
    CLOSE_TAGS = ("</think>", "<｜end▁of▁thinking｜>")
    _KEEP = max(len(tag) for tag in CLOSE_TAGS) - 1

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def _find_close(self) -> tuple[int, str] | None:
        hits = [
            (self._buffer.find(tag), tag)
            for tag in self.CLOSE_TAGS
            if self._buffer.find(tag) != -1
        ]
        return min(hits) if hits else None

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        output: list[str] = []
        while True:
            if not self._inside:
                index = self._buffer.find(self.OPEN)
                if index == -1:
                    keep = len(self.OPEN) - 1
                    if len(self._buffer) > keep:
                        output.append(self._buffer[:-keep])
                        self._buffer = self._buffer[-keep:]
                    break
                output.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(self.OPEN) :]
                self._inside = True
            else:
                hit = self._find_close()
                if hit is None:
                    if len(self._buffer) > self._KEEP:
                        self._buffer = self._buffer[-self._KEEP :]
                    break
                index, tag = hit
                self._buffer = self._buffer[index + len(tag) :]
                self._inside = False
        return "".join(output)

    def flush(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        output, self._buffer = self._buffer, ""
        return output


class LLMClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self, timeout: float = 2.0) -> bool:
        try:
            response = await self._client.get("/health", timeout=timeout)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        max_tokens: int,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": "local",
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "max_tokens": max_tokens,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": reasoning},
        }
        thinking = ThinkingFilter()
        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise LLMError(
                        f"llama-server вернул {response.status_code}: {body[:500]}"
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Некорректный чанк SSE: %r", data[:200])
                        continue
                    if isinstance(chunk, dict) and chunk.get("error"):
                        raise LLMError(str(chunk["error"])[:500])
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        visible = thinking.feed(content) if not reasoning else content
                        if visible:
                            yield visible
        except httpx.HTTPError as exc:
            raise LLMError(
                "Нет связи с llama-server. Проверь модель в /settings → Сервер."
            ) from exc
        if not reasoning:
            tail = thinking.flush()
            if tail:
                yield tail
