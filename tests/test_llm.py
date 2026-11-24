from __future__ import annotations

import httpx
import pytest

from tgllm.llm import LLMClient, LLMError, ThinkingFilter


def test_thinking_filter_basic() -> None:
    filter_ = ThinkingFilter()
    out = filter_.feed("Привет ") + filter_.flush()
    assert out == "Привет "


def test_thinking_filter_split_tags() -> None:
    filter_ = ThinkingFilter()
    out = filter_.feed("Ответ: <thi")
    out += filter_.feed("nk>секрет</thi")
    out += filter_.feed("nk> мир")
    out += filter_.flush()
    assert out == "Ответ:  мир"


def test_thinking_filter_alt_close_tag() -> None:
    filter_ = ThinkingFilter()
    out = filter_.feed("A <think>secret<｜end▁of▁thinking｜> B")
    out += filter_.flush()
    assert out == "A  B"


def _sse(*contents: str) -> bytes:
    lines = []
    for content in contents:
        lines.append(f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n')
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _client(handler) -> LLMClient:  # noqa: ANN001
    return LLMClient("http://llama.test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_stream_chat_collects_deltas() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            content=_sse("При", "вет", "!"),
            headers={"content-type": "text/event-stream"},
        )

    client = _client(handler)
    try:
        chunks = [
            chunk
            async for chunk in client.stream_chat(
                [{"role": "user", "content": "привет"}],
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.05,
                max_tokens=128,
            )
        ]
    finally:
        await client.aclose()
    assert "".join(chunks) == "Привет!"
    assert seen["path"] == "/v1/chat/completions"


@pytest.mark.asyncio
async def test_stream_chat_http_error() -> None:
    client = _client(lambda request: httpx.Response(500, text="boom"))
    try:
        with pytest.raises(LLMError):
            async for _ in client.stream_chat(
                [{"role": "user", "content": "x"}],
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.0,
                max_tokens=8,
            ):
                pass
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_reasoning_on_shows_raw_tags() -> None:
    client = _client(
        lambda request: httpx.Response(
            200,
            content=_sse("<thi", "nk>мысль</thi", "nk> ответ"),
            headers={"content-type": "text/event-stream"},
        )
    )
    try:
        chunks = [
            chunk
            async for chunk in client.stream_chat(
                [{"role": "user", "content": "x"}],
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.0,
                max_tokens=64,
                reasoning=True,
            )
        ]
    finally:
        await client.aclose()
    assert "мысль" in "".join(chunks)


@pytest.mark.asyncio
async def test_health() -> None:
    client = _client(lambda request: httpx.Response(200, json={"status": "ok"}))
    try:
        assert await client.health() is True
    finally:
        await client.aclose()
