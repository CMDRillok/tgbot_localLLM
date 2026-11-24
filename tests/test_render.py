from __future__ import annotations

from tgllm.render import split_text


def test_short_text_single_part() -> None:
    assert split_text("привет") == ["привет"]
    assert split_text("   ") == []


def test_long_text_is_split() -> None:
    text = ("строка текста\n" * 1000).strip()
    parts = split_text(text, limit=500)
    assert len(parts) > 1
    assert all(len(part) <= 500 for part in parts)
    assert "".join(part.replace("\n", "") for part in parts) == text.replace("\n", "")


def test_hard_split_without_spaces() -> None:
    text = "x" * 1001
    parts = split_text(text, limit=500)
    assert "".join(parts) == text
    assert all(len(part) <= 500 for part in parts)
