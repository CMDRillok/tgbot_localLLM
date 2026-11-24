from __future__ import annotations

import pytest

from tgllm.specs import SPECS


def test_int_parse_and_limits() -> None:
    spec = SPECS["ctx_size"]
    assert spec.parse("8192") == 8192
    assert spec.parse("8 192") == 8192
    with pytest.raises(ValueError):
        spec.parse("100")
    with pytest.raises(ValueError):
        spec.parse("abc")
    with pytest.raises(ValueError):
        spec.parse("")


def test_float_accepts_comma() -> None:
    spec = SPECS["temperature"]
    assert spec.parse("0,7") == pytest.approx(0.7)
    with pytest.raises(ValueError):
        spec.parse("3")


def test_choice_is_case_insensitive() -> None:
    assert SPECS["flash_attn"].parse("ON") == "on"
    with pytest.raises(ValueError):
        SPECS["flash_attn"].parse("maybe")


def test_text_trimmed() -> None:
    spec = SPECS["system_prompt"]
    assert spec.parse("  привет  ") == "привет"
    assert spec.format("abcd") == "4 симв."


def test_restart_flags() -> None:
    assert SPECS["ctx_size"].restart is True
    assert SPECS["temperature"].restart is False
    assert SPECS["max_messages"].restart is False
