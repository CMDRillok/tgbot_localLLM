from __future__ import annotations

import struct
from pathlib import Path

from tgllm.models import choose_model, load_model, read_gguf_metadata, scan_models


def _s(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(entries: list[tuple[str, object]]) -> bytes:
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 0)  # tensors
    out += struct.pack("<Q", len(entries))
    for key, value in entries:
        out += _s(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + _s(value)
        elif isinstance(value, int):
            out += struct.pack("<I", 10) + struct.pack("<Q", value)
        elif isinstance(value, list):
            out += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value))
            for item in value:
                out += _s(str(item))
        else:  # pragma: no cover - защита от ошибки в тесте
            raise TypeError(value)
    return bytes(out)


def test_read_metadata_and_context(tmp_path: Path) -> None:
    path = tmp_path / "Qwen3-8B-Q6_K.gguf"
    path.write_bytes(
        _gguf(
            [
                ("general.architecture", "qwen3"),
                ("general.name", "Qwen3 Test"),
                ("qwen3.block_count", 40),
                ("tokenizer.ggml.tokens", ["a", "b"]),
                ("qwen3.embedding_length", 5120),
                ("qwen3.attention.head_count", 40),
                ("qwen3.feed_forward_length", 13824),
                ("qwen3.context_length", 40960),
                ("tokenizer.ggml.merges", ["x y"]),
            ]
        )
    )

    meta = read_gguf_metadata(path)
    assert meta["general.name"] == "Qwen3 Test"
    assert meta["qwen3.context_length"] == 40960

    model = load_model(path)
    assert model is not None
    assert model.quant == "Q6_K"
    assert model.context_length == 40960
    assert model.size_gb > 0


def test_scan_deduplicates_and_skips_bad(tmp_path: Path) -> None:
    good = tmp_path / "Qwen3.5-9B-Q4_K_M.gguf"
    good.write_bytes(_gguf([("general.name", "Qwen3.5")]))
    bad = tmp_path / "broken.gguf"
    bad.write_bytes(b"not a gguf")
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "gemma-3-1b-it-Q4_K_M.gguf"
    second.write_bytes(_gguf([("general.name", "Gemma")]))
    third = nested / "Qwen3.5-9B-UD-Q5_K_XL.gguf"
    third.write_bytes(_gguf([("general.name", "Qwen3.5 UD")]))

    models = scan_models([tmp_path, tmp_path])
    names = {model.short_name for model in models}
    assert names == {
        "Qwen3.5-9B-Q4_K_M",
        "Qwen3.5-9B-UD-Q5_K_XL",
        "broken",
        "gemma-3-1b-it-Q4_K_M",
    }
    broken = next(model for model in models if model.short_name == "broken")
    assert broken.context_length is None
    ud = next(model for model in models if "UD-" in model.short_name)
    assert ud.quant == "UD-Q5_K_XL"


def test_choose_model(tmp_path: Path) -> None:
    first = load_model(_write(tmp_path, "gemma-3-1b-it-Q4_K_M.gguf"))
    preferred = load_model(_write(tmp_path, "Qwen3-8B-Q6_K.gguf"))
    bigger = load_model(_write(tmp_path, "Qwen3-14B-Q6_K.gguf"))
    assert first and preferred and bigger
    models = [first, preferred, bigger]

    assert choose_model(models, None) is preferred
    assert choose_model(models, str(first.path)) is first
    assert choose_model(models, "/nope/missing.gguf") is preferred
    assert choose_model([], None) is None


def _write(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(_gguf([("general.name", name)]))
    return path
