from __future__ import annotations

from pathlib import Path

from tgllm.server import ServerParams, build_command


def _params(**overrides: object) -> ServerParams:
    values = {
        "binary": "llama-server",
        "model": Path("/models/Qwen3-8B-Q6_K.gguf"),
        "alias": "Qwen3-8B-Q6_K",
        "host": "127.0.0.1",
        "port": 9932,
        "ctx_size": 8192,
        "n_gpu_layers": 999,
        "threads": 8,
        "parallel": 1,
        "flash_attn": "on",
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "fit": "on",
        "fit_target": 256,
        "reasoning": "off",
    }
    values.update(overrides)
    return ServerParams(**values)  # type: ignore[arg-type]


def test_auto_layers_let_fit_decide() -> None:
    command = build_command(_params())
    assert "--n-gpu-layers" not in command
    assert "--fit" in command
    assert command[command.index("--ctx-size") + 1] == "8192"


def test_explicit_layers() -> None:
    command = build_command(_params(n_gpu_layers=20))
    assert command[command.index("--n-gpu-layers") + 1] == "20"


def test_fit_off_sets_layers() -> None:
    command = build_command(_params(fit="off"))
    assert command[command.index("--n-gpu-layers") + 1] == "99"


def test_required_flags() -> None:
    command = build_command(_params())
    for flag in (
        "--model",
        "--alias",
        "--port",
        "--ctx-size",
        "--flash-attn",
        "--cache-type-k",
        "--cache-type-v",
        "--reasoning",
        "--jinja",
        "--no-webui",
    ):
        assert flag in command
