"""Поиск GGUF-моделей и чтение их метаданных."""

from __future__ import annotations

import hashlib
import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_GGUF_MAGIC = b"GGUF"
_STRING = 8
_ARRAY = 9
_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_QUANT_RE = re.compile(
    r"(UD-)?("
    r"IQ\d+_[A-Z0-9_]+"
    r"|Q\d+_K(?:_[A-Z0-9]+)?"
    r"|Q\d+_\d+"
    r"|BF16|F16|F32|MXFP4"
    r")",
    re.IGNORECASE,
)
_PREFERRED = ("Qwen3-8B-Q6_K", "Qwen3.5-9B-Q4_K_M", "Qwen3-8B-Q4_K_M")
_FIT_BYTES = 7_400_000_000


@dataclass(frozen=True, slots=True)
class ModelInfo:
    path: Path
    name: str
    size_bytes: int
    quant: str
    context_length: int | None = None

    @property
    def id(self) -> str:
        return hashlib.sha1(str(self.path).encode("utf-8")).hexdigest()[:10]

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1_000_000_000

    @property
    def short_name(self) -> str:
        return self.path.stem


class _Reader:
    def __init__(self, file) -> None:  # noqa: ANN001 - BinaryIO
        self.file = file

    def read(self, size: int) -> bytes:
        data = self.file.read(size)
        if len(data) != size:
            raise EOFError("Неожиданный конец GGUF-файла")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        size = self.u64()
        if size > 16_000_000:
            raise ValueError("Слишком длинная строка в метаданных")
        return self.read(size).decode("utf-8", "replace")


def _skip_value(reader: _Reader, value_type: int) -> None:
    if value_type in _FIXED_SIZES:
        reader.read(_FIXED_SIZES[value_type])
        return
    if value_type == _STRING:
        reader.read(reader.u64())
        return
    if value_type == _ARRAY:
        element_type = reader.u32()
        count = reader.u64()
        if element_type == _STRING:
            for _ in range(count):
                reader.read(reader.u64())
        elif element_type in _FIXED_SIZES:
            reader.read(count * _FIXED_SIZES[element_type])
        else:
            raise ValueError(f"Неизвестный тип массива GGUF: {element_type}")
        return
    raise ValueError(f"Неизвестный тип GGUF: {value_type}")


def read_gguf_metadata(path: Path, max_keys: int = 64) -> dict[str, object]:
    """Читает начало метаданных GGUF, не загружая файл целиком."""
    meta: dict[str, object] = {}
    try:
        with path.open("rb") as file:
            if file.read(4) != _GGUF_MAGIC:
                return {}
            version = struct.unpack("<I", file.read(4))[0]
            if version < 2:
                return {}
            file.read(8)  # число тензоров
            kv_count = struct.unpack("<Q", file.read(8))[0]
            reader = _Reader(file)
            for index in range(min(kv_count, max_keys)):
                key = reader.string()
                value_type = reader.u32()
                if value_type in (_STRING, 4, 5, 10, 11, 7):
                    if value_type == _STRING:
                        meta[key] = reader.string()
                    elif value_type == 4:
                        meta[key] = reader.u32()
                    elif value_type == 7:
                        meta[key] = bool(reader.read(1)[0])
                    elif value_type == 10:
                        meta[key] = reader.u64()
                    elif value_type == 5:
                        meta[key] = struct.unpack("<i", reader.read(4))[0]
                    else:
                        meta[key] = struct.unpack("<q", reader.read(8))[0]
                else:
                    _skip_value(reader, value_type)
                if key.endswith(".context_length") and index >= 6:
                    break
    except (OSError, EOFError, ValueError, struct.error) as exc:
        logger.debug("Не удалось прочитать метаданные %s: %s", path, exc)
        return meta
    return meta


def _quant_from_name(stem: str) -> str:
    match = _QUANT_RE.search(stem)
    return match.group(0).upper() if match else "?"


def _context_from_meta(meta: dict[str, object]) -> int | None:
    for key, value in meta.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def load_model(path: Path) -> ModelInfo | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    meta = read_gguf_metadata(path)
    name = str(meta.get("general.name") or path.stem)
    return ModelInfo(
        path=path.resolve(),
        name=name,
        size_bytes=stat.st_size,
        quant=_quant_from_name(path.stem),
        context_length=_context_from_meta(meta),
    )


def scan_models(directories: tuple[Path, ...] | list[Path]) -> list[ModelInfo]:
    """Ищет *.gguf в каталогах (рекурсивно), без дублей."""
    seen: set[Path] = set()
    models: list[ModelInfo] = []
    for directory in directories:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.gguf"):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            info = load_model(resolved)
            if info is not None:
                models.append(info)
    models.sort(key=lambda item: (item.short_name.lower(), str(item.path)))
    return models


def choose_model(models: list[ModelInfo], saved_path: str | None) -> ModelInfo | None:
    """Сохранённая модель, иначе разумный дефолт под 8 ГБ VRAM."""
    if saved_path:
        try:
            resolved = Path(saved_path).expanduser().resolve()
        except OSError:
            resolved = None
        if resolved is not None:
            for model in models:
                if model.path == resolved:
                    return model
    if not models:
        return None
    for preferred in _PREFERRED:
        for model in models:
            if preferred.lower() in model.short_name.lower():
                return model
    fitting = [model for model in models if model.size_bytes <= _FIT_BYTES]
    if fitting:
        return fitting[0]
    return min(models, key=lambda item: item.size_bytes)
