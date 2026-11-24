"""SQLite: пользователи, allowlist, история сообщений и настройки."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    user_id    INTEGER PRIMARY KEY,
    username   TEXT,
    first_name TEXT,
    created_at REAL NOT NULL,
    last_seen  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS allowlist (
    user_id  INTEGER PRIMARY KEY,
    added_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("База данных не открыта")
        return self._conn

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # --- пользователи и доступ ---

    async def touch_user(self, user_id: int, username: str | None, first_name: str | None) -> None:
        now = time.time()
        await self.conn.execute(
            """
            INSERT INTO users (user_id, username, first_name, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name,
                last_seen = excluded.last_seen
            """,
            (user_id, username, first_name, now, now),
        )
        await self.conn.commit()

    async def is_allowed(self, user_id: int) -> bool:
        cursor = await self.conn.execute("SELECT 1 FROM allowlist WHERE user_id = ?", (user_id,))
        return await cursor.fetchone() is not None

    async def allow(self, user_id: int) -> None:
        await self.conn.execute(
            "INSERT OR IGNORE INTO allowlist (user_id, added_at) VALUES (?, ?)",
            (user_id, time.time()),
        )
        await self.conn.commit()

    async def deny(self, user_id: int) -> None:
        await self.conn.execute("DELETE FROM allowlist WHERE user_id = ?", (user_id,))
        await self.conn.commit()

    async def allowed_users(self) -> list[tuple[int, str | None]]:
        cursor = await self.conn.execute(
            """
            SELECT a.user_id, u.username
            FROM allowlist AS a
            LEFT JOIN users AS u ON u.user_id = a.user_id
            ORDER BY a.added_at
            """
        )
        return [(row[0], row[1]) for row in await cursor.fetchall()]

    # --- история ---

    async def add_message(self, user_id: int, role: str, content: str) -> None:
        await self.conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, time.time()),
        )
        await self.conn.commit()

    async def recent_messages(self, user_id: int, limit: int) -> list[tuple[str, str]]:
        cursor = await self.conn.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (user_id, max(1, limit)),
        )
        return [(row[0], row[1]) for row in await cursor.fetchall()]

    async def clear_messages(self, user_id: int) -> int:
        cursor = await self.conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await self.conn.commit()
        return cursor.rowcount

    # --- настройки ---

    async def load_settings(self) -> dict[str, Any]:
        cursor = await self.conn.execute("SELECT key, value FROM settings")
        result: dict[str, Any] = {}
        for key, value in await cursor.fetchall():
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = value
        return result

    async def save_setting(self, key: str, value: Any) -> None:
        await self.conn.execute(
            """
            INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, json.dumps(value, ensure_ascii=False)),
        )
        await self.conn.commit()
