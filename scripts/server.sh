#!/usr/bin/env bash
# Управление llama-server вручную: ./scripts/server.sh start|stop|restart|status|logs
# Обычно это делает сам бот; скрипт нужен для отладки.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

exec "$PYTHON" "$DIR/src/tgllm/cli.py" "$@"
