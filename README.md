# tgbot_localLLM

Telegram-бот для общения с локальной LLM через `llama-server` (llama.cpp).
Никаких облачных API: модель крутится на твоей видеокарте, история — в SQLite.

## Возможности

- 💬 Обычный чат со стримингом: текст появляется по мере генерации.
- 🧠 Выбор любой установленной GGUF-модели прямо в боте (владелец).
- ⚙️ Настройка llama.cpp: контекст, слои на GPU, потоки, flash-attn, KV-кэш,
  `--fit`, число слотов.
- 🎛 Параметры генерации: temperature, top-p, top-k, repeat-penalty,
  max-tokens, системный промпт, размышления Qwen3.
- 💬 Глубина истории: сколько последних сообщений отправлять модели и лимит токенов.
- 👥 Доступ по allowlist: владелец выдаёт доступ командами `/allow` и `/deny`.
- 🖥 Управление сервером из бота: статус, лог, запуск, остановка, перезапуск.
- 🔄 Автоподбор слоёв: `n-gpu-layers = 999` означает «пусть `--fit` решит сам».

## Требования

- Linux, Python 3.12+ (проверено на 3.14).
- `llama-server` в `PATH` (llama.cpp с CUDA/Vulkan).
- Одна или несколько `.gguf`-моделей.

## Установка

```bash
git clone git@github.com:CMDRillok/tgbot_localLLM.git
cd tgbot_localLLM
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
chmod 600 .env
```

Заполни `.env`:

```dotenv
BOT_TOKEN=123456:токен_от_BotFather
OWNER_ID=твой_telegram_id
MODELS_DIRS=/home/illok/Downloads:/home/illok/projects/tgbot_localLLM/models
LLAMA_PORT=9932
```

`.env` исключён из git. `MODELS_DIRS` — каталоги через двоеточие, поиск
рекурсивный. Модели не копируются: бот читает их по месту.

## Запуск

```bash
.venv/bin/tgllm
```

При старте бот:
1. сканирует `MODELS_DIRS`;
2. выбирает сохранённую модель (или `Qwen3-8B-Q6_K`, `Qwen3.5-9B-Q4_K_M`,
   а если их нет — первую подходящую по размеру);
3. запускает `llama-server` на `127.0.0.1:9932` и пишет владельцу, что готов.

Первый запуск большой модели занимает 15–90 секунд. Бот в это время уже
отвечает на команды и сообщает, что модель загружается.

Остановка — `Ctrl+C`. По умолчанию бот гасит и `llama-server`
(`STOP_SERVER_ON_EXIT=0`, если сервер должен остаться).

### Ручное управление сервером

```bash
./scripts/server.sh start|stop|restart|status|logs
.venv/bin/tgllm-server models     # список найденных GGUF
.venv/bin/tgllm-server logs -n 80
```

## Команды

Для всех пользователей:

| Команда | Действие |
| --- | --- |
| `/start` | приветствие и имя модели |
| `/help` | справка |
| `/new` | очистить свою историю |
| `/stop` | прервать генерацию |

Для владельца дополнительно:

| Команда | Действие |
| --- | --- |
| `/settings` | панель: модель, llama.cpp, генерация, контекст, сервер, доступ |
| `/model` | список моделей и переключение |
| `/server` | статус, лог, рестарт, стоп |
| `/allow <id>` | выдать доступ |
| `/deny <id>` | забрать доступ |
| `/cancel` | отменить ввод значения |

Пользователь без доступа получает сообщение со своим Telegram ID — его нужно
передать владельцу. Владелец добавляет ID через `/allow`.

## Настройки

Все параметры хранятся в таблице `settings` (`data/bot.sqlite3`). Дефолты лежат
в `config/defaults.yaml` и применяются только при первом запуске.

- **llama.cpp** (нужен перезапуск): `ctx-size`, `n-gpu-layers`, `threads`,
  `parallel`, `fit`, `fit-target`, `flash-attn`, `cache-type-k/v`.
- **Генерация** (сразу): `temperature`, `top-p`, `top-k`, `repeat-penalty`,
  `max-tokens`, `system prompt`, `reasoning`.
- **Контекст** (сразу): `max_messages`, `history_max_tokens`.
  История хранится целиком, но в модель уходит только хвост.

Переключение модели и изменение параметров сервера перезапускают
`llama-server`. Если в этот момент генерируется ответ, бот сначала дождётся
его окончания.

## Архитектура

```
src/tgllm/
├── config.py       .env и defaults.yaml
├── db.py           SQLite: users, allowlist, messages, settings
├── store.py        настройки поверх SQLite
├── models.py       поиск GGUF и чтение заголовка (квант, ctx)
├── server.py       процесс llama-server, health, лог, PID
├── llm.py          streaming chat completions + фильтр  thinking
├── history.py      оценка токенов и обрезка истории
├── render.py       правки сообщений Telegram без флуда
├── specs.py        описание настраиваемых параметров
└── telegram/       bot, доступ, чат, панель владельца
```

`llama-server` работает на `127.0.0.1`, наружу порт не выставлен. Бот
обращается к OpenAI-совместимому `/v1/chat/completions` с `stream: true`.

## Автозапуск (systemd user unit)

`~/.config/systemd/user/tgllm.service`:

```ini
[Unit]
Description=Telegram bot with local LLM
After=network-online.target

[Service]
WorkingDirectory=/home/illok/projects/tgbot_localLLM
ExecStart=/home/illok/projects/tgbot_localLLM/.venv/bin/tgllm
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now tgllm.service
journalctl --user -u tgllm -f
```

## Если что-то не работает

- **Telegram отвечает `429 Too Many Requests` на `getMe`/`getUpdates`** — токен
  используется в другом месте (второй инстанс бота) или бот попал под
  анти-флуд. Останови второй инстанс, перевыпусти токен через @BotFather
  (`/revoke`) и запусти бота снова. Проверить токен:
  `curl "https://api.telegram.org/bot<TOKEN>/getMe"`.
- **«Не найдено ни одной .gguf-модели»** — проверь `MODELS_DIRS` и права на
  файлы; список можно обновить кнопкой в `/model`.
- **Не поднимается сервер** — `/settings → Сервер → Лог`. Частые причины:
  не хватает VRAM (уменьши `ctx-size`, поставь `n-gpu-layers` меньше или
  `fit-target` больше), занят порт.
- **Ответы медленные** — проверь `nvidia-smi`; при нехватке VRAM модель
  частично уходит на CPU. Уменьши контекст или смени модель на Q4.
- **Порт занят** — задай другой `LLAMA_PORT` в `.env`.
- **Бот молчит** — смотри `data/bot.log` и `journalctl --user -u tgllm`.

## Тесты

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```
