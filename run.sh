#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

case "${1:-web}" in
  download)
    echo "Downloading dataset..."
    python3 download_dataset.py
    ;;
  train)
    echo "Training model..."
    python3 train_model.py
    ;;
  web)
    echo "Starting web server..."
    python3 web/app.py
    ;;
  bot)
    echo "Starting Telegram bot..."
    python3 bot/bot.py
    ;;
  all)
    echo "Starting web server + bot..."
    python3 web/app.py &
    WEB_PID=$!
    python3 bot/bot.py &
    BOT_PID=$!
    echo "Web: $WEB_PID  Bot: $BOT_PID"
    wait
    ;;
  *)
    echo "Usage: $0 {download|train|web|bot|all}"
    exit 1
    ;;
esac
