# -*- coding: utf-8 -*-
"""
Парсинг сообщений из Telegram-каналов транспортной тематики.
Сохраняем результат в CSV-файлы в data/raw/
"""
import os
import csv
import asyncio
from telethon import TelegramClient

# === ВАЖНО ===
# Перед запуском установи переменные окружения:
# export TELEGRAM_API_ID=...
# export TELEGRAM_API_HASH=...

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

CHANNELS = {
    "автотранспорт": ["DtRoad", "avtodorgk", "Mintrans_Russia"],
    "метро": ["mosmetro", "spbmetro", "stroimmetro"],
}

OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

async def dump_channel(client, username, category, limit=1000):
    """Выгружаем последние сообщения из канала"""
    entity = await client.get_entity(username)
    messages = []
    async for m in client.iter_messages(entity, limit=limit):
        if m.message:
            messages.append([
                category,
                username,
                m.id,
                m.date.isoformat(),
                m.message.replace("\n", " "),
            ])
    return messages

async def main():
    async with TelegramClient("session", API_ID, API_HASH) as client:
        for cat, chans in CHANNELS.items():
            all_msgs = []
            for ch in chans:
                username = f"@{ch}"
                msgs = await dump_channel(client, username, cat)
                all_msgs.extend(msgs)
                print(f"{username}: {len(msgs)} сообщений")
            out_path = os.path.join(OUT_DIR, f"{cat}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["category", "channel", "msg_id", "date", "text"])
                writer.writerows(all_msgs)
            print(f"Сохранено: {out_path} ({len(all_msgs)} строк)")

if __name__ == "__main__":
    asyncio.run(main())
