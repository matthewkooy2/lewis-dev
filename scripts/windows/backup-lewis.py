from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(r"C:\Users\Example\OneDrive\Desktop\Lewis")
SOURCE = REPO / "data" / "lewis.sqlite3"
BACKUP_DIR = REPO / "backups"
RETENTION_DAYS = 14

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
destination = BACKUP_DIR / f"lewis_{timestamp}.sqlite3"

if not SOURCE.exists():
    raise SystemExit(f"Database not found: {SOURCE}")

with sqlite3.connect(SOURCE) as source_connection:
    with sqlite3.connect(destination) as backup_connection:
        source_connection.backup(backup_connection)

cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)

deleted = 0
for backup in BACKUP_DIR.glob("lewis_*.sqlite3"):
    modified = datetime.fromtimestamp(backup.stat().st_mtime)
    if modified < cutoff:
        backup.unlink()
        deleted += 1

print(f"Backup created: {destination}")
print(f"Old backups removed: {deleted}")
