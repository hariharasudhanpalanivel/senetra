"""SQLite connections. The pipeline reads the operational database in place and never exports it."""

from __future__ import annotations

import hashlib
import sqlite3
from functools import lru_cache
from pathlib import Path


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def connect_writable(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(Path(db_path).resolve())


def database_fingerprint(db_path: str | Path) -> str:
    """Content hash used as data lineage in MLflow (identifies the data without copying it)."""
    path = Path(db_path).resolve()
    stat = path.stat()
    return _fingerprint(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _fingerprint(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
