"""Build a SQLite copy of the Senetra database from the PostgreSQL seed files.

Translates seeds/DDL.sql and seeds/DML.sql to SQLite and loads them into a
fresh database file (any existing file at the output path is replaced).

Usage:
    python seeds/build_sqlite.py [output_path]    # default: backend/senetra.db
"""
import math
import random
import re
import sqlite3
import sys
from pathlib import Path

SEEDS_DIR = Path(__file__).resolve().parent
DEFAULT_DB = SEEDS_DIR.parent / "backend" / "senetra.db"

NOW = "datetime('now', 'localtime')"


def sub(pattern, repl, sql, flags=0):
    """re.subn that fails loudly when the pattern no longer matches the seeds."""
    sql, count = re.subn(pattern, repl, sql, flags=flags)
    if count == 0:
        raise ValueError(f"Translation pattern not found in seeds: {pattern!r}")
    return sql


def days_ago(days):
    return f"datetime('now', 'localtime', '-{days} days')"


def date_series(days_back):
    """Equivalent of generate_series(CURRENT_DATE - N days, CURRENT_DATE, 1 day)."""
    return (
        "(WITH RECURSIVE s(dt) AS ("
        f"SELECT date('now', 'localtime', '-{days_back} days') "
        "UNION ALL SELECT date(dt, '+1 day') FROM s "
        "WHERE dt < date('now', 'localtime')"
        ") SELECT dt FROM s)"
    )


def split_alter_columns(match):
    """SQLite allows only one ADD COLUMN per ALTER TABLE statement."""
    table, columns = match.group(1), match.group(2)
    return "\n".join(
        f"ALTER TABLE {table} ADD COLUMN {column.strip()};"
        for column in re.split(r",\s*ADD COLUMN IF NOT EXISTS", columns)
    )


def translate_common(sql):
    sql = re.sub(r"\bsenetra\.", "", sql)
    sql = re.sub(r"CREATE SCHEMA[^;]*;|SET search_path[^;]*;", "", sql)
    sql = re.sub(r"\bBIGSERIAL PRIMARY KEY\b", "INTEGER PRIMARY KEY AUTOINCREMENT", sql)
    return sql


def translate_ddl(sql):
    sql = translate_common(sql)

    enums = re.findall(r"CREATE TYPE (\w+) AS ENUM \((.*?)\);", sql, re.S)
    if not enums:
        raise ValueError("No enum types found in DDL")
    sql = sub(r"CREATE TYPE \w+ AS ENUM \(.*?\);", "", sql, re.S)

    for name, values in enums:
        allowed = ", ".join(v.strip() for v in values.split(","))
        sql = sub(
            rf"^(\s+)(\w+) {name}\b",
            lambda m, allowed=allowed: f"{m[1]}{m[2]} TEXT CHECK ({m[2]} IN ({allowed}))",
            sql,
            re.M,
        )
    return sql


def translate_dml(sql):
    sql = translate_common(sql)

    # NOW() - ((random() * 30)::int * INTERVAL '1 day')
    sql = sub(
        r"NOW\(\) - \(\(random\(\) \* (\d+)\)::int \* INTERVAL '1 day'\)",
        r"datetime('now', 'localtime', '-' || CAST(round(random() * \1) AS INTEGER) || ' days')",
        sql,
    )
    sql = sub(r"NOW\(\) - INTERVAL '(\d+) days'", lambda m: days_ago(m[1]), sql)
    sql = sub(
        r"generate_series\(\s*CURRENT_DATE - INTERVAL '(\d+) days',\s*CURRENT_DATE,\s*"
        r"INTERVAL '1 day'\s*\) AS dt",
        lambda m: date_series(m[1]),
        sql,
    )
    sql = sub(r"CURRENT_DATE - (\d+)", r"date('now', 'localtime', '-\1 days')", sql)
    sql = sub(r"::(numeric|int)\b", "", sql)
    sql = sub(r"\bGREATEST\(", "max(", sql)
    sql = sub(r"ALTER TABLE (\w+)\s+ADD COLUMN IF NOT EXISTS (.*?);", split_alter_columns, sql, re.S)

    # SQLite has no LATERAL joins. MATERIALIZED keeps each row's random draws
    # fixed, so closing_stock is computed from the same values that are stored.
    sql = sub(
        r"SELECT\s+p\.id,\s+m\.id,\s+dt,\s+opening_stock,\s+consumption,\s+"
        r"opening_stock - consumption\s+FROM (?P<source>.*?)\s+"
        r"CROSS JOIN LATERAL\s*\(\s*SELECT(?P<columns>.*?)\) x;",
        lambda m: (
            "WITH x AS MATERIALIZED (\n"
            f"SELECT p.id AS phc_id, m.id AS medicine_id, dt, {m['columns']}\n"
            f"FROM {m['source']}\n)\n"
            "SELECT phc_id, medicine_id, dt, opening_stock, consumption, "
            "opening_stock - consumption\nFROM x;"
        ),
        sql,
        re.S,
    )

    # SQLite's random() returns a 64-bit integer; the seeds expect [0, 1).
    sql = sub(r"\brandom\(\)", "random_unit()", sql)
    return sql


def build(db_path):
    ddl = translate_ddl((SEEDS_DIR / "DDL.sql").read_text(encoding="utf-8"))
    dml = translate_dml((SEEDS_DIR / "DML.sql").read_text(encoding="utf-8"))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.create_function("random_unit", 0, random.random)
        conn.create_function(
            "floor", 1, lambda x: None if x is None else math.floor(x), deterministic=True
        )
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(f"BEGIN;\n{ddl}\n{dml}\nCOMMIT;")
        return conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall(), conn
    except Exception:
        conn.close()
        raise


def main():
    db_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DB
    tables, conn = build(db_path)
    with conn:
        print(f"Built {db_path}")
        for (table,) in tables:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            print(f"  {table:<32} {count:>9,}")
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"foreign_key_check violations: {len(fk_errors)}; integrity_check: {integrity}")
    conn.close()


if __name__ == "__main__":
    main()
