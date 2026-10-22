#!/usr/bin/env python3
"""Initialise the SQLite database and print the resulting tables.

    python scripts/init_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daphnia.db import connection, init_db, table_names


def main() -> int:
    path = init_db()
    with connection(path) as conn:
        tables = table_names(conn)
    print(f"Initialised {path}")
    print("Tables: " + ", ".join(tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
