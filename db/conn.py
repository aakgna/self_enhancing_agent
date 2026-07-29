"""Thread-local Postgres connections.

One autocommit connection per thread (the Phase 5 regression runner runs in a
background thread and must not share a connection with the REPL thread).
"""

import os
import threading

import psycopg
from dotenv import load_dotenv

load_dotenv()

_local = threading.local()


def get_conn() -> psycopg.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed:
        conn = psycopg.connect(
            os.environ.get("DATABASE_URL", "postgresql:///enhanced_agent"),
            autocommit=True,
        )
        _local.conn = conn
    return conn
