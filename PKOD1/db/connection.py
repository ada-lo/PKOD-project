"""
PostgreSQL connection manager for PKOD (Neon-compatible).

Provides a singleton psycopg2 connection initialised from DATABASE_URL in .env.
Works with both Neon (cloud) and local PostgreSQL instances.
"""

import os
from dotenv import load_dotenv

# Load .env from project root
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(_project_root, '.env'))

_conn = None


def get_conn():
    """Return a lazily-initialised psycopg2 connection (singleton)."""
    global _conn
    if _conn is not None:
        try:
            # Check if connection is still alive
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("[DB] WARNING: DATABASE_URL not set in .env — database features disabled")
        return None

    try:
        import psycopg2
        _conn = psycopg2.connect(db_url, sslmode='require')
        _conn.autocommit = True
        print("[DB] PostgreSQL connected successfully")
        return _conn
    except Exception as e:
        print(f"[DB] Failed to connect to PostgreSQL: {e}")
        return None


def init_db():
    """Create tables if they don't exist (runs schema.sql)."""
    conn = get_conn()
    if conn is None:
        return False

    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    if not os.path.exists(schema_path):
        print("[DB] schema.sql not found — skipping table creation")
        return False

    try:
        with open(schema_path, 'r') as f:
            sql = f.read()
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        print("[DB] Schema initialized successfully")
        return True
    except Exception as e:
        print(f"[DB] Schema initialization error: {e}")
        return False
