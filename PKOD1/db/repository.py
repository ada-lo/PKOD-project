"""
Data access layer for PKOD PostgreSQL tables (Neon-compatible).

Every function is fail-safe: if the DB connection is None or a query
fails, we print a warning and return gracefully so that the core
parking system keeps running using the local JSON fallback.
"""

import time
from db.connection import get_conn


# ── helpers ──────────────────────────────────────────────────────────

def _safe(fn_name, fn, *args, **kwargs):
    """Call *fn* inside a try/except, printing errors but never crashing."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[DB] {fn_name} error: {e}")
        return None


# ── vehicle_events ───────────────────────────────────────────────────

def log_vehicle_event(track_id: int, event_type: str, occupancy: int):
    """Insert an entry/exit event."""
    conn = get_conn()
    if conn is None:
        return

    def _insert():
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO vehicle_events (track_id, event_type, occupancy, timestamp) "
            "VALUES (%s, %s, %s, %s)",
            (int(track_id), event_type, int(occupancy), time.time())
        )
        cur.close()

    _safe('log_vehicle_event', _insert)


def get_recent_events(limit: int = 50):
    """Return the most recent vehicle events."""
    conn = get_conn()
    if conn is None:
        return []

    def _query():
        cur = conn.cursor()
        cur.execute(
            "SELECT id, track_id, event_type, occupancy, timestamp, created_at "
            "FROM vehicle_events ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
        return rows

    return _safe('get_recent_events', _query) or []


# ── ocr_results ──────────────────────────────────────────────────────

def log_ocr_result(track_id: int, plate_text: str, confidence: float,
                   event_type: str = None, image_path: str = None):
    """Insert a license plate OCR reading."""
    conn = get_conn()
    if conn is None:
        return

    def _insert():
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ocr_results (track_id, plate_text, confidence, event_type, image_path, timestamp) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (int(track_id), plate_text,
             float(confidence) if confidence is not None else None,
             event_type, image_path, time.time())
        )
        cur.close()

    _safe('log_ocr_result', _insert)


def get_ocr_results(track_id: int = None, limit: int = 50):
    """Query OCR results, optionally filtered by track_id."""
    conn = get_conn()
    if conn is None:
        return []

    def _query():
        cur = conn.cursor()
        if track_id is not None:
            cur.execute(
                "SELECT * FROM ocr_results WHERE track_id = %s "
                "ORDER BY timestamp DESC LIMIT %s",
                (int(track_id), limit)
            )
        else:
            cur.execute(
                "SELECT * FROM ocr_results ORDER BY timestamp DESC LIMIT %s",
                (limit,)
            )
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
        return rows

    return _safe('get_ocr_results', _query) or []


# ── occupancy_snapshot ───────────────────────────────────────────────

def update_occupancy(occupancy: int, entry_count: int, exit_count: int,
                     max_capacity: int = 80, frozen: bool = False,
                     reason: str = None):
    """Upsert the single-row occupancy snapshot and append to audit_log."""
    conn = get_conn()
    if conn is None:
        return

    now = time.time()

    def _upsert():
        cur = conn.cursor()
        # Upsert occupancy snapshot
        cur.execute("""
            INSERT INTO occupancy_snapshot (id, occupancy, entry_count, exit_count, max_capacity, last_update, frozen)
            VALUES (1, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                occupancy = EXCLUDED.occupancy,
                entry_count = EXCLUDED.entry_count,
                exit_count = EXCLUDED.exit_count,
                max_capacity = EXCLUDED.max_capacity,
                last_update = EXCLUDED.last_update,
                frozen = EXCLUDED.frozen
        """, (int(occupancy), int(entry_count), int(exit_count),
              int(max_capacity), now, bool(frozen)))

        # Append audit log
        cur.execute(
            "INSERT INTO audit_log (occupancy, reason, timestamp) VALUES (%s, %s, %s)",
            (int(occupancy), reason or 'event', now)
        )
        cur.close()

    _safe('update_occupancy', _upsert)


def load_occupancy_from_db():
    """Load occupancy snapshot from PostgreSQL.

    Returns (occupancy, entry_count, exit_count, last_update) or None.
    """
    conn = get_conn()
    if conn is None:
        return None

    def _query():
        cur = conn.cursor()
        cur.execute("SELECT occupancy, entry_count, exit_count, last_update FROM occupancy_snapshot WHERE id = 1")
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return (int(row[0]), int(row[1]), int(row[2]), float(row[3] or 0))

    return _safe('load_occupancy_from_db', _query)


# ── vehicle_states ───────────────────────────────────────────────────

def save_vehicle_state(track_id: int, has_entered: bool, has_exited: bool):
    """Upsert a single vehicle state row."""
    conn = get_conn()
    if conn is None:
        return

    def _upsert():
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO vehicle_states (track_id, has_entered, has_exited, last_seen)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (track_id) DO UPDATE SET
                has_entered = EXCLUDED.has_entered,
                has_exited = EXCLUDED.has_exited,
                last_seen = EXCLUDED.last_seen
        """, (int(track_id), bool(has_entered), bool(has_exited), time.time()))
        cur.close()

    _safe('save_vehicle_state', _upsert)


def save_vehicle_states_bulk(states: list):
    """Bulk upsert vehicle states."""
    conn = get_conn()
    if conn is None:
        return

    def _bulk():
        cur = conn.cursor()
        now = time.time()
        for v in states:
            tid = v.get('id') if isinstance(v, dict) else getattr(v, 'id', None)
            if tid is None:
                continue
            he = bool(v.get('has_entered', False) if isinstance(v, dict) else getattr(v, 'has_entered', False))
            hx = bool(v.get('has_exited', False) if isinstance(v, dict) else getattr(v, 'has_exited', False))
            cur.execute("""
                INSERT INTO vehicle_states (track_id, has_entered, has_exited, last_seen)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (track_id) DO UPDATE SET
                    has_entered = EXCLUDED.has_entered,
                    has_exited = EXCLUDED.has_exited,
                    last_seen = EXCLUDED.last_seen
            """, (int(tid), he, hx, now))
        cur.close()

    _safe('save_vehicle_states_bulk', _bulk)


def load_vehicle_states_from_db():
    """Load all active vehicle states from PostgreSQL.

    Returns list of dicts or None if database unavailable.
    """
    conn = get_conn()
    if conn is None:
        return None

    def _query():
        cur = conn.cursor()
        cur.execute("SELECT track_id, has_entered, has_exited FROM vehicle_states")
        rows = [
            {'id': r[0], 'has_entered': bool(r[1]), 'has_exited': bool(r[2])}
            for r in cur.fetchall()
        ]
        cur.close()
        return rows

    return _safe('load_vehicle_states_from_db', _query)


def clear_vehicle_states():
    """Delete all vehicle states (used on manual reset)."""
    conn = get_conn()
    if conn is None:
        return

    def _delete():
        cur = conn.cursor()
        cur.execute("DELETE FROM vehicle_states")
        cur.close()

    _safe('clear_vehicle_states', _delete)
