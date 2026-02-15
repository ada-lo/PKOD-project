"""
Data access layer for PKOD Supabase tables.

Every function is fail-safe: if the Supabase client is None (env vars missing)
or a network error occurs, we print a warning and return gracefully so that the
core parking system keeps running using the local JSON fallback.
"""

import time
from db.connection import get_client


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
    client = get_client()
    if client is None:
        return

    def _insert():
        client.table('vehicle_events').insert({
            'track_id': int(track_id),
            'event_type': event_type,
            'occupancy': int(occupancy),
            'timestamp': time.time(),
        }).execute()

    _safe('log_vehicle_event', _insert)


def get_recent_events(limit: int = 50):
    """Return the most recent vehicle events."""
    client = get_client()
    if client is None:
        return []

    def _query():
        resp = (client.table('vehicle_events')
                .select('*')
                .order('timestamp', desc=True)
                .limit(limit)
                .execute())
        return resp.data or []

    return _safe('get_recent_events', _query) or []


# ── ocr_results ──────────────────────────────────────────────────────

def log_ocr_result(track_id: int, plate_text: str, confidence: float,
                   event_type: str = None, image_path: str = None):
    """Insert a license plate OCR reading."""
    client = get_client()
    if client is None:
        return

    def _insert():
        row = {
            'track_id': int(track_id),
            'plate_text': plate_text,
            'confidence': float(confidence) if confidence is not None else None,
            'event_type': event_type,
            'image_path': image_path,
            'timestamp': time.time(),
        }
        client.table('ocr_results').insert(row).execute()

    _safe('log_ocr_result', _insert)


def get_ocr_results(track_id: int = None, limit: int = 50):
    """Query OCR results, optionally filtered by track_id."""
    client = get_client()
    if client is None:
        return []

    def _query():
        q = client.table('ocr_results').select('*')
        if track_id is not None:
            q = q.eq('track_id', int(track_id))
        resp = q.order('timestamp', desc=True).limit(limit).execute()
        return resp.data or []

    return _safe('get_ocr_results', _query) or []


# ── occupancy_snapshot ───────────────────────────────────────────────

def update_occupancy(occupancy: int, entry_count: int, exit_count: int,
                     max_capacity: int = 80, frozen: bool = False,
                     reason: str = None):
    """Upsert the single-row occupancy snapshot and append to audit_log."""
    client = get_client()
    if client is None:
        return

    now = time.time()

    def _upsert():
        client.table('occupancy_snapshot').upsert({
            'id': 1,
            'occupancy': int(occupancy),
            'entry_count': int(entry_count),
            'exit_count': int(exit_count),
            'max_capacity': int(max_capacity),
            'last_update': now,
            'frozen': bool(frozen),
        }).execute()

        # also append audit log
        client.table('audit_log').insert({
            'occupancy': int(occupancy),
            'reason': reason or 'event',
            'timestamp': now,
        }).execute()

    _safe('update_occupancy', _upsert)


def load_occupancy_from_db():
    """Load occupancy snapshot from Supabase.

    Returns (occupancy, entry_count, exit_count, last_update) or None if
    the database is unavailable.
    """
    client = get_client()
    if client is None:
        return None

    def _query():
        resp = (client.table('occupancy_snapshot')
                .select('*')
                .eq('id', 1)
                .execute())
        rows = resp.data or []
        if not rows:
            return None
        r = rows[0]
        return (
            int(r.get('occupancy', 0)),
            int(r.get('entry_count', 0)),
            int(r.get('exit_count', 0)),
            float(r.get('last_update', 0) or 0),
        )

    return _safe('load_occupancy_from_db', _query)


# ── vehicle_states ───────────────────────────────────────────────────

def save_vehicle_state(track_id: int, has_entered: bool, has_exited: bool):
    """Upsert a single vehicle state row."""
    client = get_client()
    if client is None:
        return

    def _upsert():
        client.table('vehicle_states').upsert({
            'track_id': int(track_id),
            'has_entered': bool(has_entered),
            'has_exited': bool(has_exited),
            'last_seen': time.time(),
        }).execute()

    _safe('save_vehicle_state', _upsert)


def save_vehicle_states_bulk(states: list):
    """Bulk upsert vehicle states. Each item should be a dict with
    keys: id, has_entered, has_exited."""
    client = get_client()
    if client is None:
        return

    def _bulk():
        rows = []
        for v in states:
            tid = v.get('id') if isinstance(v, dict) else getattr(v, 'id', None)
            if tid is None:
                continue
            rows.append({
                'track_id': int(tid),
                'has_entered': bool(v.get('has_entered', False) if isinstance(v, dict) else getattr(v, 'has_entered', False)),
                'has_exited': bool(v.get('has_exited', False) if isinstance(v, dict) else getattr(v, 'has_exited', False)),
                'last_seen': time.time(),
            })
        if rows:
            client.table('vehicle_states').upsert(rows).execute()

    _safe('save_vehicle_states_bulk', _bulk)


def load_vehicle_states_from_db():
    """Load all active vehicle states from Supabase.

    Returns list of dicts: [{'id': ..., 'has_entered': ..., 'has_exited': ...}]
    or None if database unavailable.
    """
    client = get_client()
    if client is None:
        return None

    def _query():
        resp = client.table('vehicle_states').select('*').execute()
        rows = resp.data or []
        return [
            {
                'id': r['track_id'],
                'has_entered': bool(r.get('has_entered', False)),
                'has_exited': bool(r.get('has_exited', False)),
            }
            for r in rows
        ]

    return _safe('load_vehicle_states_from_db', _query)


def clear_vehicle_states():
    """Delete all vehicle states (used on manual reset)."""
    client = get_client()
    if client is None:
        return

    def _delete():
        # Delete all rows — supabase-py requires a filter, use gt(track_id, -1)
        client.table('vehicle_states').delete().gte('track_id', 0).execute()

    _safe('clear_vehicle_states', _delete)
