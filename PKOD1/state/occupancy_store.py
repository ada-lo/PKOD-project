import json
import os
import time
import config

# Supabase integration (fail-safe: falls back to JSON if unavailable)
try:
    from db import repository as db
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

# In-memory cache for vehicle states. Load once at startup and use as runtime source-of-truth.
# Disk is only used for checkpointing/recovery.
_vehicle_states_cache = None

def load_occupancy():
    """Return (occupancy, entry_count, exit_count, last_update).
    Tries Supabase first, falls back to local JSON."""
    # Try Supabase first
    if _DB_AVAILABLE:
        result = db.load_occupancy_from_db()
        if result is not None:
            occ, ent, ex, last = result
            if occ < 0 or occ > config.MAX_CAPACITY:
                print("[DB] Loaded occupancy out of range; clamping")
                occ = max(0, min(occ, config.MAX_CAPACITY))
            return occ, ent, ex, last

    # Fallback to JSON file
    if os.path.exists(config.OCCUPANCY_STATE_FILE):
        try:
            with open(config.OCCUPANCY_STATE_FILE, 'r') as f:
                data = json.load(f)
                occ = int(data.get('occupancy', 0))
                ent = int(data.get('entry_count', 0))
                ex = int(data.get('exit_count', 0))
                last = float(data.get('last_update', 0))
                if occ < 0 or occ > config.MAX_CAPACITY:
                    print("Loaded occupancy out of range; clamping and entering freeze mode")
                    occ = max(0, min(occ, config.MAX_CAPACITY))
                return occ, ent, ex, last
        except Exception as e:
            print(f"Failed to load occupancy file: {e}")
            return 0, 0, 0, 0.0
    return 0, 0, 0, 0.0


def load_vehicle_states():
    """Return list of vehicle state dicts.
    Tries Supabase first, falls back to local JSON."""
    # Try Supabase first
    if _DB_AVAILABLE:
        result = db.load_vehicle_states_from_db()
        if result is not None:
            return result

    # Fallback to JSON file
    if os.path.exists(config.OCCUPANCY_STATE_FILE):
        try:
            with open(config.OCCUPANCY_STATE_FILE, 'r') as f:
                data = json.load(f)
                vs = data.get('vehicle_states', [])
                out = []
                for v in vs:
                    try:
                        vid = v.get('id')
                        he = bool(v.get('has_entered', False))
                        hx = bool(v.get('has_exited', False))
                        out.append({'id': vid, 'has_entered': he, 'has_exited': hx})
                    except Exception:
                        continue
                return out
        except Exception as e:
            print(f"Failed to load vehicle states: {e}")
            return []
    return []


def save_vehicle_states(states):
    """Persist vehicle states list into the occupancy file, preserving audit and counters."""
    try:
        now = time.time()
        # Load existing record if present
        existing = {}
        if os.path.exists(config.OCCUPANCY_STATE_FILE):
            try:
                with open(config.OCCUPANCY_STATE_FILE, 'r') as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        out = dict(existing)
        out['vehicle_states'] = []
        for v in states:
            # `states` may be dict values or raw dicts; normalize safely
            vid = v.get('id') if isinstance(v, dict) else getattr(v, 'id', None)
            out['vehicle_states'].append({
                'id': vid,
                'has_entered': bool(v.get('has_entered', False)),
                'has_exited': bool(v.get('has_exited', False)),
            })

        # Ensure last_update exists
        out['last_update'] = float(out.get('last_update', now))

        _atomic_write(config.OCCUPANCY_STATE_FILE, out)
    except Exception as e:
        print(f"Failed to save vehicle states: {e}")


def get_vehicle_state(track_id):
    """Return a single vehicle state dict for `track_id` or None if not present."""
    global _vehicle_states_cache
    if _vehicle_states_cache is None:
        # Load on first access (startup recovery)
        lst = load_vehicle_states()
        _vehicle_states_cache = {v['id']: v for v in lst}

    return _vehicle_states_cache.get(track_id)


def update_vehicle_state(track_id, has_entered=None, has_exited=None):
    """Create or update vehicle state for `track_id`. Values left as None are not changed."""
    global _vehicle_states_cache
    if _vehicle_states_cache is None:
        lst = load_vehicle_states()
        _vehicle_states_cache = {v['id']: v for v in lst}

    cur = _vehicle_states_cache.get(track_id, {'id': track_id, 'has_entered': False, 'has_exited': False})

    # Determine prospective new values
    new_entered = cur['has_entered'] if has_entered is None else bool(has_entered)
    new_exited = cur['has_exited'] if has_exited is None else bool(has_exited)

    # Enforce invariant: cannot exit before entry
    if new_exited and not new_entered:
        raise ValueError(f"Invalid state update for ID {track_id}: exit before entry")

    cur['has_entered'] = new_entered
    cur['has_exited'] = new_exited
    _vehicle_states_cache[track_id] = cur

    # Persist snapshot (caller should avoid calling per-frame)
    try:
        save_vehicle_states(list(_vehicle_states_cache.values()))
    except Exception as e:
        print(f"Failed to persist vehicle state for {track_id}: {e}")

def _atomic_write(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def save_occupancy(occupancy, entry_count, exit_count, reason=None, vehicle_states=None):
    try:
        now = time.time()
        clamped_occ = int(max(0, min(occupancy, config.MAX_CAPACITY)))
        record = {
            "occupancy": clamped_occ,
            "entry_count": int(entry_count),
            "exit_count": int(exit_count),
            "last_update": now,
            "reason": reason or "event",
            "max_capacity": config.MAX_CAPACITY,
        }

        # Try to merge with existing audit if present
        audit = []
        if os.path.exists(config.OCCUPANCY_STATE_FILE):
            try:
                with open(config.OCCUPANCY_STATE_FILE, 'r') as f:
                    existing = json.load(f)
                    audit = existing.get('audit', [])
            except Exception:
                audit = []

        audit.append({'ts': now, 'occupancy': record['occupancy'], 'reason': record['reason']})
        audit = audit[-config.OCCUPANCY_AUDIT_LIMIT:]

        out = dict(record)
        out['audit'] = audit

        # Preserve or include vehicle_states
        vs_list = []
        if vehicle_states is not None:
            out['vehicle_states'] = []
            for v in vehicle_states:
                vid = v.get('id') if isinstance(v, dict) else getattr(v, 'id', None)
                try:
                    vid = int(vid)
                except Exception:
                    pass
                vs_entry = {
                    'id': vid,
                    'has_entered': bool(v.get('has_entered', False)),
                    'has_exited': bool(v.get('has_exited', False)),
                }
                out['vehicle_states'].append(vs_entry)
                vs_list.append(vs_entry)
        else:
            if os.path.exists(config.OCCUPANCY_STATE_FILE):
                try:
                    with open(config.OCCUPANCY_STATE_FILE, 'r') as f:
                        existing = json.load(f)
                        out['vehicle_states'] = existing.get('vehicle_states', [])
                except Exception:
                    out['vehicle_states'] = []
            else:
                out['vehicle_states'] = []

        # Write to local JSON (primary fallback)
        _atomic_write(config.OCCUPANCY_STATE_FILE, out)

        # Write to Supabase (secondary, fail-safe)
        if _DB_AVAILABLE:
            db.update_occupancy(
                clamped_occ, int(entry_count), int(exit_count),
                max_capacity=config.MAX_CAPACITY, reason=reason,
            )
            if vs_list:
                db.save_vehicle_states_bulk(vs_list)
    except Exception as e:
        print(f"Failed to save occupancy: {e}")