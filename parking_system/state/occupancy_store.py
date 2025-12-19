import json
import os
import time
import config

def load_occupancy():
    """Return (occupancy, entry_count, exit_count, last_update)"""
    if os.path.exists(config.OCCUPANCY_STATE_FILE):
        try:
            with open(config.OCCUPANCY_STATE_FILE, 'r') as f:
                data = json.load(f)
                occ = int(data.get('occupancy', 0))
                ent = int(data.get('entry_count', 0))
                ex = int(data.get('exit_count', 0))
                last = float(data.get('last_update', 0))
                # Validate range
                if occ < 0 or occ > config.MAX_CAPACITY:
                    print("Loaded occupancy out of range; clamping and entering freeze mode")
                    occ = max(0, min(occ, config.MAX_CAPACITY))
                return occ, ent, ex, last
        except Exception as e:
            print(f"Failed to load occupancy file: {e}")
            return 0, 0, 0, 0.0
    return 0, 0, 0, 0.0

def _atomic_write(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def save_occupancy(occupancy, entry_count, exit_count, reason=None):
    try:
        now = time.time()
        record = {
            'occupancy': int(max(0, min(occupancy, config.MAX_CAPACITY))),
            'entry_count': int(entry_count),
            'exit_count': int(exit_count),
            'last_update': now,
            'reason': reason or 'event',
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
        _atomic_write(config.OCCUPANCY_STATE_FILE, out)
    except Exception as e:
        print(f"Failed to save occupancy: {e}")