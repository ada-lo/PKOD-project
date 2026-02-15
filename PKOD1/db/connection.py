"""
Supabase connection manager for PKOD.

Provides a singleton Supabase client initialised from environment variables.
Uses python-dotenv to load .env from the project root.
"""

import os
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file: db/ -> PKOD1/ -> PKOD/)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(_project_root, '.env'))

_client = None


def get_client():
    """Return a lazily-initialised Supabase client (singleton)."""
    global _client
    if _client is not None:
        return _client

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')

    if not url or not key:
        print("[DB] WARNING: SUPABASE_URL or SUPABASE_KEY not set in .env — database features disabled")
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        print("[DB] Supabase client connected successfully")
        return _client
    except Exception as e:
        print(f"[DB] Failed to create Supabase client: {e}")
        return None
