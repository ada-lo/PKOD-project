-- PKOD Supabase Schema Migration
-- Run this in the Supabase SQL Editor to create all required tables.

-- ============================================================
-- 1. vehicle_events — logs every entry/exit crossing
-- ============================================================
CREATE TABLE IF NOT EXISTS vehicle_events (
    id              BIGSERIAL PRIMARY KEY,
    track_id        INTEGER NOT NULL,
    event_type      TEXT NOT NULL CHECK (event_type IN ('entry', 'exit')),
    occupancy       INTEGER NOT NULL,
    timestamp       DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_vehicle_events_track ON vehicle_events (track_id);
CREATE INDEX IF NOT EXISTS idx_vehicle_events_ts    ON vehicle_events (timestamp);

-- ============================================================
-- 2. ocr_results — license plate readings linked to vehicles
-- ============================================================
CREATE TABLE IF NOT EXISTS ocr_results (
    id              BIGSERIAL PRIMARY KEY,
    track_id        INTEGER NOT NULL,
    plate_text      TEXT,
    confidence      DOUBLE PRECISION,
    event_type      TEXT CHECK (event_type IN ('entry', 'exit')),
    image_path      TEXT,
    timestamp       DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ocr_results_track ON ocr_results (track_id);

-- ============================================================
-- 3. occupancy_snapshot — single-row current state
-- ============================================================
CREATE TABLE IF NOT EXISTS occupancy_snapshot (
    id              INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    occupancy       INTEGER NOT NULL DEFAULT 0,
    entry_count     INTEGER NOT NULL DEFAULT 0,
    exit_count      INTEGER NOT NULL DEFAULT 0,
    max_capacity    INTEGER NOT NULL DEFAULT 80,
    last_update     DOUBLE PRECISION,
    frozen          BOOLEAN NOT NULL DEFAULT FALSE
);

-- Seed the single row
INSERT INTO occupancy_snapshot (id, occupancy, entry_count, exit_count, max_capacity, frozen)
VALUES (1, 0, 0, 0, 80, FALSE)
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- 4. audit_log — history of every occupancy change
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    occupancy       INTEGER NOT NULL,
    reason          TEXT,
    timestamp       DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log (timestamp);

-- ============================================================
-- 5. vehicle_states — active vehicle tracking for session recovery
-- ============================================================
CREATE TABLE IF NOT EXISTS vehicle_states (
    track_id        INTEGER PRIMARY KEY,
    has_entered     BOOLEAN NOT NULL DEFAULT FALSE,
    has_exited      BOOLEAN NOT NULL DEFAULT FALSE,
    last_seen       DOUBLE PRECISION
);
