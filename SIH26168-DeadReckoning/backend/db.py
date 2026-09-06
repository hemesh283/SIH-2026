"""DuckDB schema and access layer for SIH26168 (dossier Section 6-7).

Embedded, single-file DuckDB database -- no external service to run for the
demo. DuckDB's Python connection is not thread-safe for concurrent writes,
so every access goes through `_LOCK`; fine at hackathon-demo scale (FastAPI
runs sync `def` endpoints in a threadpool, one DB call at a time).

Three tables, per dossier Section 8's evaluation metrics and Section 7's
data flow:
  - `traces`             one row per ingested trip (metadata)
  - `raw_imu_samples`    one row per ingested IMU(+GPS) sample
  - `evaluation_metrics` one row per trace: ATE / RTE / CEP-50 / CEP-90 / drift-rate

Column names for timestamp/position/velocity/orientation deliberately match
the `timestamp, pos_x/y/z, velocity, orientation` contract already used by
`physics-baseline/strapdown.py` and `ml-residual/common.py`, so wiring the
real pipeline in later (replacing `services/pipeline.py`) doesn't require a
schema change.
"""
import threading
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent / "data" / "traces.duckdb"

_LOCK = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id            VARCHAR PRIMARY KEY,
    device_id           VARCHAR NOT NULL,
    label               VARCHAR,
    created_at          TIMESTAMPTZ NOT NULL,
    time_unit           VARCHAR NOT NULL,      -- 's' | 'ms' | 'ns' -- raw timestamp unit as ingested
    start_timestamp_raw DOUBLE NOT NULL,        -- first sample's raw timestamp, original units
    end_timestamp_raw   DOUBLE NOT NULL,
    duration_s          DOUBLE NOT NULL,        -- end - start, normalized seconds
    sample_count        INTEGER NOT NULL,
    has_mag             BOOLEAN NOT NULL,
    has_gps             BOOLEAN NOT NULL,
    status              VARCHAR NOT NULL DEFAULT 'ready'  -- mock pipeline is synchronous: always 'ready'
);

CREATE TABLE IF NOT EXISTS raw_imu_samples (
    trace_id      VARCHAR NOT NULL REFERENCES traces(trace_id),
    seq           INTEGER NOT NULL,   -- 0-based order within the trace
    t             DOUBLE NOT NULL,    -- seconds since trace start
    timestamp_raw DOUBLE NOT NULL,    -- original raw timestamp, in traces.time_unit units
    acc_x DOUBLE NOT NULL, acc_y DOUBLE NOT NULL, acc_z DOUBLE NOT NULL,
    gyro_x DOUBLE NOT NULL, gyro_y DOUBLE NOT NULL, gyro_z DOUBLE NOT NULL,
    mag_x DOUBLE, mag_y DOUBLE, mag_z DOUBLE,
    lat DOUBLE, lon DOUBLE, alt DOUBLE, gps_accuracy_m DOUBLE,
    fix_available BOOLEAN NOT NULL,
    PRIMARY KEY (trace_id, seq)
);

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    trace_id       VARCHAR PRIMARY KEY REFERENCES traces(trace_id),
    ate_m          DOUBLE NOT NULL,   -- Absolute Trajectory Error: RMS(corrected, ground truth), meters
    rte_m          DOUBLE NOT NULL,   -- Relative Trajectory Error: mean drift over rte_window_s sub-windows, meters
    rte_window_s   DOUBLE NOT NULL,
    cep50_m        DOUBLE NOT NULL,   -- radius containing 50% of position error
    cep90_m        DOUBLE NOT NULL,   -- radius containing 90% of position error
    drift_rate_pct DOUBLE NOT NULL,   -- final position error as % of distance travelled
    distance_m     DOUBLE NOT NULL,
    computed_at    TIMESTAMPTZ NOT NULL,
    source         VARCHAR NOT NULL DEFAULT 'mock'  -- 'mock' until physics-baseline/ml-residual are wired in
);
"""


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(DB_PATH))
        _conn.execute(SCHEMA)
    return _conn


def init_db() -> None:
    """Idempotent -- call once at app startup."""
    get_conn()


def insert_trace(
    trace_id: str,
    device_id: str,
    label: str | None,
    time_unit: str,
    start_timestamp_raw: float,
    end_timestamp_raw: float,
    duration_s: float,
    sample_count: int,
    has_mag: bool,
    has_gps: bool,
) -> datetime:
    created_at = datetime.now(timezone.utc)
    with _LOCK:
        conn = get_conn()
        conn.execute(
            """
            INSERT INTO traces (
                trace_id, device_id, label, created_at, time_unit,
                start_timestamp_raw, end_timestamp_raw, duration_s,
                sample_count, has_mag, has_gps, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready')
            """,
            [
                trace_id, device_id, label, created_at, time_unit,
                start_timestamp_raw, end_timestamp_raw, duration_s,
                sample_count, has_mag, has_gps,
            ],
        )
    return created_at


def insert_samples(trace_id: str, rows: list[tuple]) -> None:
    """`rows` columns must match `raw_imu_samples` minus the leading trace_id,
    i.e.: seq, t, timestamp_raw, acc_x/y/z, gyro_x/y/z, mag_x/y/z,
    lat, lon, alt, gps_accuracy_m, fix_available."""
    with _LOCK:
        conn = get_conn()
        conn.executemany(
            """
            INSERT INTO raw_imu_samples (
                trace_id, seq, t, timestamp_raw,
                acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
                mag_x, mag_y, mag_z, lat, lon, alt, gps_accuracy_m, fix_available
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(trace_id, *row) for row in rows],
        )


def insert_evaluation_metrics(trace_id: str, metrics: dict) -> datetime:
    computed_at = datetime.now(timezone.utc)
    with _LOCK:
        conn = get_conn()
        conn.execute(
            """
            INSERT INTO evaluation_metrics (
                trace_id, ate_m, rte_m, rte_window_s, cep50_m, cep90_m,
                drift_rate_pct, distance_m, computed_at, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'mock')
            """,
            [
                trace_id, metrics["ate_m"], metrics["rte_m"], metrics["rte_window_s"],
                metrics["cep50_m"], metrics["cep90_m"], metrics["drift_rate_pct"],
                metrics["distance_m"], computed_at,
            ],
        )
    return computed_at


def get_trace(trace_id: str) -> dict | None:
    with _LOCK:
        conn = get_conn()
        row = conn.execute(
            "SELECT trace_id, device_id, label, created_at, time_unit, "
            "start_timestamp_raw, end_timestamp_raw, duration_s, sample_count, "
            "has_mag, has_gps, status FROM traces WHERE trace_id = ?",
            [trace_id],
        ).fetchone()
    if row is None:
        return None
    cols = ["trace_id", "device_id", "label", "created_at", "time_unit",
            "start_timestamp_raw", "end_timestamp_raw", "duration_s", "sample_count",
            "has_mag", "has_gps", "status"]
    return dict(zip(cols, row))


def list_traces(limit: int = 100, offset: int = 0) -> list[dict]:
    with _LOCK:
        conn = get_conn()
        rows = conn.execute(
            "SELECT trace_id, device_id, label, created_at, time_unit, "
            "start_timestamp_raw, end_timestamp_raw, duration_s, sample_count, "
            "has_mag, has_gps, status FROM traces ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
    cols = ["trace_id", "device_id", "label", "created_at", "time_unit",
            "start_timestamp_raw", "end_timestamp_raw", "duration_s", "sample_count",
            "has_mag", "has_gps", "status"]
    return [dict(zip(cols, row)) for row in rows]


def get_evaluation_metrics(trace_id: str) -> dict | None:
    with _LOCK:
        conn = get_conn()
        row = conn.execute(
            "SELECT trace_id, ate_m, rte_m, rte_window_s, cep50_m, cep90_m, "
            "drift_rate_pct, distance_m, computed_at, source "
            "FROM evaluation_metrics WHERE trace_id = ?",
            [trace_id],
        ).fetchone()
    if row is None:
        return None
    cols = ["trace_id", "ate_m", "rte_m", "rte_window_s", "cep50_m", "cep90_m",
            "drift_rate_pct", "distance_m", "computed_at", "source"]
    return dict(zip(cols, row))


def get_fix_times(trace_id: str) -> list[float]:
    """Sorted `t` (seconds since trace start) of every sample with a GNSS fix."""
    with _LOCK:
        conn = get_conn()
        rows = conn.execute(
            "SELECT t FROM raw_imu_samples WHERE trace_id = ? AND fix_available "
            "ORDER BY t",
            [trace_id],
        ).fetchall()
    return [r[0] for r in rows]
