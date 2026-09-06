"""POST /traces and the trace-metadata collection endpoints."""
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException

import db
import timeutil
from schemas import ErrorResponse, TraceIngestRequest, TraceListResponse, TraceSummary
from services import pipeline

router = APIRouter(tags=["traces"])


def _trace_row_to_summary(row: dict) -> TraceSummary:
    return TraceSummary(
        trace_id=row["trace_id"],
        device_id=row["device_id"],
        label=row["label"],
        created_at=row["created_at"],
        time_unit=row["time_unit"],
        start_timestamp=row["start_timestamp_raw"],
        end_timestamp=row["end_timestamp_raw"],
        duration_s=row["duration_s"],
        sample_count=row["sample_count"],
        has_mag=row["has_mag"],
        has_gps=row["has_gps"],
        status=row["status"],
    )


@router.post(
    "/traces",
    response_model=TraceSummary,
    status_code=201,
    summary="Ingest a raw IMU(+GPS) trace",
    responses={422: {"model": ErrorResponse, "description": "Malformed samples (see detail)"}},
)
def ingest_trace(payload: TraceIngestRequest) -> TraceSummary:
    """Accepts a full trace as one JSON body (not a stream) -- simplest
    contract for a hackathon demo where traces are recorded on-device and
    uploaded once the walk/drive is done.

    Samples are sorted by `timestamp` server-side, so client ordering
    doesn't matter. `time_unit: "auto"` (the default) detects seconds vs.
    milliseconds vs. nanoseconds the same way `physics-baseline/strapdown.py`
    does.

    Baseline/corrected/uncertainty are NOT computed here -- they're mock-
    generated on demand by the GET endpoints below, deterministically from
    the trace's duration, so this endpoint stays fast regardless of trace
    length. Evaluation metrics (ATE/RTE/CEP/drift-rate) ARE computed here
    and stored, since they summarize a completed trace once.
    """
    samples = sorted(payload.samples, key=lambda s: s.timestamp)
    raw_ts = np.array([s.timestamp for s in samples], dtype=np.float64)

    time_unit = payload.time_unit
    if time_unit == "auto":
        time_unit = timeutil.detect_time_unit(raw_ts)
    t = timeutil.to_seconds_elapsed(raw_ts, time_unit)

    has_mag = samples[0].mag_x is not None
    has_gps = any(s.has_fix for s in samples)
    duration_s = float(t[-1] - t[0])

    trace_id = str(uuid.uuid4())

    rows = [
        (
            i, float(t[i]), float(raw_ts[i]),
            s.acc_x, s.acc_y, s.acc_z, s.gyro_x, s.gyro_y, s.gyro_z,
            s.mag_x, s.mag_y, s.mag_z,
            s.lat, s.lon, s.alt, s.gps_accuracy_m,
            s.has_fix,
        )
        for i, s in enumerate(samples)
    ]

    try:
        created_at = db.insert_trace(
            trace_id=trace_id,
            device_id=payload.device_id,
            label=payload.label,
            time_unit=time_unit,
            start_timestamp_raw=float(raw_ts[0]),
            end_timestamp_raw=float(raw_ts[-1]),
            duration_s=duration_s,
            sample_count=len(samples),
            has_mag=has_mag,
            has_gps=has_gps,
        )
        db.insert_samples(trace_id, rows)
        metrics = pipeline.generate_evaluation_metrics(trace_id, duration_s)
        db.insert_evaluation_metrics(trace_id, metrics)
    except Exception as exc:  # noqa: BLE001 -- surface DB/insert failures as 500s with context
        raise HTTPException(status_code=500, detail=f"failed to store trace: {exc}") from exc

    return TraceSummary(
        trace_id=trace_id,
        device_id=payload.device_id,
        label=payload.label,
        created_at=created_at,
        time_unit=time_unit,
        start_timestamp=float(raw_ts[0]),
        end_timestamp=float(raw_ts[-1]),
        duration_s=duration_s,
        sample_count=len(samples),
        has_mag=has_mag,
        has_gps=has_gps,
        status="ready",
    )


@router.get("/traces", response_model=TraceListResponse, summary="List ingested traces")
def list_traces(limit: int = 100, offset: int = 0) -> TraceListResponse:
    rows = db.list_traces(limit=limit, offset=offset)
    return TraceListResponse(traces=[_trace_row_to_summary(r) for r in rows], count=len(rows))


@router.get(
    "/traces/{trace_id}",
    response_model=TraceSummary,
    summary="Get trace metadata",
    responses={404: {"model": ErrorResponse}},
)
def get_trace(trace_id: str) -> TraceSummary:
    row = db.get_trace(trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"trace '{trace_id}' not found")
    return _trace_row_to_summary(row)
