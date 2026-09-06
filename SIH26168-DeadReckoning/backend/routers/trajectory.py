"""GET /traces/{id}/baseline, /corrected, /uncertainty.

All three are mock-generated on demand (see services/pipeline.py) --
deterministic per trace_id, so repeated calls return the same trajectory.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

import db
from schemas import ErrorResponse, TrajectoryResponse, UncertaintyResponse
from services import pipeline

router = APIRouter(tags=["trajectory"])


def _require_trace(trace_id: str) -> dict:
    row = db.get_trace(trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"trace '{trace_id}' not found")
    return row


@router.get(
    "/traces/{trace_id}/baseline",
    response_model=TrajectoryResponse,
    summary="Naive strapdown/ZUPT trajectory (no ML correction)",
    responses={404: {"model": ErrorResponse}},
)
def get_baseline(trace_id: str) -> TrajectoryResponse:
    """The uncorrected dead-reckoned path -- what strapdown mechanization +
    ZUPT alone produces, before the ML residual correction. Currently mock
    data (see `services/pipeline.py`); will be replaced by real
    `physics-baseline/strapdown.py` output once that stage is wired in."""
    trace = _require_trace(trace_id)
    points = pipeline.generate_baseline(
        trace_id, trace["duration_s"], trace["start_timestamp_raw"], trace["time_unit"]
    )
    return TrajectoryResponse(
        trace_id=trace_id, source="mock", generated_at=datetime.now(timezone.utc), points=points
    )


@router.get(
    "/traces/{trace_id}/corrected",
    response_model=TrajectoryResponse,
    summary="Baseline + ML residual correction",
    responses={404: {"model": ErrorResponse}},
)
def get_corrected(trace_id: str) -> TrajectoryResponse:
    """Baseline trajectory with the LightGBM residual correction applied
    (dossier Section 4.2/7). Currently mock data that simulates a partial
    (not perfect) drift correction, so this endpoint is visibly closer to
    ground truth than /baseline but not identical to it -- the same
    qualitative relationship the real ml-residual model will have."""
    trace = _require_trace(trace_id)
    points = pipeline.generate_corrected(
        trace_id, trace["duration_s"], trace["start_timestamp_raw"], trace["time_unit"]
    )
    return TrajectoryResponse(
        trace_id=trace_id, source="mock", generated_at=datetime.now(timezone.utc), points=points
    )


@router.get(
    "/traces/{trace_id}/uncertainty",
    response_model=UncertaintyResponse,
    summary="Duration-bucketed confidence radius",
    responses={404: {"model": ErrorResponse}},
)
def get_uncertainty(trace_id: str) -> UncertaintyResponse:
    """Confidence radius per point, bucketed by elapsed time since the last
    GNSS fix (dossier Section 4.3 / jury Q8) -- widens with blackout
    duration, nothing else. Bucket radii are a mock calibration for now
    (`services/pipeline.UNCERTAINTY_BUCKETS`); real values will come from
    `ml-residual/uncertainty.py`'s cross-validated CEP-90-per-bucket fit.
    `blackout_duration_s` per point is derived from this trace's own
    ingested GPS-fix timestamps, so it reflects real trace metadata even
    though the radius values themselves are mock."""
    trace = _require_trace(trace_id)
    fix_times = db.get_fix_times(trace_id)
    result = pipeline.generate_uncertainty(
        trace_id, trace["duration_s"], trace["start_timestamp_raw"], trace["time_unit"], fix_times
    )
    return UncertaintyResponse(
        trace_id=trace_id,
        source="mock",
        coverage=result["coverage"],
        generated_at=datetime.now(timezone.utc),
        buckets=result["buckets"],
        points=result["points"],
    )
