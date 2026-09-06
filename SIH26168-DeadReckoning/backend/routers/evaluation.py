"""GET /traces/{id}/evaluation -- ATE/RTE/CEP/drift-rate (dossier Section 8).

Computed once at ingest time (see routers/ingest.py) and stored in
`evaluation_metrics`; this endpoint just reads it back.
"""
from fastapi import APIRouter, HTTPException

import db
from schemas import ErrorResponse, EvaluationMetrics

router = APIRouter(tags=["evaluation"])


@router.get(
    "/traces/{trace_id}/evaluation",
    response_model=EvaluationMetrics,
    summary="ATE / RTE / CEP-50 / CEP-90 / drift-rate for a trace",
    responses={404: {"model": ErrorResponse}},
)
def get_evaluation(trace_id: str) -> EvaluationMetrics:
    if db.get_trace(trace_id) is None:
        raise HTTPException(status_code=404, detail=f"trace '{trace_id}' not found")
    row = db.get_evaluation_metrics(trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no evaluation metrics for trace '{trace_id}'")
    return EvaluationMetrics(**row)
