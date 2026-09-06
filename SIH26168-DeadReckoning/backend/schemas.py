"""Pydantic request/response models -- the API contract itself.

Field names deliberately mirror the raw-trace and baseline-trace column
contracts already established in `physics-baseline/strapdown.py` and
`ml-residual/common.py` (dossier Section 4), so the Android client and the
future real pipeline agree on the same vocabulary from day one.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

TimeUnit = Literal["auto", "s", "ms", "ns"]


# --------------------------------------------------------------------------
# Ingest: POST /traces
# --------------------------------------------------------------------------

class IMUSample(BaseModel):
    """One raw sample as captured by the phone's sensor-logging app.

    `timestamp` is the device's raw clock reading, in whatever unit the
    trace declares (see `TraceIngestRequest.time_unit`) -- seconds,
    milliseconds, or nanoseconds since some device-local epoch. It is
    *not* required to be Unix time.
    """

    timestamp: float = Field(..., description="Raw device timestamp, in the trace's declared time_unit.")
    acc_x: float = Field(..., description="Raw accelerometer X, m/s^2 (specific force, includes gravity).")
    acc_y: float = Field(..., description="Raw accelerometer Y, m/s^2.")
    acc_z: float = Field(..., description="Raw accelerometer Z, m/s^2.")
    gyro_x: float = Field(..., description="Gyroscope X, rad/s.")
    gyro_y: float = Field(..., description="Gyroscope Y, rad/s.")
    gyro_z: float = Field(..., description="Gyroscope Z, rad/s.")
    mag_x: float | None = Field(None, description="Magnetometer X, microtesla. Omit entirely if the device has no magnetometer.")
    mag_y: float | None = None
    mag_z: float | None = None
    lat: float | None = Field(None, description="GPS latitude, degrees. Present only on samples with a GNSS fix.")
    lon: float | None = Field(None, description="GPS longitude, degrees.")
    alt: float | None = Field(None, description="GPS altitude, meters.")
    gps_accuracy_m: float | None = Field(None, description="GPS horizontal accuracy estimate, meters.")

    @property
    def has_fix(self) -> bool:
        return self.lat is not None and self.lon is not None

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": 1735689600.123,
                "acc_x": 0.12, "acc_y": -0.05, "acc_z": 9.79,
                "gyro_x": 0.001, "gyro_y": -0.002, "gyro_z": 0.0005,
                "mag_x": 22.1, "mag_y": -5.4, "mag_z": 41.8,
                "lat": 28.6139, "lon": 77.2090, "alt": 216.0, "gps_accuracy_m": 4.5,
            }
        }
    }


class TraceIngestRequest(BaseModel):
    device_id: str = Field(..., description="Free-form identifier for the recording device/phone.")
    label: str | None = Field(None, description="Human-readable note, e.g. 'metro tunnel walk 1'.")
    time_unit: TimeUnit = Field("auto", description="Unit of every sample's `timestamp`. 'auto' detects s/ms/ns by magnitude.")
    samples: list[IMUSample] = Field(..., min_length=2, description="Raw IMU(+GPS) samples, any order (server sorts by timestamp).")

    @field_validator("samples")
    @classmethod
    def _mag_all_or_nothing(cls, samples: list[IMUSample]) -> list[IMUSample]:
        has_any = any(s.mag_x is not None or s.mag_y is not None or s.mag_z is not None for s in samples)
        if not has_any:
            return samples
        for s in samples:
            if s.mag_x is None or s.mag_y is None or s.mag_z is None:
                raise ValueError("mag_x/mag_y/mag_z must be all-present or all-absent on every sample")
        return samples


class TraceSummary(BaseModel):
    trace_id: str
    device_id: str
    label: str | None
    created_at: datetime
    time_unit: Literal["s", "ms", "ns"]
    start_timestamp: float = Field(..., description="First sample's raw timestamp (trace's time_unit).")
    end_timestamp: float = Field(..., description="Last sample's raw timestamp (trace's time_unit).")
    duration_s: float
    sample_count: int
    has_mag: bool
    has_gps: bool
    status: Literal["ready"] = Field("ready", description="Mock pipeline is synchronous: a trace is 'ready' as soon as it's ingested.")


class TraceListResponse(BaseModel):
    traces: list[TraceSummary]
    count: int


# --------------------------------------------------------------------------
# Trajectory: GET /traces/{id}/baseline, /corrected
# --------------------------------------------------------------------------

class TrajectoryPoint(BaseModel):
    t: float = Field(..., description="Seconds elapsed since trace start.")
    timestamp: float = Field(..., description="Raw timestamp, trace's time_unit (same clock as the ingested samples).")
    pos_x: float = Field(..., description="Local ENU East position, meters, origin at trace start.")
    pos_y: float = Field(..., description="Local ENU North position, meters.")
    pos_z: float = Field(..., description="Local ENU Up position, meters.")
    velocity: float = Field(..., description="Speed, m/s.")
    orientation: float = Field(..., description="Heading, degrees, 0-360, 0=North.")


class TrajectoryResponse(BaseModel):
    trace_id: str
    source: Literal["mock", "physics-baseline", "physics-baseline+ml-residual"]
    generated_at: datetime
    points: list[TrajectoryPoint]


# --------------------------------------------------------------------------
# Uncertainty: GET /traces/{id}/uncertainty
# --------------------------------------------------------------------------

class UncertaintyBucket(BaseModel):
    label: str = Field(..., description="e.g. '0-30s', '30-90s', '90s+'.")
    lo_s: float
    hi_s: float | None = Field(None, description="null means unbounded (the last bucket).")
    radius_m: float = Field(..., description="CEP-`coverage` confidence radius for this blackout-duration bucket.")


class UncertaintyPoint(BaseModel):
    t: float = Field(..., description="Seconds elapsed since trace start.")
    timestamp: float
    blackout_duration_s: float = Field(..., description="Seconds since the most recent GNSS fix at or before t (or since trace start, if none yet).")
    bucket: str
    radius_m: float


class UncertaintyResponse(BaseModel):
    trace_id: str
    source: Literal["mock", "ml-residual"]
    coverage: float = Field(0.9, description="CEP coverage fraction the radii represent, e.g. 0.9 = CEP-90.")
    generated_at: datetime
    buckets: list[UncertaintyBucket]
    points: list[UncertaintyPoint]


# --------------------------------------------------------------------------
# Evaluation: GET /traces/{id}/evaluation
# --------------------------------------------------------------------------

class EvaluationMetrics(BaseModel):
    trace_id: str
    ate_m: float = Field(..., description="Absolute Trajectory Error: RMS deviation, corrected vs ground truth, meters.")
    rte_m: float = Field(..., description="Relative Trajectory Error: mean drift over rte_window_s sub-windows, meters.")
    rte_window_s: float
    cep50_m: float = Field(..., description="Radius within which true position falls 50% of the time.")
    cep90_m: float = Field(..., description="Radius within which true position falls 90% of the time.")
    drift_rate_pct: float = Field(..., description="Final position error as a percentage of total distance travelled.")
    distance_m: float = Field(..., description="Total distance travelled (mock ground-truth path), meters.")
    computed_at: datetime
    source: Literal["mock", "evaluation"] = "mock"


class ErrorResponse(BaseModel):
    detail: str
