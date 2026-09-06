"""Manual smoke test against the TestClient -- not pytest, just a quick
end-to-end sanity run during development. Run from backend/:
`python scripts/smoke_test.py`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

n = 400
t = np.arange(n) * 0.1
samples = []
for i in range(n):
    fix = {}
    if i % 50 == 0:
        fix = {"lat": 28.6139 + i * 1e-6, "lon": 77.2090 + i * 1e-6, "alt": 216.0, "gps_accuracy_m": 4.0}
    samples.append({
        "timestamp": 1735689600.0 + t[i],
        "acc_x": 0.1, "acc_y": -0.05, "acc_z": 9.79,
        "gyro_x": 0.001, "gyro_y": -0.002, "gyro_z": 0.0005,
        **fix,
    })

r = client.post("/traces", json={"device_id": "pixel-7-test", "label": "smoke test", "samples": samples})
print("POST /traces ->", r.status_code)
assert r.status_code == 201, r.text
trace = r.json()
print(trace)
trace_id = trace["trace_id"]

r = client.get(f"/traces/{trace_id}")
assert r.status_code == 200, r.text
print("GET /traces/{id} ->", r.status_code)

r = client.get("/traces")
assert r.status_code == 200
print("GET /traces -> count", r.json()["count"])

r = client.get(f"/traces/{trace_id}/baseline")
assert r.status_code == 200, r.text
pts = r.json()["points"]
print("GET .../baseline -> n_points", len(pts), "first", pts[0], "last", pts[-1])

r = client.get(f"/traces/{trace_id}/corrected")
assert r.status_code == 200, r.text
cpts = r.json()["points"]
print("GET .../corrected -> n_points", len(cpts))

r = client.get(f"/traces/{trace_id}/uncertainty")
assert r.status_code == 200, r.text
u = r.json()
print("GET .../uncertainty -> buckets", u["buckets"], "n_points", len(u["points"]))
print("sample uncertainty point", u["points"][0], u["points"][-1])

r = client.get(f"/traces/{trace_id}/evaluation")
assert r.status_code == 200, r.text
print("GET .../evaluation ->", r.json())

r = client.get(f"/traces/{trace_id}/baseline")
pts2 = r.json()["points"]
assert pts == pts2, "baseline should be deterministic across repeated calls"
print("determinism check passed")

r = client.get("/traces/does-not-exist/baseline")
assert r.status_code == 404
print("404 check passed")

print("\nALL SMOKE TESTS PASSED")
