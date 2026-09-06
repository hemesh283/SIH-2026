"""Deterministic Python <-> Android Numerical Parity Test Generator for IO-VNBD.
SIH26168 Intelligent Dead Reckoning System.
"""

import json
from pathlib import Path
import hashlib
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ONNX_FROZEN = PROJECT_ROOT / "models" / "frozen_io_vnbd" / "gru_io_vnbd.onnx"
NORM_JSON_FROZEN = PROJECT_ROOT / "models" / "frozen_io_vnbd" / "io_vnbd_normalization.json"

MODEL_ONNX_ASSET = PROJECT_ROOT / "FE" / "gudumap" / "gudumap" / "app" / "src" / "main" / "assets" / "gru_io_vnbd.onnx"
NORM_JSON_ASSET = PROJECT_ROOT / "FE" / "gudumap" / "gudumap" / "app" / "src" / "main" / "assets" / "io_vnbd_normalization.json"

OUT_JSON = PROJECT_ROOT / "results" / "io_vnbd" / "deterministic_parity_test.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest().upper()


def main():
    print("=" * 70)
    print("IO-VNBD PYTHON <-> ANDROID NUMERICAL PARITY GENERATOR")
    print("=" * 70)

    # 1. Verify SHA-256 hashes
    hash_onnx_frozen = sha256(MODEL_ONNX_FROZEN)
    hash_onnx_asset = sha256(MODEL_ONNX_ASSET)
    assert hash_onnx_frozen == hash_onnx_asset, "ONNX hash mismatch!"

    hash_norm_frozen = sha256(NORM_JSON_FROZEN)
    hash_norm_asset = sha256(NORM_JSON_ASSET)
    assert hash_norm_frozen == hash_norm_asset, "Normalization hash mismatch!"

    print(f"ONNX Model SHA-256        : {hash_onnx_frozen}")
    print(f"Normalization JSON SHA-256: {hash_norm_frozen}")

    # 2. Deterministic generator with fixed seed 26168
    rng = np.random.RandomState(26168)

    # Generate 20 samples x 6 features in float32
    # Features: [acc_x(g), acc_y(g), acc_z(g), gyro_x(rad/s), gyro_y(rad/s), gyro_z(rad/s)]
    acc_g = rng.uniform(-0.25, 0.25, size=(20, 3)).astype(np.float32)
    gyro_rads = rng.uniform(-0.15, 0.15, size=(20, 3)).astype(np.float32)
    raw_input_20x6 = np.concatenate([acc_g, gyro_rads], axis=1)  # (20, 6)

    # 3. Load normalization parameters
    with open(NORM_JSON_FROZEN, "r") as f:
        norm_data = json.load(f)

    f_mean = np.array(norm_data["mean"], dtype=np.float32)
    f_std = np.array(norm_data["std"], dtype=np.float32)
    t_mean = np.array(norm_data["target_mean"], dtype=np.float32)
    t_std = np.array(norm_data["target_std"], dtype=np.float32)

    # 4. Normalize
    norm_input = (raw_input_20x6 - f_mean) / f_std
    norm_input_batch = np.expand_dims(norm_input, axis=0)  # (1, 20, 6)

    # 5. Run ONNX Runtime
    sess = ort.InferenceSession(str(MODEL_ONNX_FROZEN))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    raw_output = sess.run([output_name], {input_name: norm_input_batch})[0]  # (1, 3)
    raw_out_1d = raw_output[0]

    # 6. Target denormalization
    denorm_output = raw_out_1d * t_std + t_mean

    dx = float(denorm_output[0])
    dy = float(denorm_output[1])
    dz = float(denorm_output[2])

    print("\nDeterministic Execution Results:")
    print(f"Raw ONNX Output [dx_norm, dy_norm, dz_norm]: {raw_out_1d.tolist()}")
    print(f"Denormalized Displacement [dx, dy, dz] (m) : [{dx:.8f}, {dy:.8f}, {dz:.8f}]")

    # 7. Save JSON test record
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "description": "Deterministic IO-VNBD numerical parity test tensor and reference outputs",
        "seed": 26168,
        "input_shape": list(raw_input_20x6.shape),
        "onnx_model_sha256": hash_onnx_frozen,
        "normalization_sha256": hash_norm_frozen,
        "input_tensor_20x6": raw_input_20x6.tolist(),
        "normalized_tensor_20x6": norm_input.tolist(),
        "raw_onnx_output": raw_out_1d.tolist(),
        "denormalized_displacement_meters": [dx, dy, dz],
    }

    with open(OUT_JSON, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nSaved parity test record to {OUT_JSON}")


if __name__ == "__main__":
    main()
