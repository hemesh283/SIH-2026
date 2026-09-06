# IO-VNBD ML FREEZE MANIFEST

Project: SIH26168
Dataset: IO-VNBD
Model: 2-layer GRU
Hidden size: 64
Dropout: 0.2

Input shape: [1,20,6]
Output shape: [1,3]

Sampling rate: 10 Hz
Window: 20 samples
Physical window: 2 seconds
Stride: 10 samples

Feature order:
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

Output:
dx, dy, dz in meters
local initial frame

Frozen artifacts:
gru_io_vnbd_best.pt
gru_io_vnbd.onnx
io_vnbd_normalization.json
model_metadata.json

Status: FROZEN
