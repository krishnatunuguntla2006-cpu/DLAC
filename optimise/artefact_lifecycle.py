"""VisionPort - Model Artefact Lifecycle
PyTorch (.pt) -> ONNX (.onnx) -> TensorRT (.engine)
"""
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 1. Device and Environment Check
# ----------------------------------------------------------------------
print("=" * 60)
print("1. ENVIRONMENT CHECK")
print("=" * 60)
print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version used by PyTorch:", torch.version.cuda)
else:
    print("Running on CPU (use Google Colab with T4 GPU for TensorRT engine build)")

# ----------------------------------------------------------------------
# 2. Load YOLO Model & Inspect Parameters
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. LOAD YOLO MODEL & INSPECT PARAMETERS")
print("=" * 60)
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")
print("Task:", model.task)
print("Number of classes:", len(model.names))
print("Class names:", {k: model.names[k] for k in list(model.names.keys())[:5]}, "...")

total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
print("Total parameters    :", f"{total_params:,}")
print("Trainable parameters:", f"{trainable_params:,}")

# ----------------------------------------------------------------------
# 3. Baseline Prediction
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. BASELINE PREDICTION")
print("=" * 60)
# Use a sample local image or synthetic image
sample_img_path = Path("data/idd/images/traffic_001.jpg")
if sample_img_path.exists():
    sample_img = Image.open(sample_img_path).convert("RGB")
else:
    sample_img = Image.new("RGB", (640, 640), color=(128, 128, 128))

results = model.predict(source=sample_img, imgsz=640, device=device, conf=0.25, verbose=False)
print("Baseline prediction completed. Detections found:", len(results[0].boxes))

# ----------------------------------------------------------------------
# 4. Export PyTorch to ONNX
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. EXPORT TO ONNX")
print("=" * 60)
onnx_path = model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    device=device
)
print("ONNX export completed.")
print("ONNX file:", onnx_path)

# ----------------------------------------------------------------------
# 5. Validate and Inspect ONNX Graph
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. VALIDATE & INSPECT ONNX GRAPH")
print("=" * 60)
import onnx
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
print("ONNX checker: model is valid")
print("IR version   :", onnx_model.ir_version)
print("Opset version:", onnx_model.opset_import[0].version)

print("\nInputs:")
for tensor in onnx_model.graph.input:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in tensor.type.tensor_type.shape.dim]
    print(f"  {tensor.name}: {dims}")

print("\nOutputs:")
for tensor in onnx_model.graph.output:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in tensor.type.tensor_type.shape.dim]
    print(f"  {tensor.name}: {dims}")

print("\nNumber of graph nodes:", len(onnx_model.graph.node))

# ----------------------------------------------------------------------
# 6. ONNX Runtime Smoke Test
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. ONNX RUNTIME EXECUTION TEST")
print("=" * 60)
import onnxruntime as ort
onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
input_info = onnx_session.get_inputs()[0]
print("Input name :", input_info.name)
print("Input shape:", input_info.shape)
print("Input type :", input_info.type)

dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
onnx_outputs = onnx_session.run(None, {input_info.name: dummy_input})
print("ONNX Runtime execution completed.")
for idx, out in enumerate(onnx_outputs):
    print(f"  Output {idx} shape: {out.shape}")

# ----------------------------------------------------------------------
# 7. Package Artefacts & Size Report
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. ARTEFACT PACKAGING & SIZE REPORT")
print("=" * 60)
engine_path = Path("yolov8n.engine")
artifact_paths = [
    Path(MODEL_PATH),
    Path(onnx_path),
    engine_path
]

print("VISIONPORT – ARTEFACT REPORT")
print("-" * 50)
for p in artifact_paths:
    if p.exists():
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:<25} {size_mb:8.2f} MB")
    else:
        print(f"  {p.name:<25} [Not built in CPU environment - requires NVIDIA GPU + TensorRT]")

zip_name = "VisionPort_Artifacts.zip"
with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
    for p in artifact_paths:
        if p.exists():
            zip_file.write(p, arcname=p.name)
print(f"\nCreated archive: {zip_name}")
print("=" * 60)
