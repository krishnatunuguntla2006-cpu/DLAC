"""Create the two sample trace files used in Session 8.

1. docs/sample_trace_cpu.json
   The numbers from the instructor's own Session 3 run: a 512 x 512 PNG on a
   CPU-only laptop. MEASURED.
2. docs/sample_trace_gpu_example.json
   An instructor-provided example of what a GPU machine might look like.
   ILLUSTRATIVE: these numbers were NOT measured. They exist so the class can
   see how the bottleneck moves when the hardware changes.

Run from the project root:  python make_sample_traces.py
"""
import json
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "docs"
out_dir.mkdir(parents=True, exist_ok=True)


def stage(n, name, where, fmt, shape, ms, note=""):
    return {"stage": n, "name": name, "where": where, "format": fmt,
            "shape": shape, "note": note, "ms": ms}


# ---- 1. Measured on CPU (your Session 3 trace) -------------------------------
cpu_trace = {
    "source": "MEASURED: instructor's Session 3 run, CPU only, 512 x 512 PNG",
    "device": "cpu",
    "original_size_hw": [512, 512],
    "stages": [
        stage(1, "receive upload", "CPU / gateway", "bytes", "540612 bytes", 0.12),
        stage(2, "decode image", "CPU / gateway", "numpy uint8 (H,W,C)", [512, 512, 3], 36.67),
        stage(3, "letterbox resize", "CPU / gateway", "numpy uint8 (H,W,C)", [640, 640, 3], 4.17),
        stage(4, "to tensor", "CPU / gateway", "torch float32 (N,C,H,W)", [1, 3, 640, 640], 7.5),
        stage(5, "host to device", "CPU -> cpu", "torch float32 (N,C,H,W)", [1, 3, 640, 640], 0.05,
              "no copy (already on CPU)"),
        stage(6, "model forward", "cpu", "torch float32 (N,4+classes,anchors)", [1, 84, 8400], 315.15),
        stage(7, "NMS", "cpu", "torch float32 (n,6)", [9, 6], 70.59),
        stage(8, "rescale boxes", "cpu", "torch float32 (n,6)", [9, 6], 2.81),
        stage(9, "serialise response", "CPU / gateway", "JSON", "9 detections", 1.52),
    ],
    "total_ms": 438.76,
}

# ---- 2. Illustrative GPU example (NOT measured) ------------------------------
gpu_trace = {
    "source": "ILLUSTRATIVE: instructor example for a GPU machine, 1280 x 720 image. NOT measured.",
    "device": "cuda",
    "original_size_hw": [720, 1280],
    "stages": [
        stage(1, "receive upload", "CPU / gateway", "bytes", "file size", 0.2),
        stage(2, "decode image", "CPU / gateway", "numpy uint8 (H,W,C)", [720, 1280, 3], 6.5),
        stage(3, "letterbox resize", "CPU / gateway", "numpy uint8 (H,W,C)", [640, 640, 3], 2.1),
        stage(4, "to tensor", "CPU / gateway", "torch float32 (N,C,H,W)", [1, 3, 640, 640], 1.4),
        stage(5, "host to device", "CPU -> cuda", "torch float32 (N,C,H,W)", [1, 3, 640, 640], 0.9,
              "copy over PCIe"),
        stage(6, "model forward", "cuda", "torch float32 (N,4+classes,anchors)", [1, 84, 8400], 11.8),
        stage(7, "NMS", "cuda", "torch float32 (n,6)", ["n", 6], 2.2),
        stage(8, "rescale boxes", "cuda", "torch float32 (n,6)", ["n", 6], 0.3),
        stage(9, "serialise response", "CPU / gateway", "JSON", "n detections", 0.1),
    ],
    "total_ms": 25.6,
}

for name, trace in (("sample_trace_cpu.json", cpu_trace),
                    ("sample_trace_gpu_example.json", gpu_trace)):
    (out_dir / name).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print("Wrote", out_dir / name)
print("The GPU file is an illustration, not a measurement.")
