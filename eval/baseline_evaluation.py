"""
VisionPort - Inference Baseline Evaluation Script
Corresponds to Session 9 workflow.
Measures latency (mean, median, P95), throughput, RAM, and GPU memory (if available).
Outputs:
  - eval/predictions.json
  - eval/baseline_report.csv
"""

import os
import sys

# Ensure ultralytics uses local folder and does not attempt offline auto-updates
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")
os.environ["ULTRALYTICS_AUTOUPDATE"] = "0"

import json
import time
import statistics
from pathlib import Path
from PIL import Image
import psutil
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO

def run_baseline_evaluation(
    model_path="yolov8n.pt",
    image_dir="data",
    output_dir="eval",
    max_images=200,
    rounds=3,
    warmup_runs=10,
    img_size=640,
    conf_thresh=0.25
):
    model_file = Path(model_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file.resolve()}")
    
    # Collect images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    search_dir = Path(image_dir)
    all_image_paths = [
        p for p in search_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    
    if not all_image_paths:
        raise FileNotFoundError(f"No image files found in {search_dir.resolve()}")
    
    # Validate images
    valid_images = []
    invalid_images = []
    for img_path in all_image_paths:
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid_images.append(img_path)
        except Exception as e:
            invalid_images.append({"file": str(img_path), "error": str(e)})
            
    print(f"Total image files found: {len(all_image_paths)}")
    print(f"Valid images: {len(valid_images)}")
    if invalid_images:
        print(f"Invalid images: {len(invalid_images)}")
        
    if not valid_images:
        raise ValueError("No valid images available for benchmarking.")
        
    benchmark_images = valid_images[:max_images]
    print(f"Images selected for benchmarking: {len(benchmark_images)}")
    
    # Device detection
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"Device: GPU (0: {device_name})")
    else:
        device = "cpu"
        device_name = "CPU"
        print("Device: CPU")
        
    # Load model
    print(f"Loading YOLO model from {model_file}...")
    model = YOLO(str(model_file))
    print("Model loaded successfully.")
    
    # Warm-up runs
    print(f"Starting model warm-up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        img_p = str(benchmark_images[i % len(benchmark_images)])
        _ = model.predict(
            source=img_p,
            imgsz=img_size,
            conf=conf_thresh,
            device=device,
            verbose=False
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print("Warm-up completed.")
    
    # Benchmark loop
    latencies = []
    prediction_records = []
    
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 ** 2)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
    print(f"\nStarting VisionPort benchmark ({rounds} rounds x {len(benchmark_images)} images)...")
    total_start = time.perf_counter()
    
    for round_no in range(rounds):
        print(f"Measurement round: {round_no + 1}/{rounds}")
        for img_path in benchmark_images:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            results = model.predict(
                source=str(img_path),
                imgsz=img_size,
                conf=conf_thresh,
                device=device,
                verbose=False
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000.0)
            
            if results is None or len(results) == 0:
                continue
                
            if round_no == rounds - 1:
                result = results[0]
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().tolist()
                        detections.append({
                            "class_id": class_id,
                            "class_name": model.names[class_id],
                            "confidence": round(confidence, 4),
                            "bbox_xyxy": [round(c, 2) for c in xyxy]
                        })
                prediction_records.append({
                    "image": str(img_path.relative_to(search_dir) if img_path.is_relative_to(search_dir) else img_path.name),
                    "detections": detections
                })
                
    total_end = time.perf_counter()
    total_elapsed = total_end - total_start
    ram_after = process.memory_info().rss / (1024 ** 2)
    
    total_operations = len(latencies)
    avg_lat = statistics.mean(latencies)
    med_lat = statistics.median(latencies)
    p95_lat = float(np.percentile(latencies, 95))
    throughput = total_operations / total_elapsed
    ram_observed = ram_after
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        gpu_mem = None
        gpu_res = None
        
    print("\n========== VISIONPORT BASELINE RESULTS ==========")
    print(f"Total inference operations: {total_operations}")
    print(f"Average latency: {round(avg_lat, 2)} ms")
    print(f"Median latency: {round(med_lat, 2)} ms")
    print(f"P95 latency: {round(p95_lat, 2)} ms")
    print(f"Throughput: {round(throughput, 2)} images/sec")
    print(f"Process RAM after benchmark: {round(ram_observed, 2)} MB")
    if gpu_mem is not None:
        print(f"GPU peak allocated memory: {round(gpu_mem, 2)} MB")
        print(f"GPU peak reserved memory: {round(gpu_res, 2)} MB")
    else:
        print("GPU memory: N/A - CUDA unavailable")
    print("================================================")
    
    # Save predictions.json
    pred_file = output_path / "predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(prediction_records, f, indent=4)
    print(f"Predictions saved: {pred_file} ({len(prediction_records)} records)")
    
    # Save baseline_report.csv
    report = {
        "Model": model_file.name,
        "Dataset": search_dir.name if search_dir.name else "Custom",
        "Images": len(benchmark_images),
        "Inference Operations": total_operations,
        "Average Latency (ms)": round(avg_lat, 2),
        "Median Latency (ms)": round(med_lat, 2),
        "P95 Latency (ms)": round(p95_lat, 2),
        "Throughput (images/sec)": round(throughput, 2),
        "Process RAM (MB)": round(ram_observed, 2),
        "GPU": device_name,
        "GPU Memory (MB)": round(gpu_mem, 2) if gpu_mem is not None else "N/A"
    }
    
    df = pd.DataFrame([report])
    csv_file = output_path / "baseline_report.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV report saved: {csv_file}")
    
    return report

if __name__ == "__main__":
    run_baseline_evaluation()
