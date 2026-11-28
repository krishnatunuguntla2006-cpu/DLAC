"""VisionPort Session: YOLO Inference Wrapper and Tensor Contracts

Install:
    pip install ultralytics torch pillow numpy

Run wrapper test:
    python yolo_inference_wrapper.py

Run wrapper test + ONNX export:
    python yolo_inference_wrapper.py --export

Update MODEL_PATH and IMAGE_PATH to match your project.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure Ultralytics to use local directory and disable auto-updates
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath(".ultralytics"))
os.environ.setdefault("ULTRALYTICS_AUTOUPDATE", "0")

from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# 1. Project paths and inference settings
MODEL_PATH = Path("models/yolov8n.pt")
IMAGE_PATH = Path("data/idd/images/traffic-1.png")
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100
DEVICE = "cpu"  # CPU is suitable for this session; GPU is optional.


class YOLOInferenceWrapper:
    """
    Input contract:
      images: float32 [B, 3, H, W], RGB, values in [0, 1]
      H and W must be multiples of 32.
    Output contract:
      detections: float32 [B, max_detections, 6]
                  columns = x1, y1, x2, y2, confidence, class_id
      valid_counts: int64 [B], number of valid rows per image.

    Box coordinates refer to the input tensor's image canvas.
    """

    def __init__(self, model_path, device="cpu", conf=0.25,
                 iou=0.45, max_detections=100):
        # Stop early if the trained checkpoint path is incorrect.
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {Path(model_path).resolve()}")

        # Load the trained YOLO checkpoint.
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_detections = max_detections

    def validate_input(self, images):
        # Input must be a PyTorch tensor.
        if not isinstance(images, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        # Four dimensions are required: B, C, H, W.
        if images.ndim != 4:
            raise ValueError("Expected shape [B, 3, H, W].")

        batch, channels, height, width = images.shape

        # This wrapper expects RGB images.
        if channels != 3:
            raise ValueError(f"Expected 3 RGB channels; received {channels}.")

        if batch < 1:
            raise ValueError("Batch size must be at least one.")

        # Use dimensions divisible by 32 for this exercise.
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError("Height and width must be multiples of 32.")

        # The declared contract requires float32 values.
        if images.dtype != torch.float32:
            raise TypeError("Input tensor must be torch.float32.")

        # Reject invalid numeric values.
        if not torch.isfinite(images).all():
            raise ValueError("Input contains NaN or infinite values.")

        # Check the normalized pixel range.
        if images.min().item() < 0.0 or images.max().item() > 1.0:
            raise ValueError("Input values must be in [0.0, 1.0].")

    @torch.inference_mode()
    def __call__(self, images):
        # Check the tensor contract before inference.
        self.validate_input(images)

        # Ultralytics accepts BCHW float tensors normalized to [0, 1].
        results = self.model.predict(
            source=images,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )

        batch = images.shape[0]

        # Fixed-size output: unused rows remain zero.
        detections = torch.zeros(
            (batch, self.max_detections, 6), dtype=torch.float32
        )

        # This tensor tells downstream code how many rows are valid.
        valid_counts = torch.zeros((batch,), dtype=torch.int64)

        # Convert each variable-length Results object to six-column rows.
        for index, result in enumerate(results):
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.detach().cpu().to(torch.float32)
            confidence = result.boxes.conf.detach().cpu().to(torch.float32)
            class_id = result.boxes.cls.detach().cpu().to(torch.float32)

            rows = torch.cat(
                (xyxy, confidence.unsqueeze(1), class_id.unsqueeze(1)), dim=1
            )

            # Respect the maximum number of rows in the output contract.
            count = min(rows.shape[0], self.max_detections)
            detections[index, :count] = rows[:count]
            valid_counts[index] = count

        return detections, valid_counts


def image_to_tensor(image_path, image_size):
    # Open the sample and standardize its colour mode.
    image = Image.open(image_path).convert("RGB")

    # Simple square resize for this classroom demonstration.
    # Production pipelines should use the same letterbox preprocessing
    # as training/validation and retain resize/padding metadata.
    image = image.resize((image_size, image_size))

    # Convert PIL pixels to a uint8 NumPy array [H, W, C].
    array = np.asarray(image)

    # Convert to a tensor, then change HWC to CHW.
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    # Convert pixels from uint8 [0, 255] to float32 [0, 1].
    tensor = tensor.to(torch.float32) / 255.0

    # Add batch dimension: [3, H, W] becomes [1, 3, H, W].
    return tensor.unsqueeze(0).contiguous()


def export_onnx(model_path, image_size, device):
    # Export through Ultralytics' supported exporter.
    model = YOLO(str(model_path))
    output_path = model.export(
        format="onnx",
        imgsz=image_size,
        opset=17,
        dynamic=False,
        simplify=False,
        half=False,
        device=device,
    )
    print("ONNX export completed:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export", action="store_true",
        help="Also export the trained model to ONNX."
    )
    args = parser.parse_args()

    # Create the wrapper and load the trained checkpoint.
    wrapper = YOLOInferenceWrapper(
        MODEL_PATH, DEVICE, CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD, MAX_DETECTIONS
    )

    # Prepare one image using the documented input contract.
    input_tensor = image_to_tensor(IMAGE_PATH, IMAGE_SIZE)

    print("INPUT CONTRACT")
    print("shape:", tuple(input_tensor.shape))
    print("dtype:", input_tensor.dtype)
    max_val = input_tensor.max().item() if hasattr(input_tensor.max(), "item") else float(input_tensor.max())
    print("range:", (round(input_tensor.min().item(), 1), round(max_val, 1)))

    # Run inference and receive the standardized output tensors.
    detections, valid_counts = wrapper(input_tensor)

    print("\nOUTPUT CONTRACT")
    print("detections shape:", tuple(detections.shape))
    print("detections dtype:", detections.dtype)
    print("valid_counts shape:", tuple(valid_counts.shape))
    print("valid_counts dtype:", valid_counts.dtype)
    print("number of detections:", valid_counts.tolist())

    if valid_counts[0].item() > 0:
        print("first detection [x1,y1,x2,y2,conf,class]:",
              detections[0, 0].tolist())
    else:
        print("No objects detected in the sample image.")

    # ONNX export is optional and runs only with --export.
    if args.export:
        export_onnx(MODEL_PATH, IMAGE_SIZE, DEVICE)


if __name__ == "__main__":
    main()
