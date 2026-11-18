import io
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="VisionPort Gateway")

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model once
model = YOLO("yolov8n.pt")
model.to(device)


@app.get("/")
def home():
    return {
        "project": "VisionPort",
        "message": "VisionPort FastAPI Gateway is running!",
        "status": "success",
        "device": device,
        "available_endpoints": [
            "/",
            "/health",
            "/detect",
            "/docs"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "yolov8n",
        "device": device
    }


@app.post("/detect")
async def detect(
        file: UploadFile = File(...),
        conf: float = 0.35
):

    # Validate confidence threshold
    if not 0 <= conf <= 1:
        raise HTTPException(
            status_code=400,
            detail="conf must be between 0 and 1"
        )

    # Validate file type
    if file.content_type not in (
            "image/jpeg",
            "image/png"
    ):
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported"
        )

    try:
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Run YOLO inference
    result = model.predict(
        image,
        conf=conf,
        device=device,
        verbose=False
    )[0]

    # Format detections
    detections = [
        {
            "class": result.names[int(cls)],
            "confidence": round(float(score), 3),
            "box_xyxy": [
                round(float(v), 1)
                for v in box
            ]
        }
        for cls, score, box in zip(
            result.boxes.cls,
            result.boxes.conf,
            result.boxes.xyxy
        )
    ]

    return {
        "image_size": {
            "width": image.width,
            "height": image.height
        },
        "num_detections": len(detections),
        "detections": detections,
        "timing_ms": {
            key: round(value, 2)
            for key, value in result.speed.items()
        }
    }