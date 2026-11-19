"""Create the Session 5 SYNTHETIC images sample.

NOT real traffic data and NOT the official IDD release. The 'objects' are
coloured rectangles drawn exactly where the annotation boxes are, so you can
see that the boxes are read correctly.

Run from the project root:
python make_sample_idd.py

Creates:
  data/idd/images/traffic_001.jpg
  data/idd/images/traffic_002.jpg
  data/idd/images/traffic_003.jpg
  data/idd/images/traffic_004.jpg
  data/idd/annotations/instances.json
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
image_dir = root / "data" / "idd" / "images"
ann_dir = root / "data" / "idd" / "annotations"
image_dir.mkdir(parents=True, exist_ok=True)
ann_dir.mkdir(parents=True, exist_ok=True)

categories = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 3, "name": "pedestrian", "supercategory": "person"},
    {"id": 4, "name": "bus", "supercategory": "vehicle"},
    {"id": 5, "name": "truck", "supercategory": "vehicle"},
]
images = [
    {"id": 1, "file_name": "traffic_001.jpg", "width": 1280, "height": 720},
    {"id": 2, "file_name": "traffic_002.jpg", "width": 1280, "height": 720},
    {"id": 3, "file_name": "traffic_003.jpg", "width": 1280, "height": 720},
    {"id": 4, "file_name": "traffic_004.jpg", "width": 1280, "height": 720},
]
# bbox = [x, y, width, height], x and y = top-left corner, in pixels
annotations = [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 180, 260, 150], "area": 39000, "iscrowd": 0},
    {"id": 2, "image_id": 1, "category_id": 3, "bbox": [500, 200, 60, 160], "area": 9600, "iscrowd": 0},
    {"id": 3, "image_id": 1, "category_id": 2, "bbox": [700, 300, 120, 140], "area": 16800, "iscrowd": 0},
    {"id": 4, "image_id": 2, "category_id": 4, "bbox": [200, 150, 400, 300], "area": 120000, "iscrowd": 0},
    {"id": 5, "image_id": 2, "category_id": 5, "bbox": [800, 250, 300, 200], "area": 60000, "iscrowd": 0},
    {"id": 6, "image_id": 3, "category_id": 1, "bbox": [100, 180, 260, 150], "area": 39000, "iscrowd": 0},
    {"id": 7, "image_id": 3, "category_id": 2, "bbox": [700, 300, 120, 140], "area": 16800, "iscrowd": 0},
    {"id": 8, "image_id": 3, "category_id": 4, "bbox": [200, 150, 400, 300], "area": 120000, "iscrowd": 0},
    {"id": 9, "image_id": 4, "category_id": 1, "bbox": [120, 260, 260, 160], "area": 41600, "iscrowd": 0},
    {"id": 10, "image_id": 4, "category_id": 2, "bbox": [440, 300, 100, 140], "area": 14000, "iscrowd": 0},
    {"id": 11, "image_id": 4, "category_id": 3, "bbox": [590, 220, 50, 150], "area": 7500, "iscrowd": 0},
    {"id": 12, "image_id": 4, "category_id": 4, "bbox": [700, 160, 350, 270], "area": 94500, "iscrowd": 0},
    {"id": 13, "image_id": 4, "category_id": 5, "bbox": [1080, 220, 180, 200], "area": 36000, "iscrowd": 0},
]

fill = {
    1: (41, 128, 185),   # car: vibrant blue
    2: (39, 174, 96),    # motorcycle: vivid green
    3: (230, 126, 34),   # pedestrian: amber orange
    4: (231, 76, 60),    # bus: coral red
    5: (142, 68, 173),   # truck: deep purple
}
for info in images:
    canvas = Image.new("RGB", (info["width"], info["height"]), (225, 225, 225))
    draw = ImageDraw.Draw(canvas)
    for a in annotations:
        if a["image_id"] == info["id"]:
            x, y, w, h = a["bbox"]
            draw.rectangle([x, y, x + w, y + h], fill=fill[a["category_id"]])
    canvas.save(image_dir / info["file_name"])

with open(ann_dir / "instances.json", "w", encoding="utf-8") as f:
    json.dump({"images": images, "categories": categories,
               "annotations": annotations}, f, indent=2)

print("Created 4 images and instances.json in data/idd/")
print("Reminder: this is synthetic teaching data, not the official IDD dataset.")
