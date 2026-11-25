"""VisionPort Sessions 6-8, Part C - collect repeated traces (Session 8)

Start the Session 3 trace server first (in another terminal):
    uvicorn serve.gateway_trace:app --reload

Then run:
    python collect_trace.py --image data/idd/images/traffic_001.jpg
    python collect_trace.py --image my_photo.png --runs 20 --out docs/trace_runs.json

It sends the same image several times and saves every response. The FIRST
request includes warm-up, so the analysis script skips it by default.
"""
import argparse
import json
import statistics
from pathlib import Path

import requests

parser = argparse.ArgumentParser(description="Collect repeated /trace responses")
parser.add_argument("--image", required=True, help="image to send (JPEG or PNG)")
parser.add_argument("--url", default="http://127.0.0.1:8000/trace", help="address of /trace")
parser.add_argument("--runs", type=int, default=10, help="how many times to send it")
parser.add_argument("--out", default="docs/trace_runs.json", help="where to save the results")
args = parser.parse_args()

image_path = Path(args.image)
if not image_path.exists():
    raise SystemExit(f"Image not found: {image_path}")
content_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

runs = []
for i in range(1, args.runs + 1):
    with open(image_path, "rb") as f:
        try:
            response = requests.post(args.url, files={"file": (image_path.name, f, content_type)},
                                     timeout=120)
        except requests.exceptions.ConnectionError:
            raise SystemExit(f"Could not connect to {args.url}. Is the trace server running?")
    if response.status_code != 200:
        raise SystemExit(f"Run {i} failed with status {response.status_code}: {response.text[:200]}")
    trace = response.json()
    runs.append(trace)
    print(f"run {i:>2}/{args.runs}: total {trace['total_ms']:>8.2f} ms")

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")

totals = [t["total_ms"] for t in runs]
print(f"\nSaved {len(runs)} runs to {out_path}")
print(f"First run (includes warm-up): {totals[0]:.2f} ms")
if len(totals) > 1:
    rest = totals[1:]
    print(f"Later runs: min {min(rest):.2f} | median {statistics.median(rest):.2f} | max {max(rest):.2f} ms")
