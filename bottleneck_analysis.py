"""VisionPort Sessions 6-8, Part C - bottleneck analysis (Session 8)

Run:
  python bottleneck_analysis.py docs/sample_trace_cpu.json
  python bottleneck_analysis.py docs/sample_trace_cpu.json docs/sample_trace_gpu_example.json
  python bottleneck_analysis.py docs/trace_runs.json --price 1.0

Reads one or more trace files (from the Session 3 /trace endpoint), puts every
stage into a category, shows where the time goes, and says which stage to
profile first. Writes docs/bottleneck_report.md and docs/latency_breakdown.png.
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 0. Options --------------------------------------------------------------
parser = argparse.ArgumentParser(description="Find the bottleneck in a request trace")
parser.add_argument("traces", nargs="+", help="one or more trace JSON files")
parser.add_argument("--price", type=float, default=None,
                    help="price of the machine per hour (you look it up); enables the cost calculation")
parser.add_argument("--keep-first", action="store_true",
                    help="keep the first run of a multi-run file (it normally includes warm-up)")
parser.add_argument("--out", default="docs", help="folder for the report and chart")
args = parser.parse_args()
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- 1. Categories -----------------------------------------------------------
CATEGORIES = ["Serving / I/O", "CPU preprocessing", "Memory transfer", "Compute", "Postprocessing"]
COLOURS = {"Serving / I/O": "#888780", "CPU preprocessing": "#378ADD",
           "Memory transfer": "#EF9F27", "Compute": "#1D9E75", "Postprocessing": "#7F77DD"}
BY_NAME = {
    "receive upload": "Serving / I/O",
    "decode image": "CPU preprocessing",
    "letterbox resize": "CPU preprocessing",
    "to tensor": "CPU preprocessing",
    "host to device": "Memory transfer",
    "model forward": "Compute",
    "NMS": "Postprocessing",
    "rescale boxes": "Postprocessing",
    "serialise response": "Serving / I/O",
}
KEYWORDS = [("receive", "Serving / I/O"), ("serialis", "Serving / I/O"),
            ("decode", "CPU preprocessing"), ("letterbox", "CPU preprocessing"),
            ("tensor", "CPU preprocessing"), ("device", "Memory transfer"),
            ("forward", "Compute"), ("nms", "Postprocessing"), ("rescale", "Postprocessing")]


def classify(name):
    """Exact name first, then keywords; unknown stages are reported, not hidden."""
    if name in BY_NAME:
        return BY_NAME[name]
    for word, category in KEYWORDS:
        if word in name.lower():
            return category
    return "Unclassified"


report = []


def say(line=""):
    print(line)
    report.append(line)


def load_runs(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


# ---- 2. Analyse one trace file ----------------------------------------------
def analyse(path):
    runs = load_runs(path)
    skipped = 0
    if len(runs) > 1 and not args.keep_first:
        runs, skipped = runs[1:], 1                  # the first run includes warm-up
    first = runs[0]
    ms_by_stage = {}
    for run in runs:
        for s in run["stages"]:
            ms_by_stage.setdefault(s["name"], []).append(s["ms"])
    rows = [(name, classify(name), statistics.median(values)) for name, values in ms_by_stage.items()]
    total = sum(ms for _, _, ms in rows)
    recorded = statistics.median([r["total_ms"] for r in runs if "total_ms" in r] or [total])

    say(f"\n=== {path} ===")
    say(f"Source : {first.get('source', 'not stated')}")
    say(f"Device : {first.get('device', '?')} | runs used: {len(runs)}"
        + (f" (first run skipped as warm-up)" if skipped else ""))
    say(f"\n{'#':>2}  {'stage':<20}{'category':<19}{'ms':>8}  {'share':>6}")
    for i, (name, category, ms) in enumerate(rows, 1):
        share = ms / total
        say(f"{i:>2}  {name:<20}{category:<19}{ms:>8.2f}  {share:>6.1%}  {'#' * round(share * 40)}")
    say(f"\nSum of stage times: {total:.2f} ms (recorded total {recorded:.2f} ms)")

    by_category = {c: sum(ms for _, cat, ms in rows if cat == c) for c in CATEGORIES + ["Unclassified"]}
    say("\nBy category:")
    for category in CATEGORIES + ["Unclassified"]:
        ms = by_category[category]
        if ms == 0 and category == "Unclassified":
            continue
        say(f"  {category:<19}{ms:>8.2f} ms  {ms / total:>6.1%}  {'#' * round(ms / total * 40)}")

    # ---- the bottleneck ----
    name, category, ms = max(rows, key=lambda r: r[2])
    say(f"\nLargest stage: {name} ({category}), {ms:.2f} ms = {ms / total:.1%} of the measured time")
    say(f"PROFILE FIRST: {name}. Any saving in the largest stage has the largest possible effect.")

    # ---- rules of thumb ----
    hints = []
    if first.get("device") == "cpu":
        hints.append("This ran on the CPU, so 'host to device' costs almost nothing. On a GPU machine "
                     "it will not be zero, and the compute stage should shrink.")
    if by_category["CPU preprocessing"] / total > 0.25:
        hints.append("CPU preprocessing is over 25% of the time. A fast GPU would sit waiting for it. "
                     "Look at decoding, resizing and tensor conversion before tuning the model.")
    if by_category["Memory transfer"] / total > 0.10:
        hints.append("Memory transfer is over 10% of the time. Check for avoidable copies.")
    if by_category["Compute"] / total > 0.50:
        hints.append("Compute is over half of the time. Faster precision (FP16, INT8), TensorRT and "
                     "a GPU are the levers (Sessions 13 and 14).")
    if by_category["Postprocessing"] / total > 0.15:
        hints.append("Postprocessing is over 15% of the time. NMS cost grows with the number of "
                     "candidate boxes.")
    if len(runs) == 1 and not skipped:
        hints.append("This is a single run, so it includes warm-up. Use collect_trace.py to measure "
                     "repeated runs.")
    hints.append("Not measured here: time waiting in a queue and network time. They appear when a "
                 "server such as Triton is added.")
    say("\nRules of thumb applied to this trace:")
    for h in hints:
        say(f"  - {h}")

    # ---- what if the largest stage were faster? (Amdahl) ----
    say(f"\nWhat if '{name}' were faster? (other stages unchanged)")
    for factor in (2, 10):
        new_total = total - ms + ms / factor
        say(f"  {factor:>2}x faster -> {new_total:7.2f} ms total, overall speed-up {total / new_total:.2f}x")
    floor = total - ms
    say(f"  free (0 ms)  -> {floor:7.2f} ms total, overall speed-up {total / floor:.2f}x  <- the limit")

    # ---- throughput and cost ----
    per_second = 1000 / total
    say(f"\nThroughput for one request at a time: {per_second:.2f} images per second")
    if args.price:
        per_price = per_second * 3600 / args.price
        say(f"At a price of {args.price} per hour: {per_second:.2f} x 3600 / {args.price} "
            f"= {per_price:,.0f} images per unit of price")
    return {"path": path, "rows": rows, "total": total, "by_category": by_category,
            "device": first.get("device", "?")}


results = [analyse(p) for p in args.traces]

# ---- 3. Compare traces -------------------------------------------------------
if len(results) > 1:
    say("\n=== COMPARISON: share of time by category ===")
    say(f"{'category':<19}" + "".join(f"{Path(r['path']).stem[:18]:>20}" for r in results))
    for category in CATEGORIES:
        say(f"{category:<19}" + "".join(f"{r['by_category'][category] / r['total']:>20.1%}" for r in results))
    say(f"{'total (ms)':<19}" + "".join(f"{r['total']:>20.2f}" for r in results))
    say("\nThe bottleneck moves when the hardware changes. Optimise the stage that is largest on the "
        "machine you will actually serve from.")

# ---- 4. Chart ------------------------------------------------------------------
fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 4.2), squeeze=False)
for ax, r in zip(axes[0], results):
    names = [name for name, _, _ in r["rows"]][::-1]
    values = [ms for _, _, ms in r["rows"]][::-1]
    colours = [COLOURS.get(cat, "#cccccc") for _, cat, _ in r["rows"]][::-1]
    ax.barh(names, values, color=colours)
    for y, v in enumerate(values):
        ax.text(v, y, f" {v:.2f}" if v < 1 else f" {v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, max(values) * 1.15)                 # room for the value labels
    ax.set_title(f"{Path(r['path']).stem} ({r['device']}, {r['total']:.0f} ms)", fontsize=10)
    ax.set_xlabel("milliseconds")
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[c]) for c in CATEGORIES]
fig.legend(handles, CATEGORIES, loc="lower center", ncol=5, fontsize=8, frameon=False)
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig(out_dir / "latency_breakdown.png", dpi=110)

(out_dir / "bottleneck_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print(f"\nSaved: {out_dir / 'bottleneck_report.md'} and {out_dir / 'latency_breakdown.png'}")
