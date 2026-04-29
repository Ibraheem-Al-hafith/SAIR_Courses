"""
download_weights.py — Download pre-trained YOLO weights for Module 4 Vision demos.

The .pt model files are NOT stored in the SAIR git repo (binary blobs).
Run this script once before using the Demos/ scripts or the YOLO notebooks.

Usage:
    uv run python download_weights.py
    # or
    python download_weights.py
"""
from ultralytics import YOLO
import pathlib, sys

SCRIPT_DIR = pathlib.Path(__file__).parent

MODELS = [
    ("yolov8n.pt",      "YOLOv8 Nano        — fast, lightweight, good for demos"),
    ("yolov8s.pt",      "YOLOv8 Small       — better accuracy, still real-time"),
    ("yolov8m.pt",      "YOLOv8 Medium      — used in the model comparison demo"),
    ("yolov8n-seg.pt",  "YOLOv8 Nano Seg    — instance segmentation"),
    ("yolov8n-pose.pt", "YOLOv8 Nano Pose   — keypoint / pose estimation"),
]

def download_all():
    print("Downloading YOLOv8 pre-trained weights...\n")
    for model_name, description in MODELS:
        dest = SCRIPT_DIR / "Demos" / model_name
        if dest.exists():
            print(f"  [skip] {model_name:25s} already exists")
            continue
        print(f"  [download] {model_name:25s} — {description}")
        try:
            model = YOLO(model_name)           # downloads to ultralytics cache
            model.export(format="pt")          # ensures local copy is valid
            # Move from cwd to Demos/ if ultralytics saved it here
            local = pathlib.Path(model_name)
            if local.exists() and not dest.exists():
                local.rename(dest)
            print(f"             saved → {dest.relative_to(SCRIPT_DIR)}")
        except Exception as e:
            print(f"  [error] {model_name}: {e}", file=sys.stderr)

    print("\nDone. All weights are in Demos/")


if __name__ == "__main__":
    download_all()
