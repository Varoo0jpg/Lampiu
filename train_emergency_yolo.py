#!/usr/bin/env python3
"""Train a Lampiu emergency-vehicle detector from Coba/dataset_darurat.

Run from the repository root:
    python train_emergency_yolo.py --epochs 50 --imgsz 640 --batch 8

The script uses Coba/yolo26n.pt as the pretrained starting point, trains the five
classes declared by dataset_darurat, evaluates the held-out test split, and writes
Coba/yolo26n_emergency_best.pt so Coba/Coba_kamera.py can load it automatically.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Coba" / "dataset_darurat"
INITIAL_WEIGHTS = ROOT / "Coba" / "yolo26n.pt"
OUTPUT_WEIGHTS = ROOT / "Coba" / "yolo26n_emergency_best.pt"
RUNS_DIR = ROOT / "Coba" / "runs_emergency"
METRICS_PATH = ROOT / "Coba" / "emergency_training_metrics.json"

CLASS_NAMES = [
    "Ambulance",
    "Fire_truck",
    "Normal_vehicle",
    "Police_car",
    "Towing_truck",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--workers", type=int, default=2, help="Data loader workers (default: 2)")
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', '0', or a supported PyTorch device (default: auto)",
    )
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    return parser.parse_args()


def validate_dataset() -> dict[str, Any]:
    if not INITIAL_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Pretrained starting weights not found: {INITIAL_WEIGHTS}. "
            "Make sure the repository includes Coba/yolo26n.pt."
        )

    splits = {}
    for split, image_folder in (
        ("train", DATASET_DIR / "train" / "images"),
        ("val", DATASET_DIR / "valid" / "images"),
        ("test", DATASET_DIR / "test" / "images"),
    ):
        label_folder = image_folder.parent / "labels"
        if not image_folder.is_dir() or not label_folder.is_dir():
            raise FileNotFoundError(
                f"Expected image and label folders for split '{split}': "
                f"{image_folder} and {label_folder}"
            )
        images = [p for p in image_folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        labels = list(label_folder.glob("*.txt"))
        if not images:
            raise RuntimeError(f"No images found in {image_folder}")
        if not labels:
            raise RuntimeError(f"No YOLO labels found in {label_folder}")
        splits[split] = {"images": len(images), "label_files": len(labels)}

    print("Dataset verified:")
    print(json.dumps(splits, indent=2))
    return {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }


def _metric_value(metrics: Any, attribute: str) -> float | None:
    value = getattr(metrics, attribute, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs, imgsz, and batch must be positive (imgsz >= 32).")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install them with: pip install ultralytics"
        ) from exc

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        workers = min(args.workers, 2)
    else:
        workers = args.workers

    data_config = validate_dataset()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting model: {INITIAL_WEIGHTS}")
    print(f"Device: {device}. Epochs: {args.epochs}. Image size: {args.imgsz}. Batch: {args.batch}.")

    model = YOLO(str(INITIAL_WEIGHTS))
    model.train(
        data=data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=workers,
        device=device,
        project=str(RUNS_DIR),
        name="lampiu_emergency_yolo26n",
        exist_ok=True,
        pretrained=True,
        patience=args.patience,
        save=True,
        plots=True,
        cache=False,
        seed=42,
        deterministic=True,
        amp=(device != "cpu"),
        verbose=True,
    )

    trainer = getattr(model, "trainer", None)
    best_path = Path(trainer.best) if trainer is not None else Path()
    if not best_path.is_file():
        raise FileNotFoundError(
            f"Training finished but best.pt was not found at {best_path}. "
            "Check the training log and output directory."
        )

    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, OUTPUT_WEIGHTS)
    print(f"Saved trained model: {OUTPUT_WEIGHTS}")

    best_model = YOLO(str(OUTPUT_WEIGHTS))
    test_metrics = best_model.val(
        data=data_config,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=workers,
        device=device,
        project=str(RUNS_DIR),
        name="test_evaluation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    box_metrics = getattr(test_metrics, "box", None)
    report = {
        "initial_weights": str(INITIAL_WEIGHTS.relative_to(ROOT)),
        "trained_weights": str(OUTPUT_WEIGHTS.relative_to(ROOT)),
        "dataset": str(DATASET_DIR.relative_to(ROOT)),
        "classes": CLASS_NAMES,
        "epochs_requested": args.epochs,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "device": device,
        "test_metrics": {
            "mAP50_95": _metric_value(box_metrics, "map"),
            "mAP50": _metric_value(box_metrics, "map50"),
            "mAP75": _metric_value(box_metrics, "map75"),
            "precision": _metric_value(box_metrics, "mp"),
            "recall": _metric_value(box_metrics, "mr"),
        },
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Test metrics:")
    print(json.dumps(report, indent=2))
    print(f"Metrics saved: {METRICS_PATH}")
    print("Training and held-out test evaluation completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"TRAINING FAILED: {error}", file=sys.stderr)
        raise
