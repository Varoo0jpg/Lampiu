"""Train a custom five-class emergency-vehicle detector for Lampiu.

Run from the repository root:
    python Coba/train_darurat.py --epochs 5
    python Coba/train_darurat.py --epochs 50

The script uses Coba/yolo26n.pt as the pretrained starting point and writes
training outputs under Coba/runs/detect/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
DATA_YAML = HERE / "dataset_darurat" / "data.yaml"
BASE_MODEL = HERE / "yolo26n.pt"
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_dataset() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML tidak ditemukan: {DATA_YAML}")
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Model awal tidak ditemukan: {BASE_MODEL}")

    root = DATA_YAML.parent
    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Folder split '{split}' harus memiliki images/ dan labels/: "
                f"{image_dir} | {label_dir}"
            )

        images = [p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        labels = list(label_dir.glob("*.txt"))
        if not images:
            raise RuntimeError(f"Tidak ada gambar di {image_dir}")
        if not labels:
            raise RuntimeError(f"Tidak ada label .txt di {label_dir}")
        print(f"{split}: {len(images)} gambar, {len(labels)} file label")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training YOLO untuk lima kelas kendaraan Lampiu."
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Jumlah epoch (default: 50; mulai dengan 5 untuk uji).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Ukuran gambar (default: 640).")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size; default 8 untuk GPU dan 4 untuk CPU.")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', atau indeks GPU seperti '0'.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Jumlah worker data loader (default: 2).")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs harus minimal 1")

    check_dataset()
    has_cuda = torch.cuda.is_available()
    device = ("0" if has_cuda else "cpu") if args.device == "auto" else args.device
    batch = args.batch if args.batch is not None else (8 if device != "cpu" else 4)

    print(f"Model awal : {BASE_MODEL}")
    print(f"Dataset    : {DATA_YAML}")
    print(f"Perangkat  : {device}")
    print(f"Epoch      : {args.epochs}")
    print(f"Batch      : {batch}")

    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=args.workers,
        project=str(HERE / "runs" / "detect"),
        name="lampiu_darurat",
        save=True,
        save_period=5,
        plots=True,
        exist_ok=True,
    )

    print("\nTraining selesai. Periksa folder:")
    print(HERE / "runs" / "detect" / "lampiu_darurat" / "weights")
    print("best.pt = bobot terbaik; last.pt = bobot epoch terakhir.")


if __name__ == "__main__":
    main()
