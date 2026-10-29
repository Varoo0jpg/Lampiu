"""Fine-tune the existing 8-class Lampiu model with extra motorcycle examples.

Run from C:\\Smart Traffic Light\\Coba:
    python train_motorcycle_ft_v2.py

This script never overwrites the original model or previous fine-tuning run.
It starts from the original 8-class best.pt and uses a low learning rate.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset_8kelas_motorcycle_ft" / "dataset.yaml"
START_WEIGHTS = ROOT / "runs" / "detect" / "smart_traffic_8kelas" / "weights" / "best.pt"
OUTPUT_PROJECT = ROOT / "runs" / "detect"
RUN_NAME = "motorcycle_finetune_v2"

EXPECTED_NAMES = [
    "Ambulance",
    "Fire_truck",
    "Car",
    "Motorcycle",
    "Police_car",
    "Truck",
    "Towing_truck",
    "Bus",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def fail(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


def normalize(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def get_names(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def main() -> None:
    print("=== FINE-TUNING 8 KELAS: MOTORCYCLE V2 ===")
    print(f"Starting weights: {START_WEIGHTS}")
    print(f"Dataset YAML:     {DATA_YAML}")

    if not START_WEIGHTS.is_file():
        fail(
            "Bobot awal 8 kelas tidak ditemukan. Pastikan file asli masih ada di "
            "runs\\detect\\smart_traffic_8kelas\\weights\\best.pt"
        )
    if not DATA_YAML.is_file():
        fail(f"Dataset gabungan tidak ditemukan: {DATA_YAML}")
    run_dir = OUTPUT_PROJECT / RUN_NAME
    if run_dir.exists():
        fail(
            f"Folder output sudah ada: {run_dir}\n"
            "Agar hasil sebelumnya aman, skrip tidak menimpanya. "
            "Ubah RUN_NAME di bagian atas skrip jika memang ingin run baru."
        )

    with DATA_YAML.open("r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    dataset_names = get_names(cfg.get("names"))
    if int(cfg.get("nc", len(dataset_names))) != 8:
        fail(f"Dataset harus berisi 8 kelas, ditemukan nc={cfg.get('nc')}.")
    if [normalize(x) for x in dataset_names] != [normalize(x) for x in EXPECTED_NAMES]:
        fail(f"Urutan kelas dataset tidak cocok: {dataset_names}")

    train_images = DATA_YAML.parent / "train" / "images"
    train_labels = DATA_YAML.parent / "train" / "labels"
    valid_images = DATA_YAML.parent / "valid" / "images"
    valid_labels = DATA_YAML.parent / "valid" / "labels"

    for folder in (train_images, train_labels, valid_images, valid_labels):
        if not folder.is_dir():
            fail(f"Folder dataset tidak ditemukan: {folder}")

    images = [p for p in train_images.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        fail(f"Tidak ada gambar training di {train_images}")

    counts = Counter()
    invalid = []
    missing_labels = 0
    for image_path in images:
        label_path = train_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
            continue
        for line_no, line in enumerate(label_path.read_text(
                encoding="utf-8-sig").splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                invalid.append(f"{label_path.name}:{line_no} kolom={len(parts)}")
                continue
            try:
                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                invalid.append(f"{label_path.name}:{line_no} nilai bukan angka")
                continue
            if class_id < 0 or class_id > 7 or any(v < 0 or v > 1 for v in coords):
                invalid.append(f"{label_path.name}:{line_no} ID/koordinat di luar batas")
                continue
            counts[class_id] += 1

    print(f"Training images: {len(images)}")
    print(f"Images without label file: {missing_labels}")
    print("Jumlah objek per kelas pada label training:")
    for idx, name in enumerate(EXPECTED_NAMES):
        print(f"  {idx} {name}: {counts[idx]}")
    if invalid:
        fail(
            f"Ditemukan {len(invalid)} baris label tidak valid. Contoh: "
            + "; ".join(invalid[:5])
        )
    if counts[3] == 0:
        fail("Tidak ada objek Motorcycle (class ID 3) di label training.")
    if any(counts[i] == 0 for i in range(8)):
        fail("Ada kelas yang tidak memiliki objek training; periksa dataset gabungan.")

    print("\nMemuat bobot asli 8 kelas (bukan hasil fine-tuning yang terlalu fokus motor)...")
    model = YOLO(str(START_WEIGHTS))
    model_names = get_names(model.names)
    if [normalize(x) for x in model_names] != [normalize(x) for x in EXPECTED_NAMES]:
        fail(f"Urutan kelas bobot awal tidak cocok: {model.names}")

    print("\nKonfigurasi:")
    print("  epochs=30, imgsz=640, batch=4, device=cpu")
    print("  lr0=0.001 (lebih rendah), freeze=10 (membantu mempertahankan fitur dasar)")
    print("  output= runs\\detect\\motorcycle_finetune_v2")
    print("\nTraining dimulai. Dataset dan bobot awal tidak akan ditimpa.\n")

    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        patience=10,
        project=str(OUTPUT_PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2.0,
        cos_lr=True,
        freeze=10,
        close_mosaic=10,
        save=True,
        plots=True,
        seed=42,
        pretrained=False,
    )

    print("\nTraining selesai.")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")
    print("Uji bobot ini sebelum mengganti model pada GUI.")


if __name__ == "__main__":
    main()
