#!/usr/bin/env python3
"""Build a small low-light car/motorcycle YOLO dataset from an extracted source dataset.

Expected source layout (YOLO):
  SOURCE_ROOT/
    data.yaml
    train|valid|val|test/images/*
    train|valid|val|test/labels/*.txt

The script does not invent annotations. It copies existing boxes, remaps the selected
car and motorcycle classes to IDs 0 and 1, and skips truck/other classes.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_class_names(yaml_path: Path) -> list[str]:
    text = yaml_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^\s*names\s*:\s*\[(.*?)\]\s*$", text)
    if match:
        return [x.strip().strip("'\"") for x in match.group(1).split(",")]
    match = re.search(r"(?ms)^\s*names\s*:\s*\n((?:\s+\d+\s*:\s*[^\n]+\n?)+)", text)
    if match:
        pairs = re.findall(r"(\d+)\s*:\s*([^\n]+)", match.group(1))
        return [name.strip().strip("'\"") for _, name in sorted(pairs, key=lambda p: int(p[0]))]
    raise ValueError(f"Tidak bisa membaca daftar class dari {yaml_path}")


def normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def locate_split_dirs(root: Path):
    for split in ("train", "valid", "val", "test"):
        images = root / split / "images"
        labels = root / split / "labels"
        if images.is_dir() and labels.is_dir():
            yield images, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Folder dataset low-light yang sudah diekstrak")
    parser.add_argument("--output", type=Path, default=Path("Coba/dataset_malam"))
    parser.add_argument("--count", type=int, default=100, help="Jumlah gambar target (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    yaml_path = args.source / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"data.yaml tidak ditemukan: {yaml_path}")

    names = read_class_names(yaml_path)
    normalized = [normalize(x) for x in names]
    car_ids = {i for i, n in enumerate(normalized) if n in {"car", "cars", "automobile", "automobiles"}}
    moto_ids = {i for i, n in enumerate(normalized) if n in {"motorcycle", "motorcycles", "motorbike", "motorbikes"}}
    if not car_ids or not moto_ids:
        raise SystemExit(f"Class car/motorcycle tidak ditemukan. Class sumber: {names}")

    records = []
    for image_dir, label_dir in locate_split_dirs(args.source):
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            rows = []
            present = set()
            for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                try:
                    cls = int(float(fields[0]))
                    coords = [float(v) for v in fields[1:5]]
                except ValueError:
                    continue
                if cls in car_ids:
                    rows.append("0 " + " ".join(f"{v:.6f}" for v in coords))
                    present.add("car")
                elif cls in moto_ids:
                    rows.append("1 " + " ".join(f"{v:.6f}" for v in coords))
                    present.add("motorcycle")
            if rows:
                records.append((image_path, label_path, rows, present))

    if len(records) < args.count:
        raise SystemExit(f"Hanya ada {len(records)} gambar berlabel mobil/motor; dibutuhkan {args.count}.")

    # Round-robin preference keeps both target classes represented when possible.
    rng = random.Random(args.seed)
    car_only = [r for r in records if "car" in r[3] and "motorcycle" not in r[3]]
    moto_only = [r for r in records if "motorcycle" in r[3] and "car" not in r[3]]
    mixed = [r for r in records if len(r[3]) == 2]
    rng.shuffle(car_only)
    rng.shuffle(moto_only)
    rng.shuffle(mixed)
    selected = []
    pools = [car_only, moto_only, mixed]
    while len(selected) < args.count and any(pools):
        for pool in pools:
            if pool and len(selected) < args.count:
                selected.append(pool.pop())

    out_images = args.output / "images"
    out_labels = args.output / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    manifest = ["filename,classes,source_image"]
    for i, (image_path, _label_path, rows, present) in enumerate(selected, 1):
        target_name = f"night_vehicle_{i:03d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, out_images / target_name)
        (out_labels / f"night_vehicle_{i:03d}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest.append(f"{target_name},{'|'.join(sorted(present))},{image_path.as_posix()}")

    (args.output / "manifest.csv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (args.output / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\ntest: images\n"
        "nc: 2\nnames: ['car', 'motorcycle']\n",
        encoding="utf-8",
    )
    print(f"Selesai: {len(selected)} gambar disalin ke {args.output}")
    print("Periksa manifest.csv dan tinjau sudut kamera secara visual; skrip tidak dapat memastikan POV dari metadata saja.")


if __name__ == "__main__":
    main()
