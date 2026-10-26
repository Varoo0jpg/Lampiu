# Training model kendaraan darurat Lampiu (Windows)

Panduan ini melatih model YOLO lima kelas dari dataset yang ada di repo. Training belum dijalankan oleh GitHub; jalankan perintah berikut di PC sendiri.

## Struktur yang dipakai

- `Coba/yolo26n.pt`: bobot awal pretrained.
- `Coba/dataset_darurat/data.yaml`: konfigurasi dataset dengan path relatif.
- `Coba/dataset_darurat/{train,valid,test}/images`: gambar.
- `Coba/dataset_darurat/{train,valid,test}/labels`: anotasi YOLO.
- `Coba/train_darurat.py`: skrip training.
- `Coba/runs/detect/lampiu_darurat/weights/best.pt`: bobot terbaik hasil training (dibuat setelah training).

## 1. Buka PowerShell di root repo

Contoh:
```powershell
cd C:\Users\alvar\Downloads\Coba
.\.venv\Scripts\Activate.ps1
```

Pastikan dependency tersedia:
```powershell
python -m pip show ultralytics torch
```

Jika Ultralytics belum ada:
```powershell
python -m pip install ultralytics
```

Periksa GPU NVIDIA/CUDA:
```powershell
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Jika CUDA tidak tersedia, skrip otomatis menggunakan CPU. Training CPU dapat jauh lebih lambat.

## 2. Jalankan uji singkat

Mulai dengan 5 epoch untuk memastikan dataset, label, dan environment bekerja:
```powershell
python .\Coba\train_darurat.py --epochs 5
```

Untuk mengubah batch size, misalnya:
```powershell
python .\Coba\train_darurat.py --epochs 5 --batch 4
```

## 3. Training lebih lama

Setelah uji singkat berhasil, jalankan:
```powershell
python .\Coba\train_darurat.py --epochs 50
```

Untuk mengurangi beban GPU, coba `--batch 4` atau `--imgsz 512`. Jika terjadi CUDA out of memory, hentikan proses dan turunkan batch size terlebih dahulu.

## 4. Hasil training

Periksa:
```
Coba/runs/detect/lampiu_darurat/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
└── args.yaml
```

`best.pt` adalah bobot terbaik selama training. Pastikan file tersebut ada sebelum integrasi ke aplikasi.

## 5. Penting sebelum mengganti model aplikasi

Aplikasi `Coba/Coba_kamera.py` saat ini memetakan kelas COCO (mobil, motor, bus, truk). Model khusus ini punya lima indeks kelas berbeda:

0 Ambulance, 1 Fire_truck, 2 Normal_vehicle, 3 Police_car, 4 Towing_truck.

Karena itu, **jangan hanya mengganti nama file model**. Pemetaan kelas, penghitungan kendaraan, dan validasi model di aplikasi harus diperbarui secara konsisten terlebih dahulu. Pertahankan bobot COCO lama sebagai fallback sampai model khusus berhasil dievaluasi.

Dataset dan model berukuran besar tidak perlu ditambahkan lagi ke commit hasil training; simpan bobot hasil training lokal atau di penyimpanan model khusus.
