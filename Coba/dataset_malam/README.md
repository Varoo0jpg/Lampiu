# Dataset kendaraan malam — Lampiu

Folder ini disiapkan untuk subset sekitar 100 gambar **mobil dan sepeda motor** dalam kondisi malam/low-light, dengan anotasi YOLO dan sudut pandang CCTV jalan yang diutamakan.

## Sumber data

Gunakan **L-VAD (Low-Light Vehicle & Annotation Dataset), versi 2**, yang menyediakan anotasi YOLO untuk mobil, sepeda motor, dan truk dalam kondisi malam/low-light. Lisensi yang tercantum pada halaman sumber adalah **CC BY 4.0**. Cantumkan atribusi sumber jika dataset digunakan atau dibagikan.

- Halaman dataset: https://data.mendeley.com/datasets/h6p2w53my5/2
- DOI: https://doi.org/10.17632/h6p2w53my5.2

## Membuat subset 100 gambar

1. Unduh dan ekstrak dataset sumber secara manual dari halaman di atas.
2. Pastikan folder hasil ekstraksi memiliki `data.yaml` dan struktur YOLO seperti `train/images`, `train/labels`, `valid/images`, `valid/labels` (nama split `val` atau `test` juga didukung).
3. Dari root repo, jalankan:

   ```bash
   python Coba/dataset_malam/prepare_samples.py --source "PATH/KE/DATASET_L-VAD" --output Coba/dataset_malam --count 100
   ```

Skrip menyalin gambar dan label yang sudah tersedia, mengubah ID class menjadi `0=car` dan `1=motorcycle`, serta membuat `manifest.csv`. Anotasi tidak dibuat secara otomatis atau ditebak.

## Kriteria kurasi CCTV

Setelah subset dibuat, tinjau gambar secara visual dan prioritaskan:
- kamera tinggi dari atas / serong atas seperti CCTV persimpangan;
- malam, minim cahaya, silau lampu, bayangan, dan kendaraan bergerak;
- mobil dan sepeda motor dalam ukuran, jarak, serta kepadatan berbeda.

**Catatan status:** file gambar belum disertakan di commit ini karena arsip sumber perlu diunduh terlebih dahulu. Folder ini berisi prosedur untuk membentuk subset berlabel dari sumber berlisensi; jangan menganggap 100 gambar sudah tersedia sebelum skrip dijalankan dan hasilnya diperiksa.
