# Smart Traffic Light GUI — YOLO26n 100 Epochs

Aplikasi GUI pemantauan lalu lintas dengan empat feed kamera, deteksi objek YOLO, dan simulasi kendali lampu lalu lintas.

## Pembaruan model dan deteksi

GUI sekarang menggunakan checkpoint **YOLO26n hasil training 100 epoch** dari repositori [Yolo26n_100ep](https://github.com/Varoo0jpg/Yolo26n_100ep).

- Sumber checkpoint: `weights/best.pt` pada repositori training.
- Nama lokal yang digunakan GUI: `GUI/100ep.pt`.
- Jika `100ep.pt` belum tersedia, aplikasi mengunduh checkpoint secara otomatis saat pertama dijalankan dan menyimpannya di folder GUI.
- Jika unduhan otomatis gagal, aplikasi menyediakan dialog untuk memilih file checkpoint secara manual.
- Variabel lingkungan `SMART_TRAFFIC_MODEL` dapat digunakan untuk menunjuk ke checkpoint lokal lain.

> File model tidak disalin ke repositori GUI sebagai file biner. Aplikasi mengambil checkpoint dari repositori training saat diperlukan, sehingga ukuran repositori GUI tetap lebih ringan.

## Sensitivitas deteksi

Panel **SENSITIVITAS DETEKSI** pada sisi kiri GUI menyediakan dua pengaturan yang diterapkan langsung pada proses inferensi:

| Pengaturan | Nilai awal | Rentang GUI | Dampak |
|---|---:|---:|---|
| Confidence | 0.15 | 0.05–0.80 | Nilai lebih rendah menerima lebih banyak kandidat deteksi, tetapi dapat menambah false positive. |
| NMS IoU | 0.60 | 0.30–0.90 | Nilai lebih tinggi membuat proses NMS lebih toleran terhadap kotak yang saling tumpang tindih. |

Nilai awal 0.15 dan 0.60 merupakan titik awal untuk pengujian yang lebih sensitif, bukan jaminan akurasi terbaik. Uji beberapa kondisi kamera dan bandingkan false positive serta objek yang terlewat sebelum menetapkan konfigurasi penelitian.

## Fitur

- Tampilan empat lajur: Utara, Timur, Selatan, dan Barat.
- Deteksi YOLO pada feed kamera/video dan input manual jika kamera tidak tersedia.
- Validasi jumlah serta urutan delapan kelas model sebelum inferensi.
- Pengaturan confidence dan NMS IoU secara langsung melalui slider GUI.
- Simulasi fase lampu hijau/kuning dan mode otomatis/manual.
- Panel data penelitian dengan pencatatan snapshot per detik untuk setiap lajur.
- Ekspor data observasi mentah dan ringkasan statistik per lajur ke CSV.

## Kelas yang dideteksi

| ID | Kelas |
|---:|---|
| 0 | Ambulance |
| 1 | Bus |
| 2 | Car |
| 3 | Fire engine |
| 4 | Military |
| 5 | Motorcycle |
| 6 | Police car |
| 7 | Truck |

## Struktur folder

```text
repository/
└── GUI/
    ├── smart_traffic_gui.py
    ├── 100ep.pt          # dibuat otomatis saat unduhan pertama
    ├── requirements.txt
    └── README.md
```

## Instalasi dan menjalankan

Jalankan dari direktori utama repository:

```bash
python -m pip install -r GUI/requirements.txt
python GUI/smart_traffic_gui.py
```

Koneksi internet diperlukan saat pertama kali aplikasi mengunduh model. Setelah file `GUI/100ep.pt` tersedia, aplikasi dapat memuatnya secara lokal.

## Menggunakan checkpoint lokal lain

Windows CMD:

```bat
set SMART_TRAFFIC_MODEL=C:\\path\\to\\model.pt
python GUI\\smart_traffic_gui.py
```

PowerShell:

```powershell
$env:SMART_TRAFFIC_MODEL = "C:\\path\\to\\model.pt"
python GUI\\smart_traffic_gui.py
```

## Konfigurasi kamera

Ubah dictionary `SUMBER_KAMERA` pada bagian konfigurasi di `smart_traffic_gui.py`. Nilai yang didukung:

- indeks kamera lokal, misalnya `0` atau `1`;
- path video lokal, misalnya `videos/utara.mp4`;
- URL video/live yang dapat dibaca resolver `yt-dlp`;
- `None` untuk memakai input manual pada lajur tersebut.

## Data penelitian

Klik **Mulai Rekam** untuk mengambil snapshot setiap 1 detik pada masing-masing lajur. Setiap baris merupakan satu snapshot, bukan identitas kendaraan unik. Karena itu, total objek menunjukkan jumlah kendaraan yang terlihat pada snapshot tersebut, bukan jumlah kendaraan unik selama keseluruhan sesi.

- **Ekspor Data Mentah** menghasilkan snapshot beserta waktu, lajur, jumlah tiap kelas, total objek, skor berbobot, dan status lampu.
- **Ekspor Ringkasan Paper** menghasilkan statistik per lajur: jumlah sampel, durasi observasi, rerata, simpangan baku, nilai maksimum, dan rerata/maksimum tiap kelas.

Skor berbobot menggunakan bobot heuristik internal GUI, bukan nilai PCU terstandar. Jika digunakan dalam laporan atau paper, jelaskan sebagai indeks internal dan dokumentasikan konfigurasi yang dipakai.

## Catatan evaluasi

Menurunkan confidence dapat meningkatkan recall, tetapi biasanya juga meningkatkan false positive. NMS IoU mengatur penyaringan kotak yang bertumpang tindih; parameter ini tidak meningkatkan kemampuan model mengenali kelas yang belum dipelajari. Untuk evaluasi yang dapat dibandingkan, catat nilai confidence, IoU, kondisi pencahayaan, posisi kamera, dan hasil pemeriksaan manual.
