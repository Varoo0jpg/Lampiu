# Lampiu — Smart Traffic Light Simulation

Aplikasi simulasi pengaturan lampu lalu lintas empat lajur berbasis **YOLO26n** dan input audio MP3. GUI dibuat dengan CustomTkinter. Proyek ini ditujukan untuk eksperimen, demonstrasi, dan pengumpulan data penelitian—**bukan pengendali lampu lalu lintas untuk dipasang langsung di jalan umum**.

## Fitur utama

- Simulasi empat lajur: Utara, Timur, Selatan, dan Barat.
- Mode kendali **Otomatis** dan **Manual**.
- Deteksi delapan kelas kendaraan menggunakan checkpoint YOLO26n:
  - Ambulance
  - Bus
  - Car
  - Fire engine
  - Military
  - Motorcycle
  - Police car
  - Truck
- Tampilan feed kamera dan penghitungan kendaraan per lajur.
- Pengaturan confidence dan NMS IoU dari GUI.
- Skor kepadatan berbobot untuk membantu memilih lajur berikutnya.
- Prioritas simulasi bertingkat:
  1. Kendaraan darurat terdeteksi dan indikasi sirene pada lajur yang sama.
  2. Kendaraan darurat terdeteksi.
  3. Indikasi sirene dari analisis MP3.
  4. Ambang kepadatan minimal **2 Car dan 2 Truck**.
  5. Anti-starvation dan skor kepadatan biasa.
- Transisi lampu melewati fase kuning sebelum berpindah lajur.
- Input file MP3 untuk simulasi pengujian audio; pengguna memilih lajur yang akan diasosiasikan dengan file tersebut.
- Perekaman snapshot data penelitian per detik serta ekspor CSV mentah dan ringkasan.

## Status deteksi audio

**Penting:** versi `GUI/smart_traffic_gui_100.py` saat ini menggunakan analisis heuristik berbasis fitur audio melalui `librosa` (energi pita frekuensi dan perubahan puncak spektral). Ini **belum menggunakan model klasifikasi sirene terlatih**, termasuk YAMNet. Nilai yang ditampilkan sebagai confidence merupakan skor indikasi heuristik, bukan probabilitas terkalibrasi bahwa audio benar-benar sirene.

Suara bernada atau berulang—misalnya klakson, alarm, musik, atau bunyi mesin—dapat menghasilkan indikasi keliru. Ambang `AMBANG_SIRENE` adalah parameter awal yang perlu diuji dengan audio positif dan negatif yang representatif sebelum menarik kesimpulan kinerja.

MP3 dipasangkan ke lajur melalui pilihan pengguna pada GUI. Aplikasi tidak menentukan lokasi sirene secara otomatis dari satu rekaman audio umum.

## Struktur file yang relevan

```text
Lampiu/
└── GUI/
    └── smart_traffic_gui_100.py
```

File utama yang dijalankan adalah `GUI/smart_traffic_gui_100.py`.

## Persyaratan

- Python 3.10 atau versi yang kompatibel dengan versi pustaka yang dipasang.
- Kamera USB atau file video lokal (opsional; sumber kamera bawaan belum diisi).
- Koneksi internet pada proses awal jika checkpoint YOLO26n belum tersedia secara lokal.
- MP3 untuk pengujian audio (opsional).

## Instalasi

Jalankan dari direktori utama repository:

```bash
python -m pip install --upgrade pip
python -m pip install customtkinter pillow opencv-python ultralytics numpy librosa yt-dlp
```

`yt-dlp` hanya diperlukan jika Anda mengonfigurasi sumber kamera berupa URL YouTube. Untuk kamera USB atau file video lokal, paket tersebut tidak diperlukan oleh jalur sumber tersebut.

## Menjalankan aplikasi

Dari direktori utama repository:

```bash
python GUI/smart_traffic_gui_100.py
```

Saat dijalankan, aplikasi mencoba menyiapkan checkpoint YOLO26n 100-epoch secara otomatis dari:

`https://raw.githubusercontent.com/Varoo0jpg/Yolo26n_100ep/main/weights/best.pt`

Checkpoint disimpan sebagai `100ep.pt` di lokasi file aplikasi. Jika pengunduhan otomatis gagal atau file tidak tersedia, GUI meminta pengguna memilih file checkpoint secara manual.

Checkpoint harus memiliki **tepat delapan kelas** dengan urutan nama yang sesuai dengan konfigurasi aplikasi. Aplikasi memvalidasi nama kelas saat model dimuat.

Untuk menggunakan checkpoint di lokasi khusus, atur environment variable sebelum menjalankan aplikasi:

**Windows PowerShell**
```powershell
$env:SMART_TRAFFIC_MODEL="C:/path/to/best.pt"
python GUI/smart_traffic_gui_100.py
```

## Konfigurasi kamera

Secara default, seluruh `SUMBER_KAMERA` bernilai `None`. Artinya, feed kamera tidak akan aktif sampai sumber diatur sendiri pada bagian konfigurasi di awal file `GUI/smart_traffic_gui_100.py`.

Contoh kamera USB:

```python
SUMBER_KAMERA = {
    "Utara": 0,
    "Timur": 1,
    "Selatan": 2,
    "Barat": 3,
}
```

Contoh file video lokal:

```python
SUMBER_KAMERA = {
    "Utara": "videos/utara.mp4",
    "Timur": "videos/timur.mp4",
    "Selatan": "videos/selatan.mp4",
    "Barat": "videos/barat.mp4",
}
```

Indeks kamera, path video, dan URL stream harus disesuaikan dengan perangkat serta sumber yang benar-benar tersedia. Pastikan setiap kamera mengarah ke lajur yang sesuai. ROI bawaan tidak aktif (`GUNAKAN_ROI = False`); jika ROI diaktifkan, koordinat poligon perlu dikalibrasi untuk masing-masing kamera.

## Cara menggunakan fitur audio MP3

1. Buka aplikasi dan pilih file MP3 melalui tombol **Pilih MP3**.
2. Pilih lajur yang ingin diasosiasikan dengan rekaman tersebut.
3. Jalankan **Analisis Sirene**.
4. Periksa skor dan status yang ditampilkan pada panel audio.
5. Dalam mode Otomatis, indikasi audio yang melewati ambang dapat memengaruhi pemilihan lajur berikutnya.

Fitur ini adalah simulasi berbasis file, bukan pemantauan audio real-time dan bukan pelokasian sirene otomatis.

## Logika simulasi dan batasan

- Prioritas hanya dievaluasi untuk kandidat lajur berikutnya, bukan memotong fase hijau yang sedang berjalan secara langsung.
- Perpindahan lajur tetap melewati fase kuning.
- Skor kendaraan adalah bobot heuristik internal aplikasi, **bukan standar Passenger Car Unit (PCU) yang telah divalidasi**.
- Ambang 2 mobil dan 2 truk merupakan aturan pemicu simulasi yang dapat disesuaikan, bukan standar rekayasa lalu lintas.
- Satu input audio MP3 tidak dapat membuktikan dari lajur mana sirene berasal. Pemilihan lajur dilakukan manual untuk kebutuhan simulasi.
- Deteksi YOLO dapat dipengaruhi sudut kamera, pencahayaan, oklusi, jarak, dan kualitas model.
- Tidak ada klaim bahwa sistem telah memenuhi standar keselamatan, keandalan, atau regulasi pengendali lalu lintas.

## Data penelitian

Panel **DATA PENELITIAN** menyediakan:
- Perekaman snapshot berkala (interval bawaan 1 detik).
- Ekspor CSV mentah.
- Ekspor CSV ringkasan.
- Reset data yang direkam.

Periksa isi dan kelengkapan CSV sebelum digunakan untuk analisis. Data simulasi, input manual, dan hasil kamera perlu dibedakan saat menyusun metodologi penelitian.

## Pengembangan berikutnya

Rencana pengembangan audio dapat mencakup integrasi model audio terlatih seperti YAMNet, pengujian pada dataset sirene dan non-sirene, kalibrasi threshold, serta evaluasi false positive dan false negative. Fitur tersebut **belum diimplementasikan pada file saat ini**.

## Status pengujian

README ini mendokumentasikan perilaku dan konfigurasi yang tampak pada kode saat penulisan. Keberhasilan instalasi, kompatibilitas perangkat, performa inferensi, dan ketepatan deteksi tetap perlu diverifikasi melalui pengujian pada lingkungan pengguna.
