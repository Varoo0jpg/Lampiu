# Smart Traffic Light GUI — YOLO26n 100 Epochs

Dokumentasi untuk aplikasi pada folder ini, terutama `smart_traffic_gui_100.py`. Aplikasi dibuat dengan CustomTkinter dan ditujukan untuk simulasi, eksperimen, serta pengumpulan data penelitian. Aplikasi ini **bukan pengendali lampu lalu lintas yang siap dipasang pada jalan umum**.

## File utama

| File | Fungsi |
|---|---|
| `smart_traffic_gui_100.py` | GUI simulasi empat lajur dengan deteksi YOLO26n 100 epoch, prioritas kendaraan, dan input audio MP3 |
| `smart_traffic_gui.py` | Versi GUI sebelumnya; tidak memuat seluruh fitur tambahan pada versi 100 epoch |
| `requirements.txt` | Dependensi dasar proyek GUI |

Jalankan versi terbaru yang didokumentasikan di sini:

```bash
python GUI/smart_traffic_gui_100.py
```

## Model YOLO26n 100 epoch

Aplikasi menggunakan checkpoint hasil training pada repositori [Yolo26n_100ep](https://github.com/Varoo0jpg/Yolo26n_100ep).

- Sumber checkpoint: `weights/best.pt` pada repositori training.
- Nama file lokal default: `100ep.pt`, disimpan di folder `GUI`.
- Jika checkpoint belum ada, aplikasi mencoba mengunduhnya otomatis dari repositori training.
- Jika unduhan gagal, pengguna dapat memilih file checkpoint `.pt` secara manual melalui dialog aplikasi.
- Variabel lingkungan `SMART_TRAFFIC_MODEL` dapat digunakan untuk menunjuk ke checkpoint lokal lain.

Model harus memiliki tepat delapan kelas dengan urutan berikut. Aplikasi memvalidasi jumlah dan nama kelas sebelum inferensi.

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

## Instalasi

Jalankan dari direktori utama repository:

```bash
python -m pip install --upgrade pip
python -m pip install -r GUI/requirements.txt
python -m pip install librosa
```

Dependensi `librosa` diperlukan untuk analisis audio MP3 pada `smart_traffic_gui_100.py`. Untuk menghindari pemasangan ke interpreter Python yang berbeda, gunakan interpreter virtual environment yang sama untuk instalasi dan menjalankan aplikasi.

Contoh Windows (sesuaikan path virtual environment):

```bat
.venv\Scripts\python.exe -m pip install -r GUI\requirements.txt
.venv\Scripts\python.exe -m pip install librosa
.venv\Scripts\python.exe GUI\smart_traffic_gui_100.py
```

Jika virtual environment berada di dalam folder lain, gunakan path lengkap ke `python.exe` milik environment tersebut pada ketiga perintah.

## Konfigurasi model lokal

Windows CMD:

```bat
set SMART_TRAFFIC_MODEL=C:/path/to/best.pt
python GUI/smart_traffic_gui_100.py
```

PowerShell:

```powershell
$env:SMART_TRAFFIC_MODEL = "C:/path/to/best.pt"
python GUI/smart_traffic_gui_100.py
```

Jika `SMART_TRAFFIC_MODEL` diatur tetapi file tidak ditemukan, aplikasi tidak menggunakan unduhan otomatis sebagai pengganti; pastikan path tersebut benar atau hapus variabel itu agar mekanisme unduhan default dapat digunakan.

## Kamera dan sumber video

Pada kode saat ini, seluruh sumber kamera bawaan bernilai `None`. Karena itu, feed kamera tidak otomatis aktif setelah aplikasi dibuka. Atur dictionary `SUMBER_KAMERA` di bagian konfigurasi awal `smart_traffic_gui_100.py` menggunakan perangkat atau file video yang benar-benar tersedia.

Contoh kamera USB:

```python
SUMBER_KAMERA = {
    "Utara": 0,
    "Timur": 1,
    "Selatan": 2,
    "Barat": 3,
}
```

Contoh video lokal:

```python
SUMBER_KAMERA = {
    "Utara": "videos/utara.mp4",
    "Timur": "videos/timur.mp4",
    "Selatan": "videos/selatan.mp4",
    "Barat": "videos/barat.mp4",
}
```

URL YouTube, jika digunakan, memerlukan konfigurasi dan dependensi resolver yang sesuai. `yt-dlp` tercantum sebagai dependensi opsional untuk skenario tersebut. Jangan menganggap sumber kamera telah aktif sebelum diuji pada perangkat yang digunakan.

ROI bawaan tidak aktif (`GUNAKAN_ROI = False`). Jika ROI diaktifkan, koordinat poligon harus disesuaikan dan dikalibrasi untuk masing-masing sudut kamera.

## Logika simulasi lampu

GUI menyediakan mode Otomatis dan Manual untuk empat lajur: Utara, Timur, Selatan, dan Barat. Pada mode Otomatis, pemilihan lajur berikutnya mempertimbangkan prioritas berikut:

1. Kendaraan darurat terdeteksi dan indikasi sirene pada lajur yang sama.
2. Kendaraan darurat terdeteksi (Ambulance, Fire engine, atau Police car).
3. Indikasi sirene dari analisis MP3 yang diasosiasikan pengguna dengan suatu lajur.
4. Ambang kepadatan berdasarkan total bobot seluruh kendaraan pada lajur. Ambang default adalah **5,0**, setara dengan contoh kombinasi 2 Car (2 × 1,0) + 2 Truck (2 × 1,5). Kombinasi tersebut bukan syarat wajib; kombinasi kelas apa pun dapat memicu prioritas jika total bobotnya mencapai atau melampaui 5,0.
5. Mekanisme anti-starvation dan skor kepadatan berbobot biasa.

Rumus total bobot:

`W = Σ (jumlah_kendaraan_kelas × bobot_kelas)`

Bobot yang digunakan: Ambulance 5,0; Bus 1,5; Car 1,0; Fire engine 5,0; Military 2,0; Motorcycle 0,5; Police car 5,0; Truck 1,5. Contoh lain yang memenuhi ambang 5,0 adalah 5 Car, 10 Motorcycle, atau 4 Bus.

Perpindahan fase tetap melewati lampu kuning. Prioritas tidak memotong fase hijau yang sedang berjalan secara langsung; pemilihan prioritas berlaku ketika sistem menentukan lajur berikutnya. Durasi hijau tetap mengikuti mekanisme alokasi waktu pada aplikasi.

Bobot kendaraan dan ambang 5,0 merupakan parameter model simulasi yang harus dijustifikasi dan divalidasi melalui skenario penelitian. Nilai ini bukan standar PCU atau standar rekayasa lalu lintas yang telah divalidasi.

## Analisis audio MP3

Versi `smart_traffic_gui_100.py` menyediakan pemilihan file MP3, pemilihan lajur terkait, dan analisis indikasi sirene.

**Status implementasi:** analisis audio saat ini menggunakan pendekatan heuristik berbasis fitur audio melalui `librosa`, bukan model klasifikasi audio terlatih. Implementasi ini **belum menggunakan YAMNet**. Nilai confidence yang ditampilkan merupakan skor heuristik, bukan probabilitas terkalibrasi.

Beberapa suara seperti klakson, alarm, musik, atau suara mesin dapat memicu indikasi keliru. Ambang deteksi perlu diuji dan dikalibrasi menggunakan kumpulan audio sirene serta non-sirene yang representatif.

File MP3 dipasangkan ke lajur melalui pilihan pengguna. Aplikasi tidak dapat menentukan lokasi sumber sirene secara otomatis hanya dari satu rekaman audio umum. Fitur ini merupakan analisis file, bukan pemantauan audio real-time.

## Data penelitian

Panel data penelitian mendukung perekaman snapshot berkala dan ekspor CSV mentah maupun ringkasan. Setiap snapshot merepresentasikan kondisi yang terbaca pada saat pencatatan; baris snapshot tidak menunjukkan identitas kendaraan unik. Periksa data hasil ekspor dan dokumentasikan kondisi pengujian sebelum menggunakannya dalam analisis atau laporan.

## Batasan dan status pengujian

- Sumber kamera default belum diisi dan harus dikonfigurasi sendiri.
- Ketepatan deteksi YOLO dapat berubah karena pencahayaan, sudut kamera, jarak, oklusi, dan kualitas video.
- Analisis sirene masih heuristik dan dapat menghasilkan false positive maupun false negative.
- Input MP3 tidak melokalisasi sumber suara secara otomatis.
- Bobot kendaraan, ambang kepadatan, dan skor audio belum boleh dianggap sebagai standar terukur tanpa validasi.
- Kompatibilitas dependensi, performa, dan hasil deteksi perlu diuji pada lingkungan pengguna.

Gunakan aplikasi ini untuk simulasi dan penelitian terkontrol. README ini tidak menyatakan bahwa sistem telah memenuhi persyaratan keselamatan atau regulasi untuk mengendalikan lampu lalu lintas nyata.
