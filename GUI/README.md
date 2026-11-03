# Lampiu — Simulasi Lampu Lalu Lintas Adaptif

Repository ini berisi beberapa generasi prototipe deteksi kendaraan dan analisis audio. **Implementasi GUI utama yang didokumentasikan saat ini adalah `GUI/smart_traffic_gui_100.py`**, dengan launcher pemilihan sumber CCTV/video pada `GUI/smart_traffic_jogja.py`.

Lampiu pada repository ini merupakan **simulasi perangkat lunak untuk penelitian dan pengujian terkontrol**. Program tidak mengirim perintah ke pengendali lampu lalu lintas fisik dan tidak boleh diperlakukan sebagai sistem kendali jalan umum.

## Mulai menjalankan GUI utama

Gunakan Python 3.10 atau 3.11. Dari direktori root repository:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r GUI/requirements.txt
python GUI/smart_traffic_jogja.py
```

Launcher menyediakan pemilihan sumber untuk masing-masing lajur: CCTV Jogja yang tersedia di `GUI/cameras.json`, webcam lokal, file video lokal, sumber bawaan, atau Nonaktif. Untuk skenario audio darurat yang saat ini didukung, pilih **satu video lokal yang memiliki trek audio** pada lajur yang disimulasikan.

Untuk membuka GUI tanpa pemilih sumber CCTV, jalankan:

```bash
python GUI/smart_traffic_gui_100.py
```

## Peta file

| Lokasi | Peran |
|---|---|
| `GUI/smart_traffic_jogja.py` | Launcher pemilihan sumber video/CCTV per lajur |
| `GUI/smart_traffic_gui_100.py` | Implementasi GUI utama, YOLO, alokasi waktu hijau, preemption, audio, dan pencatatan data |
| `GUI/audio_siren.py` | Inferensi YAMNet embedding + classifier Logistic Regression |
| `GUI/video_audio.py` | Ekstraksi soundtrack dari video lokal menggunakan FFmpeg |
| `GUI/audio_models/best_audio_classifier_v2.joblib` | Artefak classifier audio terlatih |
| `GUI/100ep.pt` | Checkpoint YOLO26n 100 epoch (jika tersedia lokal) |
| `GUI/cameras.json` | Daftar sumber CCTV Jogja yang ditampilkan launcher |
| `GUI/cctv_sources.json` | Sumber bawaan yang dibaca GUI utama; nilai kosong berarti tidak ada sumber bawaan |
| `GUI/README.md` | Konfigurasi dan rincian implementasi GUI utama |
| `AUDIO_SETUP.md` | Catatan prototipe audio lama pada folder `Coba/` |
| `Coba/` | Eksperimen/prototipe terdahulu; bukan entry point GUI utama saat ini |

## Ringkasan konfigurasi implementasi utama

### Deteksi visual

- Checkpoint: YOLO26n hasil pelatihan 100 epoch, dengan delapan kelas: Ambulance, Bus, Car, Fire engine, Military, Motorcycle, Police car, dan Truck.
- Confidence awal: 0,15; rentang slider GUI 0,05–0,80.
- NMS IoU awal: 0,60; rentang slider GUI 0,30–0,90.
- Ukuran inferensi: 640; frame proses: 640 × 480.
- Jumlah kendaraan distabilkan dengan median lima frame.
- ROI nonaktif secara default; bila diaktifkan, poligon perlu disesuaikan dengan masing-masing kamera.

### Bobot kendaraan

Nilai yang benar-benar dipakai oleh GUI utama:

| Kelas | Bobot operasional |
|---|---:|
| Ambulance | 1,00 |
| Bus | 2,00 |
| Car | 1,00 |
| Fire engine | 2,00 |
| Military | 2,00 |
| Motorcycle | 0,33 |
| Police car | 1,00 |
| Truck | 2,00 |

Rumus skor lajur adalah `W = Σ(jumlah kelas × bobot kelas)`. Nilai di atas adalah **parameter operasional simulator**, bukan klaim bahwa seluruhnya merupakan nilai PCU universal atau telah divalidasi untuk kondisi lalu lintas Yogyakarta. Prioritas kendaraan darurat ditangani secara terpisah melalui konfirmasi visual-audio, bukan dengan menaikkan bobot PCU kendaraan darurat.

### Alokasi dan transisi sinyal

- Urutan normal lajur adalah Utara → Timur → Selatan → Barat.
- Total alokasi waktu hijau untuk empat lajur adalah 120 detik per putaran alokasi.
- Durasi hijau setiap lajur dibatasi 10–40 detik dan dibagi secara proporsional terhadap skor bobot kendaraan.
- Bobot diambil sebagai satu snapshot untuk satu putaran alokasi.
- Kuning berlangsung 3 detik dan all-red 1 detik pada setiap transisi. Satu putaran normal empat lajur memiliki empat transisi, sehingga total waktu kalendernya 120 + 4 × (3 + 1) = **136 detik**, sebelum memperhitungkan preemption atau jeda lain.
- Jika skor semua lajur nol, waktu hijau dibagi rata.
- Preemption darurat dapat mengubah target lajur berikutnya, tetapi tetap melewati kuning dan all-red. Urutan normal selain preemption tetap searah jarum jam.

### Audio dan konfirmasi darurat

GUI utama tidak menerima MP3 terpisah sebagai sumber input skenario saat ini. Audio berasal dari **trek audio video lokal** yang dipilih melalui launcher.

Alur audio:
1. Program memilih video lokal yang dipasangkan dengan salah satu lajur.
2. `GUI/video_audio.py` mengekstrak soundtrack video menjadi PCM stereo 44,1 kHz.
3. GUI memproses potongan audio berdurasi 3 detik, mengubahnya menjadi mono, lalu melakukan resampling ke 16 kHz.
4. `GUI/audio_siren.py` menjalankan YAMNet untuk menghasilkan embedding 1.024 dimensi.
5. Embedding dirata-ratakan per potongan dan diklasifikasikan oleh Logistic Regression yang disimpan pada artefak `best_audio_classifier_v2.joblib`.
6. Kelas classifier: `ambulance`, `firetruck`, `police`, dan `traffic`. Skor audio darurat adalah penjumlahan skor tiga kelas darurat, bukan skor kelas YAMNet `Siren` mentah.

Catatan pengoperasian:
- Video lokal diputar dalam keadaan **mute secara default**. Analisis audio tetap berjalan. Tekan tombol **Nyalakan suara** jika soundtrack perlu terdengar pada skenario uji.
- Kontrol jeda, seek, mute, dan pemindahan video ke lajur tersedia di panel kontrol video.
- Untuk mencegah pemetaan audio yang ambigu, gunakan satu video lokal sebagai skenario audio darurat. Bila beberapa video lokal dipasang, GUI saat ini mengambil video lokal pertama yang ditemukan sebagai sumber soundtrack; video lokal lainnya tidak menjadi sumber audio terpisah.
- Skor agregat classifier bukan probabilitas yang telah dikalibrasi untuk penggunaan operasional.

Konfirmasi dual-modal pada lajur yang sama mensyaratkan:
- Confidence deteksi visual kelas Ambulance, Fire engine, atau Police car **lebih besar dari 0,65**; dan
- Skor agregat audio darurat **lebih besar dari 0,65**.

Kedua syarat menggunakan operator `>` (bukan `>=`). Nilai 0,65 adalah ambang awal konfigurasi dan belum dinyatakan sebagai hasil kalibrasi lapangan.

### Artefak classifier audio

Riwayat eksperimen pelatihan proyek mencatat Logistic Regression dengan `C=100` sebagai konfigurasi terpilih dari kandidat C yang diuji. File inferensi hanya memuat artefak terlatih dan memeriksa keberadaan `predict_proba` serta empat nama kelas; modul inferensi tidak melatih ulang model atau menetapkan ulang nilai C. Catatan parameter dan batas verifikasinya ada di `GUI/audio_models/README.md`.

## Data penelitian

GUI utama dapat merekam snapshot per lajur dan mengekspor CSV mentah serta ringkasan. Setiap baris adalah snapshot kondisi saat pencatatan, bukan identitas kendaraan unik atau hasil evaluasi performa model. Catat sumber video, konfigurasi slider, bobot, ambang, dan skenario saat melakukan eksperimen agar data dapat direproduksi.

**Bab 4/hasil eksperimen belum menjadi bagian dari audit perubahan ini.** Repository dan dokumentasi ini tidak menambahkan angka performa atau klaim efektivitas sistem yang belum diuji.

## Prototipe terdahulu

Folder `Coba/` berisi eksperimen lama, termasuk pipeline YAMNet yang menilai kelas AudioSet `Siren` secara langsung dengan `SIREN_THRESHOLD=0.25`. Itu **bukan** pipeline audio GUI utama saat ini. Lihat `AUDIO_SETUP.md` untuk batasan dan dependensinya.

## Batasan keselamatan dan penggunaan

- Gunakan hanya untuk simulasi, pengembangan, dan penelitian terkontrol.
- Jangan menghubungkan prototipe ini ke pengendali lampu lalu lintas jalan umum.
- CCTV dan file video merupakan input pengamatan; sistem tidak mengontrol kamera atau infrastruktur CCTV.
- Akurasi, latency, threshold, bobot kendaraan, dan keselamatan transisi perlu diuji pada lingkungan serta skenario yang ditentukan sebelum menarik kesimpulan penelitian.
