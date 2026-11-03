# Lampiu — Dokumentasi Sistem ARGUS

Lampiu merupakan repository pengembangan prototipe perangkat lunak untuk simulasi pengaturan sinyal lalu lintas pada persimpangan empat lajur. Naskah penelitian menggunakan nama sistem **ARGUS (Adaptive Real-time Green-time Update System)**.

**Launcher GUI utama:** `GUI/smart_traffic_jogja.py`  
**Implementasi inti:** `GUI/smart_traffic_gui_100.py`

> **Batas penggunaan:** perangkat lunak ini ditujukan untuk simulasi, pengembangan, dan penelitian terkontrol. Aplikasi tidak mengirim perintah ke pengendali lampu lalu lintas fisik dan tidak boleh digunakan untuk mengendalikan lalu lintas jalan umum.

## 1. Menjalankan GUI

Gunakan Python 3.10 atau 3.11. Jalankan perintah berikut dari direktori root repository.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r GUI\requirements.txt
python GUI\smart_traffic_jogja.py
```

Launcher memungkinkan pengguna memilih sumber video secara manual untuk setiap lajur Utara, Timur, Selatan, dan Barat:

- CCTV Jogja yang tercantum dalam `GUI/cameras.json`;
- webcam lokal;
- file video lokal;
- sumber bawaan yang tersedia; atau
- Nonaktif.

Daftar CCTV merupakan konfigurasi lokal. Status `alive` di dalam JSON berasal dari pemeriksaan sebelumnya dan bukan jaminan bahwa stream masih aktif saat aplikasi dijalankan.

Untuk membuka GUI inti tanpa launcher pemilihan sumber, jalankan:

```bash
python GUI/smart_traffic_gui_100.py
```

## 2. Peta dokumentasi dan file utama

| Lokasi | Fungsi |
|---|---|
| `GUI/smart_traffic_jogja.py` | Launcher pemilihan sumber CCTV, webcam, dan video lokal per lajur |
| `GUI/smart_traffic_gui_100.py` | GUI inti, inferensi YOLO, perhitungan bobot kendaraan, pengaturan fase, preemption, dan pencatatan data |
| `GUI/audio_siren.py` | Inferensi YAMNet embedding dan classifier Logistic Regression |
| `GUI/video_audio.py` | Dekode trek audio dari video lokal menggunakan FFmpeg |
| `GUI/audio_models/best_audio_classifier_v2.joblib` | Artefak classifier audio |
| `GUI/audio_models/README.md` | Kelas model, kontrak inferensi, dan catatan asal konfigurasi pelatihan |
| `GUI/requirements.txt` | Dependensi GUI utama |
| `GUI/cameras.json` | Daftar stream CCTV yang ditawarkan launcher |
| `AUDIO_SETUP.md` | Penjelasan prototipe audio lama pada folder `Coba/` |
| `Coba/` | Eksperimen terdahulu; bukan jalur GUI utama yang didokumentasikan |

## 3. Ringkasan implementasi GUI utama

### Deteksi visual

- Model yang digunakan adalah checkpoint YOLO26n hasil pelatihan 100 epoch.
- Delapan kelas kendaraan: Ambulance, Bus, Car, Fire engine, Military, Motorcycle, Police car, dan Truck.
- Confidence awal 0,15 dengan rentang pengaturan GUI 0,05–0,80.
- NMS IoU awal 0,60 dengan rentang pengaturan GUI 0,30–0,90.
- Ukuran inferensi 640; frame pemrosesan 640 × 480.
- Jumlah kendaraan distabilkan menggunakan median lima frame.
- ROI nonaktif secara default. Jika diaktifkan, area ROI perlu disesuaikan dengan tampilan kamera.

### Bobot operasional kendaraan

| Kelas | Bobot yang digunakan |
|---|---:|
| Ambulance | 1,00 |
| Bus | 2,00 |
| Car | 1,00 |
| Fire engine | 2,00 |
| Military | 2,00 |
| Motorcycle | 0,33 |
| Police car | 1,00 |
| Truck | 2,00 |

Skor bobot lajur dihitung dengan menjumlahkan hasil perkalian jumlah kendaraan tiap kelas dan bobot kelasnya:

`W_l = Σ(n_{l,c} × w_c)`

Angka tersebut merupakan **parameter operasional simulator**, bukan pernyataan bahwa semua bobot merupakan nilai PCU universal atau telah divalidasi untuk kondisi lalu lintas Yogyakarta. Prioritas kendaraan darurat ditentukan melalui konfirmasi visual-audio dan bukan melalui pembesaran bobot PCU.

### Alokasi waktu dan urutan fase

- Urutan normal lajur: Utara → Timur → Selatan → Barat.
- Total alokasi waktu hijau empat lajur: 120 detik per putaran alokasi.
- Batas waktu hijau: minimum 10 detik dan maksimum 40 detik per lajur.
- Alokasi dibagi secara proporsional terhadap skor bobot kendaraan, dengan batas minimum dan maksimum diterapkan melalui distribusi ulang sisa waktu.
- Snapshot skor bobot dipertahankan selama satu putaran agar durasi antarlajur dalam putaran tersebut konsisten.
- Jika seluruh skor bobot nol, sisa waktu dibagi rata.
- Kuning berlangsung 3 detik dan all-red 1 detik pada setiap transisi.
- Satu putaran normal empat lajur memiliki empat transisi. Dengan demikian, durasi kalender putaran normal adalah 120 detik alokasi hijau + 4 × 4 detik clearance = 136 detik. Durasi ini dapat berubah ketika terjadi preemption atau pemaksaan lajur manual.

### Audio dan emergency preemption

GUI utama menggunakan **trek audio dari video lokal yang dipilih pengguna**, bukan file MP3 terpisah sebagai sumber input audio pada alur launcher saat ini.

Alur pemrosesan:
1. Pengguna memilih file video lokal dan memasangkannya dengan salah satu lajur.
2. `GUI/video_audio.py` mendekode trek audio menjadi PCM stereo 44,1 kHz.
3. Audio diproses per jendela 3 detik; kanal stereo dirata-ratakan menjadi mono dan di-resampling ke 16 kHz.
4. YAMNet menghasilkan embedding 1.024 dimensi yang dirata-ratakan untuk jendela tersebut.
5. Logistic Regression menghasilkan skor untuk empat kelas: `ambulance`, `firetruck`, `police`, dan `traffic`.
6. Skor audio darurat merupakan penjumlahan skor tiga kelas darurat: ambulance, firetruck, dan police.

Video lokal diputar dalam keadaan mute secara default, tetapi analisis audio tetap berjalan. Pengguna dapat mengaktifkan suara melalui kontrol GUI. Audio hanya berasal dari video lokal pertama yang ditemukan menurut urutan lajur. Jika tidak ada video lokal, tidak tersedia skor audio dari jalur ini.

Skor audio dipetakan ke lajur tempat video lokal dipilih. Nilai tersebut kemudian digunakan bersama confidence visual untuk pemeriksaan dual-modal pada lajur yang sama:

- confidence visual maksimum untuk kelas Ambulance, Fire engine, atau Police car harus **lebih besar dari 0,65**;
- skor agregat audio darurat harus **lebih besar dari 0,65**.

Kedua kondisi menggunakan operator `>`, bukan `>=`. Ambang tersebut merupakan konfigurasi awal dan belum dinyatakan sebagai hasil kalibrasi lapangan. Skor `predict_proba` classifier juga tidak boleh dianggap sebagai probabilitas yang telah dikalibrasi untuk kondisi operasional.

Jika konfirmasi dual-modal terpenuhi pada lajur selain lajur aktif, engine simulasi meminta lajur tersebut mendapat giliran hijau berikutnya. Perpindahan tetap melalui urutan HIJAU → KUNING → ALL_RED → HIJAU; preemption tidak melewati fase clearance. Pemaksaan lajur manual memiliki prioritas atas preemption otomatis selama override manual aktif.

### Pencatatan data

GUI menyediakan perekaman snapshot kondisi tiap lajur serta ekspor CSV mentah dan ringkasan. Baris data merupakan snapshot pada waktu pencatatan, bukan identitas kendaraan unik. Catat sumber video, waktu, konfigurasi slider, bobot, dan skenario ketika melakukan eksperimen agar data dapat ditelusuri.

## 4. Artefak model audio

Rincian kontrak inferensi dan catatan parameter pelatihan tersedia pada [GUI/audio_models/README.md](GUI/audio_models/README.md). Artefak classifier tidak dilatih ulang oleh modul inferensi. Perubahan artefak harus disertai pembaruan informasi dataset, pemisahan data, parameter, kelas, metrik, dan checksum model.

## 5. Prototipe audio terdahulu

Folder `Coba/` menyimpan prototipe audio terpisah yang menggunakan skor kelas AudioSet `Siren` dari YAMNet secara langsung. Prototipe tersebut bukan pipeline Logistic Regression empat kelas pada GUI utama. Lihat [AUDIO_SETUP.md](AUDIO_SETUP.md) untuk perbedaan dan dependensinya.

## 6. Batasan dan status hasil penelitian

- Lampiu/ARGUS pada repository ini adalah simulasi perangkat lunak, bukan pengendali lampu fisik.
- CCTV dan video lokal digunakan sebagai input pengamatan; aplikasi tidak mengendalikan infrastruktur kamera.
- Bobot kendaraan, threshold, akurasi, latency, dan keselamatan transisi membutuhkan pengujian dan validasi sesuai skenario penelitian.
- Dokumentasi ini tidak menyatakan adanya peningkatan kinerja, penurunan tundaan, atau pengurangan antrean sebelum hasil eksperimen tersedia.
- Bab 4 dan hasil eksperimen tidak disusun dalam pembaruan dokumentasi ini.
