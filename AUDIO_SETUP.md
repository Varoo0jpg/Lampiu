# Catatan audio Lampiu

Dokumen ini menjelaskan **prototipe audio lama** pada `Coba/siren_audio.py`. Modul tersebut menggunakan YAMNet untuk menilai kelas AudioSet `Siren` secara langsung. Modul ini berbeda dari pipeline audio pada GUI utama dan tidak boleh dipakai sebagai dokumentasi implementasi GUI utama.

## Pilih dokumentasi sesuai implementasi

| Implementasi | Pipeline | Ambang awal |
|---|---|---:|
| GUI utama: `GUI/smart_traffic_gui_100.py` | Soundtrack video lokal → YAMNet embedding → Logistic Regression empat kelas | Skor agregat audio darurat 0,65 |
| Prototipe lama: `Coba/siren_audio.py` | YAMNet langsung menilai kelas AudioSet `Siren` | `SIREN_THRESHOLD = 0.25` |

Jangan mencampur kedua ambang atau menyebut classifier Logistic Regression sebagai bagian dari prototipe `Coba/`.

## Prototipe lama pada folder Coba

Modul `Coba/siren_audio.py` mendukung sumber file lokal atau URL HTTP(S), termasuk URL YouTube yang di-resolve menggunakan `yt-dlp`. FFmpeg mengubah sumber menjadi PCM mono 16 kHz. Implementasi ini menggunakan jendela audio 1,5 detik, smoothing `SMOOTHING_ALPHA = 0.45`, dan indikasi positif jika sedikitnya dua dari tiga jendela terakhir melewati ambang.

Ambang `SIREN_THRESHOLD = 0.25` adalah nilai awal prototipe, bukan ambang terkalibrasi. Skor YAMNet merupakan model score, bukan probabilitas terkalibrasi.

### Dependensi prototipe

Gunakan Python 3.10 atau 3.11 pada environment terpisah atau environment yang kompatibel:

```bash
python -m pip install -r requirements-audio.txt
```

`requirements-audio.txt` dipertahankan untuk prototipe audio lama. Untuk GUI utama, gunakan `GUI/requirements.txt` dan ikuti [GUI/README.md](GUI/README.md).

FFmpeg dibutuhkan oleh `Coba/siren_audio.py` dan harus tersedia pada PATH. Untuk GUI utama, `imageio-ffmpeg` disertakan sebagai dependensi dan digunakan oleh `GUI/video_audio.py` untuk mengekstrak soundtrack video lokal.

### Status integrasi

`Coba/Coba_kamera.py` saat ini tidak mendefinisikan `SUMBER_AUDIO` dan tidak menginisialisasi `SirenAudioEngine`. Karena itu, jangan mengikuti instruksi konfigurasi `SUMBER_AUDIO` versi lama. Modul `Coba/siren_audio.py` adalah engine terpisah/prototipe dan belum menjadi jalur audio GUI utama.

## Pipeline audio GUI utama

Untuk implementasi yang digunakan dalam penelitian Lampiu saat ini, lihat:

- [GUI/README.md](GUI/README.md) — alur input video lokal, playback/mute, 3-second window, 16 kHz, YAMNet embeddings, Logistic Regression, dan dual-modal confirmation.
- [GUI/audio_models/README.md](GUI/audio_models/README.md) — kelas classifier, konfigurasi pelatihan yang tercatat, dan batas verifikasi artefak.

GUI utama mengambil soundtrack dari satu video lokal yang dipasangkan ke lajur. Audio dianalisis meskipun playback dalam keadaan mute. Video lokal diputar mute secara default; tombol **Nyalakan suara** mengaktifkan playback soundtrack untuk skenario uji.

Semua ambang audio pada kedua pipeline masih merupakan konfigurasi awal dan tidak boleh dipresentasikan sebagai hasil kalibrasi lapangan tanpa pengujian tersendiri.
