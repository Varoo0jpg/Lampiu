# Audio sirene Lampiu (prototipe)

Modul ini memakai YAMNet pralatih dari TensorFlow Hub untuk mencari kelas suara **Siren** pada sumber audio tiap simpang. YAMNet mengklasifikasikan 521 jenis peristiwa suara, termasuk sirene. Skor model yang ditampilkan di GUI bukan probabilitas yang sudah dikalibrasi.

## 1. Instalasi

Gunakan Python 3.10 atau 3.11 pada lingkungan yang sama dengan aplikasi Lampiu.

```bash
pip install -r requirements-audio.txt
```

Instal FFmpeg secara terpisah dan pastikan perintah `ffmpeg` tersedia di PATH. Pada pemakaian pertama, TensorFlow Hub perlu mengunduh model YAMNet dari internet.

## 2. Sumber audio

Di bagian konfigurasi `Coba/Coba_kamera.py`, variabel `SUMBER_AUDIO` secara default memakai URL yang sama dengan sumber video untuk masing-masing simpang. Kamu dapat mengganti nilai setiap simpang menjadi path MP4 lokal, misalnya:

```python
SUMBER_AUDIO = {
    "Utara": "C:/Lampiu/video/utara.mp4",
    "Timur": "C:/Lampiu/video/timur.mp4",
    "Selatan": "C:/Lampiu/video/selatan.mp4",
    "Barat": "C:/Lampiu/video/barat.mp4",
}
```

Gunakan `None` untuk menonaktifkan sumber audio pada simpang tertentu. MP4 harus memiliki trek audio. Untuk URL YouTube, modul menggunakan `yt-dlp` untuk menemukan aliran audio.

## 3. Tampilan dan ambang

Panel menampilkan skor sirene yang diperhalus dari waktu ke waktu. Ambang awal `SIREN_THRESHOLD = 0.25` adalah nilai awal untuk eksperimen, bukan hasil kalibrasi untuk kondisi jalan tertentu. Sistem meminta sedikitnya dua dari tiga jendela terakhir melewati ambang sebelum menandai indikasi sirene.

Audio belum mengubah lampu lalu lintas secara otomatis. Tetap gunakan deteksi visual sebagai konfirmasi. Uji dengan rekaman sirene, lalu lintas biasa, klakson, musik, dan suara latar lainnya untuk mengevaluasi false positive dan false negative.

## 4. Batasan prototipe

- Setiap simpang harus mengacu ke sumber audio yang benar. Perbedaan level mikrofon, jarak, dan kebisingan memengaruhi skor.
- URL streaming memerlukan koneksi internet dan dapat terputus.
- Kelas YAMNet bersifat umum. Untuk pemakaian lebih serius, kumpulkan data audio lokal yang berlabel dan kalibrasi atau latih classifier khusus sirene.
