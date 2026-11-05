# Dokumentasi artefak classifier audio V2

Dokumen ini menjelaskan artefak yang dimuat oleh `GUI/audio_siren.py` pada GUI utama Lampiu.

## Kontrak inferensi

| Komponen | Konfigurasi |
|---|---|
| Artefak | `best_audio_classifier_v2.joblib` |
| Ekstraktor fitur | Google YAMNet pretrained |
| Sumber model YAMNet | `https://tfhub.dev/google/yamnet/1` |
| Input classifier | Rata-rata embedding YAMNet berdimensi 1.024 |
| Sampling rate input | Mono, 16 kHz |
| Jendela analisis GUI | 3 detik |
| Kelas | `ambulance`, `firetruck`, `police`, `traffic` |

Fungsi inferensi menghasilkan skor `predict_proba` untuk setiap kelas. Skor agregat audio darurat dihitung sebagai:

`emergency_score = P(ambulance) + P(firetruck) + P(police)`

Skor ini merupakan agregasi keluaran classifier. Jangan menyebutnya sebagai probabilitas sirene yang telah dikalibrasi secara independen untuk penggunaan operasional.

## Asal konfigurasi pelatihan

Rekap eksperimen pelatihan proyek mencatat Logistic Regression dengan `C=100` sebagai konfigurasi yang dipilih. Nilai ini merupakan informasi dari riwayat pelatihan yang tersedia, bukan parameter yang ditetapkan ulang oleh modul inferensi.

**Batas verifikasi repository:** repository saat ini menyimpan artefak estimator, tetapi tidak menyertakan notebook/script pelatihan lengkap atau manifest pelatihan yang dapat direproduksi. Modul inferensi memeriksa bahwa estimator menyediakan `predict_proba()` dan `classes_`, serta memverifikasi bahwa empat nama kelas sesuai dengan kontrak. Modul tersebut tidak memeriksa ulang nilai `C` estimator.

Karena itu, `C=100` dicatat sebagai konfigurasi pelatihan yang dilaporkan, bukan sebagai nilai yang telah dibuktikan ulang melalui kode inferensi.

## Pembaruan artefak

Jika model dilatih ulang, dokumentasi ini perlu diperbarui bersama artefaknya. Catat sekurang-kurangnya:

1. versi dan sumber dataset;
2. definisi kelas dan jumlah sampel per kelas;
3. strategi pembagian train/validation/test;
4. random seed dan prapemrosesan;
5. ekstraktor fitur serta versi model YAMNet;
6. algoritma classifier dan seluruh hyperparameter;
7. urutan kelas yang tersimpan pada estimator;
8. metrik validasi dan pengujian; dan
9. checksum artefak model.

Jangan mengganti file `best_audio_classifier_v2.joblib` tanpa memperbarui catatan asal model dan kontrak inferensinya.
