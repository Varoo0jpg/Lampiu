# Coba_kamera.py
# GUI CustomTkinter (Coba.py) + deteksi kendaraan 4 kamera (YOLO)
#
# Instalasi:
#   pip install customtkinter pillow opencv-python ultralytics numpy
#
# Penanda perubahan terhadap Coba.py asli:
#   [BARU]       = bagian tambahan (deteksi kamera, tampilan feed)
#   [PERBAIKAN]  = bug pada kode asli yang diperbaiki
import collections
import os
import random
import sys
import threading
import time

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

# ===============================================================
# KONFIGURASI SISTEM
# ===============================================================
DAFTAR_LAJUR = ["Utara", "Timur", "Selatan", "Barat"]

BOBOT_KENDARAAN = {
    "Motor": 0.25,
    "Mobil": 1.0,
    "Truk": 2.5,
    "Bus": 2.0
}

FASE_HIJAU_MIN = 10     # Batas minimal waktu hijau (detik)
FASE_HIJAU_MAX = 40     # Batas maksimal waktu hijau (detik)
FASE_KUNING = 3         # Durasi lampu kuning (detik)

# Kode Warna UI Hexadecimal
HEX_RED = "#e74c3c"
HEX_YELLOW = "#f1c40f"
HEX_GREEN = "#2ecc71"
HEX_DARK_BG = "#1e2530"

# ---------------- [BARU] Konfigurasi kamera & deteksi ----------------
# Sumber tiap lajur: indeks kamera (0,1,2,..), path file video, atau None (input manual).
# Default: hanya kamera laptop untuk lajur Utara.
SUMBER_KAMERA = {"Utara": "https://www.youtube.com/watch?v=Ck45cxJAh-s", "Timur": "https://www.youtube.com/watch?v=u-8iSRWgi_c", "Selatan": "https://www.youtube.com/live/BdPvBNLy5kc?si=tm8jdiBL6TW7jkJy", "Barat": "https://www.youtube.com/watch?v=D_R5EpqQIpo"}

# Sumber audio per simpang. Boleh berupa URL streaming atau path MP4 lokal.
# Secara default audio diambil dari URL yang sama dengan kamera.
# Untuk MP4 lokal, ganti nilai setiap simpang, misalnya "C:/Lampiu/video/utara.mp4".
SUMBER_AUDIO = dict(SUMBER_KAMERA)
# Contoh 4 kamera USB:
# SUMBER_KAMERA = {"Utara": 0, "Timur": 1, "Selatan": 2, "Barat": 3}
# Contoh 4 file video:
# SUMBER_KAMERA = {"Utara": "utara.mp4", "Timur": "timur.mp4",
#                  "Selatan": "selatan.mp4", "Barat": "barat.mp4"}


def resource_path(nama):
    """Lokasi file bawaan; juga bekerja saat dibungkus PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, nama)


MODEL_PATH = resource_path("yolo26n.pt")   # YOLO26 nano; alternatif: yolo26s.pt (lebih akurat, lebih berat)
CONF = 0.35
IMGSZ = 416                      # kecil = lebih cepat di CPU
UKURAN_FRAME = (640, 480)        # ukuran proses deteksi
UKURAN_TAMPIL = (330, 248)       # ukuran feed di kotak kamera GUI

# Kelas COCO -> jenis kendaraan pada panel input
KELAS_YOLO = {2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}

# Nama kelas yang wajib cocok dengan bobot YOLO COCO bawaan.
# Pemeriksaan ini mencegah model hasil training khusus dipakai dengan indeks kelas COCO.
NAMA_KELAS_COCO = {
    2: {"car"},
    3: {"motorcycle", "motorbike"},
    5: {"bus"},
    7: {"truck"},
}


def validasi_model_kendaraan(model):
    """Pastikan indeks kelas model cocok dengan KELAS_YOLO sebelum menghitung kendaraan."""
    nama = getattr(model, "names", None)
    if nama is None:
        raise RuntimeError(
            "Model YOLO tidak menyediakan daftar nama kelas. "
            "Pastikan MODEL_PATH menunjuk ke bobot deteksi kendaraan yang valid."
        )

    for indeks, nama_yang_diharapkan in NAMA_KELAS_COCO.items():
        if isinstance(nama, dict):
            aktual = nama.get(indeks, nama.get(str(indeks)))
        elif isinstance(nama, (list, tuple)) and indeks < len(nama):
            aktual = nama[indeks]
        else:
            aktual = None

        aktual = str(aktual).strip().lower().replace("_", " ") if aktual is not None else ""
        if aktual not in nama_yang_diharapkan:
            raise RuntimeError(
                f"Model pada {MODEL_PATH!r} tidak cocok dengan pemetaan kelas kendaraan COCO. "
                f"Kelas indeks {indeks} seharusnya salah satu dari "
                f"{sorted(nama_yang_diharapkan)}, tetapi model berisi {aktual!r}. "
                "Gunakan bobot YOLO COCO yang sesuai, atau sesuaikan pemetaan kelas "
                "dan bobot ketika memakai dataset kendaraan darurat."
            )

# ROI per lajur (pecahan frame): area antrean di depan kamera. SESUAIKAN per kamera.
ROI_DEFAULT = [(0.15, 0.30), (0.85, 0.30), (0.98, 1.00), (0.02, 1.00)]
ROI = {l: list(ROI_DEFAULT) for l in DAFTAR_LAJUR}
# False = hitung kendaraan di SELURUH frame kamera (ROI di atas diabaikan)
# True  = hanya hitung kendaraan di dalam poligon ROI (garis kuning)
GUNAKAN_ROI = False

# Pemilihan lajur hijau berikutnya pada mode OTOMATIS:
#   "PADAT"    = lajur dengan skor tertinggi didahulukan (dengan batas tunggu maksimum)
#   "BERGILIR" = urutan tetap Utara -> Timur -> Selatan -> Barat
PILIH_LAJUR_BERIKUT = "PADAT"
BATAS_TUNGGU_MAKS = 180          # detik; lajur yang menunggu lebih lama dipaksa mendapat giliran


# ===============================================================
# [BARU] SUBSISTEM: DETEKSI KENDARAAN DARI KAMERA (THREAD)
# ===============================================================
def resolve_sumber(sumber):
    """[BARU] Ubah link YouTube (live/video) menjadi URL stream langsung via yt-dlp.
    Sumber lain (indeks kamera, file video, None) dikembalikan apa adanya."""
    if isinstance(sumber, str) and ("youtube.com" in sumber or "youtu.be" in sumber):
        import yt_dlp
        # Audio tidak dipakai, jadi stream video-saja (bv*) juga boleh
        opsi = {"quiet": True, "noplaylist": True,
                "format": "bv*[height<=480]/b[height<=480]/bv*/b"}
        with yt_dlp.YoutubeDL(opsi) as ydl:
            info = ydl.extract_info(sumber, download=False)
        return info["url"]
    return sumber


class DetektorKamera(threading.Thread):
    """Membaca semua kamera, deteksi YOLO (batch), hitung kendaraan per jenis dalam ROI."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.running = True
        self.siap = False          # True setelah model dimuat & kamera dicoba dibuka
        self.error = None
        self.caps = {}
        self.frames = {l: None for l in DAFTAR_LAJUR}
        self.jumlah = {l: {j: 0 for j in BOBOT_KENDARAAN} for l in DAFTAR_LAJUR}
        self._riwayat = {l: collections.deque(maxlen=5) for l in DAFTAR_LAJUR}

    def _buka_sumber(self):
        for lajur, sumber in SUMBER_KAMERA.items():
            if sumber is None:
                continue
            try:
                sumber_nyata = resolve_sumber(sumber)
            except Exception as e:
                print(f"[!] Gagal membaca link YouTube lajur {lajur}: {e}")
                continue
            if isinstance(sumber_nyata, int) and os.name == "nt":
                cap = cv2.VideoCapture(sumber_nyata, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(sumber_nyata)
            if cap.isOpened():
                self.caps[lajur] = cap
            else:
                print(f"[!] Sumber lajur {lajur} ({sumber}) tidak terbuka")

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
            validasi_model_kendaraan(model)
            self._buka_sumber()
        except Exception as e:
            self.error = repr(e)
            self.siap = True
            return
        self.siap = True

        w, h = UKURAN_FRAME
        while self.running and self.caps:
            imgs = {}
            for lajur, cap in self.caps.items():
                ok, img = cap.read()
                if not ok:
                    if isinstance(SUMBER_KAMERA[lajur], str):     # file video: ulang
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                imgs[lajur] = cv2.resize(img, UKURAN_FRAME)
            if not imgs:
                time.sleep(0.02)
                continue

            daftar = list(imgs)
            hasil = model([imgs[l] for l in daftar], conf=CONF, imgsz=IMGSZ,
                          classes=list(KELAS_YOLO), verbose=False)

            for lajur, res in zip(daftar, hasil):
                img = imgs[lajur]
                poly = np.array([(int(x * w), int(y * h)) for x, y in ROI[lajur]], np.int32)
                hitung = {j: 0 for j in BOBOT_KENDARAAN}
                for box in res.boxes:
                    jenis = KELAS_YOLO[int(box.cls[0])]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dalam = (not GUNAKAN_ROI) or cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
                    warna = (0, 255, 0) if dalam else (120, 120, 120)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), warna, 2)
                    if dalam:
                        hitung[jenis] += 1
                if GUNAKAN_ROI:
                    cv2.polylines(img, [poly], True, (0, 255, 255), 2)

                # median 5 frame terakhir agar angka tidak berkedip
                rw = self._riwayat[lajur]
                rw.append(hitung)
                median = {j: int(round(float(np.median([r[j] for r in rw]))))
                          for j in BOBOT_KENDARAAN}
                teks = " ".join(f"{j[:2]}:{n}" for j, n in median.items())
                cv2.putText(img, f"{lajur}  {teks}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                with self.lock:
                    self.jumlah[lajur] = median
                    self.frames[lajur] = img

    def ambil(self):
        with self.lock:
            jumlah = {l: dict(v) for l, v in self.jumlah.items()}
            frames = {l: (None if f is None else f.copy()) for l, f in self.frames.items()}
        return jumlah, frames

    def stop(self):
        self.running = False
        time.sleep(0.1)
        for cap in self.caps.values():
            cap.release()

# BANYUUU
# ===============================================================
# SUBSISTEM: INPUT DATA KENDARAAN (LAJUR PANEL CLASS)
# ===============================================================
class PanelInputLajur(ctk.CTkFrame):
    def __init__(self, parent, nama_lajur):
        super().__init__(parent, fg_color="#2d2d2d")
        self.nama_lajur = nama_lajur
        self.dikunci = False      # [BARU] True jika angka diisi otomatis oleh kamera

        # Label Title Lajur
        ctk.CTkLabel(self, text=f"Data Kendaraan - {nama_lajur}", font=("Arial", 11, "bold"), text_color="#FFCC00").pack(pady=2)

        # Frame Kontainer Kotak Input
        grid_input = ctk.CTkFrame(self, fg_color="transparent")
        grid_input.pack(fill="x", padx=10, pady=2)
        grid_input.columnconfigure((0, 1, 2, 3), weight=1)

        # Simpan objek widget entry ke dalam dictionary
        self.kolom_input = {}

        for indeks, jenis in enumerate(BOBOT_KENDARAAN.keys()):
            ctk.CTkLabel(grid_input, text=jenis, font=("Arial", 10)).grid(row=0, column=indeks, padx=2, pady=1)

            kotak_teks = ctk.CTkEntry(grid_input, width=45, justify="center")
            kotak_teks.insert(0, str(random.randint(0, 10)))  # Nilai acak awal
            kotak_teks.grid(row=1, column=indeks, padx=2, pady=2)
            self.kolom_input[jenis] = kotak_teks

    def ambil_skor_bobot(self):
        """Menghitung total beban PCU lajur dari angka input user."""
        total_skor = 0.0
        for jenis, bobot in BOBOT_KENDARAAN.items():
            try:
                jumlah = int(self.kolom_input[jenis].get())
                total_skor += jumlah * bobot
            except ValueError:
                total_skor += 0.0
        return round(total_skor, 2)

    def acak_jumlah_kendaraan(self):
        """Mengisi kolom input dengan angka tiruan acak baru."""
        if self.dikunci:      # [BARU] lajur berkamera tidak diacak
            return
        for kotak_teks in self.kolom_input.values():
            kotak_teks.delete(0, "end")
            kotak_teks.insert(0, str(random.randint(0, 20)))

    # ---------------- [BARU] ----------------
    def set_jumlah(self, data):
        """Mengisi kolom dari hasil deteksi, mis. {'Motor': 3, 'Mobil': 2, ...}."""
        for jenis, kotak in self.kolom_input.items():
            kotak.configure(state="normal")
            kotak.delete(0, "end")
            kotak.insert(0, str(data.get(jenis, 0)))
            if self.dikunci:
                kotak.configure(state="disabled")

    def kunci(self, status=True):
        self.dikunci = status
        for kotak in self.kolom_input.values():
            kotak.configure(state="disabled" if status else "normal")


# ===============================================================
# APLIKASI UTAMA: ENGINE GUI & CORE LOGIC SIMULATION
# ===============================================================
class TrafficGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup Window Utama Aplikasi
        self.title("Smart Traffic Control System - Deteksi Kamera")
        self.geometry("1450x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # --- STATE ENGINE SIMULASI INTERNAL ---
        self.simulasi_aktif = False
        self.fase_saat_ini = "HIJAU"     # Status: "HIJAU" atau "KUNING"
        self.lajur_aktif = "Utara"       # Menunjukkan lajur mana yang sedang hijau
        self.hitung_mundur = 10          # Durasi detik yang tersisa
        self.lajur_terpaksa = None       # Menyimpan antrean paksa lajur dari mode manual

        # [BARU] State kamera & penjadwalan
        self._id_ticker = None
        self._id_audio_poll = None
        self.waktu_hijau_terakhir = {l: time.time() for l in DAFTAR_LAJUR}
        self.detektor = None
        self.audio_engine = None
        self.audio_detection_status = {l: False for l in DAFTAR_LAJUR}
        self.audio_lane_status = {}
        self.kotak_kamera = {}
        self.label_kamera = {}
        self.gambar_kamera = {}
        self.jumlah_terakhir = {}
        self._lajur_gagal = set()

        # Konfigurasi Master Grid Layout
        self.grid_columnconfigure(0, weight=0, minsize=360)  # Kolom Kontrol Kiri
        self.grid_columnconfigure(1, weight=1)               # Kolom Grafik Tengah
        self.grid_columnconfigure(2, weight=1)               # Kolom Kamera & Data
        self.grid_columnconfigure(3, weight=1)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0, minsize=120)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0, minsize=120)

        # Bangun Komponen Tampilan
        self.bangun_panel_kontrol_kiri()
        self.bangun_grid_simulasi_kanan()

        # Mulai audio detector secara terpisah agar loading model tidak membekukan GUI.
        self.mulai_deteksi_audio()

        # [BARU] Jalankan deteksi kamera
        self.mulai_deteksi()
        self.protocol("WM_DELETE_WINDOW", self.tutup_aplikasi)

        # Visualisasi awal status lampu
        self.sinkronisasi_tampilan_ui()

    # -----------------------------------------------------------
    # TATA LETAK UI & WIDGET GENERATOR
    # -----------------------------------------------------------
    def bangun_panel_kontrol_kiri(self):
        """Membangun bilah kendali simulasi di area sebelah kiri."""
        panel_kiri = ctk.CTkFrame(self, width=340, corner_radius=0, fg_color="#1e1e1e")
        panel_kiri.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(panel_kiri, text="PANEL KENDALI UTAMA", font=("Arial", 16, "bold"), text_color="#FFCC00").pack(pady=15)

        # Wadah Indikator Fisik Lampu
        bingkai_lampu = ctk.CTkFrame(panel_kiri, fg_color="#2b2b2b", border_width=1)
        bingkai_lampu.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(bingkai_lampu, text="Status Lampu Lalu Lintas", font=("Arial", 12, "bold")).pack(pady=5)

        self.visual_lampu = {}
        for lajur in DAFTAR_LAJUR:
            baris = ctk.CTkFrame(bingkai_lampu, fg_color="transparent")
            baris.pack(fill="x", padx=20, pady=4)

            ctk.CTkLabel(baris, text=f"Lajur {lajur}: ", width=100, anchor="w").pack(side="left")

            bulatan = ctk.CTkFrame(baris, width=20, height=20, corner_radius=10, fg_color=HEX_RED)
            bulatan.pack(side="left", padx=10)
            self.visual_lampu[lajur] = bulatan

        # Papan Informasi Countdown Status
        bingkai_info = ctk.CTkFrame(panel_kiri, fg_color="#151515", border_width=1, border_color="#333333")
        bingkai_info.pack(fill="x", padx=15, pady=15)

        self.txt_status = ctk.CTkLabel(bingkai_info, text="Status: Non-Aktif", font=("Arial", 14, "bold"), text_color="yellow")
        self.txt_status.pack(pady=5)

        self.txt_timer = ctk.CTkLabel(bingkai_info, text="-- dtk", font=("Arial", 28, "bold"), text_color="#ffffff")
        self.txt_timer.pack(pady=5)

        # Pilihan Mode Operasional
        self.var_mode = ctk.StringVar(value="OTOMATIS")
        bingkai_mode = ctk.CTkFrame(panel_kiri, fg_color="transparent")
        bingkai_mode.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(bingkai_mode, text="Mode Kendali:", font=("Arial", 11, "bold")).pack(anchor="w", padx=5)
        ctk.CTkRadioButton(bingkai_mode, text="Otomatis (Adaptif PCU)", variable=self.var_mode, value="OTOMATIS").pack(anchor="w", padx=10, pady=2)
        ctk.CTkRadioButton(bingkai_mode, text="Manual (Paksa Klik)", variable=self.var_mode, value="MANUAL").pack(anchor="w", padx=10, pady=2)

        # Tombol Aksi Kendali
        self.btn_mulai = ctk.CTkButton(panel_kiri, text="Mulai Simulasi", fg_color="#27ae60", hover_color="#1e8449", command=self.action_mulai_simulasi)
        self.btn_mulai.pack(fill="x", padx=15, pady=5)

        self.btn_stop = ctk.CTkButton(panel_kiri, text="Hentikan Simulasi", fg_color="#c0392b", hover_color="#962d22", command=self.action_hentikan_simulasi, state="disabled")
        self.btn_stop.pack(fill="x", padx=15, pady=5)

        self.btn_acak = ctk.CTkButton(panel_kiri, text="Acak Lajur Tanpa Kamera", fg_color="#2980b9", hover_color="#1f618d", command=self.action_acak_input_kendaraan)
        self.btn_acak.pack(fill="x", padx=15, pady=5)

    def bangun_grid_simulasi_kanan(self):
        """Membangun penempatan layout grid 4 kamera dan grafik suara."""
        # --- KOLOM 1: AUDIO CONFIDENCE & AREA GRAFIK ---
        f_suara1 = ctk.CTkFrame(self, fg_color=HEX_DARK_BG, border_width=1)
        f_suara1.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        ctk.CTkLabel(
            f_suara1,
            text="AUDIO CONFIDENCE",
            font=("Arial", 12, "bold"),
            text_color="#5dade2",
        ).pack(pady=(8, 1))
        ctk.CTkLabel(
            f_suara1,
            text="Skor sirene per simpang (belum terkalibrasi)",
            font=("Arial", 9),
            text_color="#aab7c4",
        ).pack(pady=(0, 4))

        # Nilai dimulai sebagai None. Persentase baru muncul setelah modul audio
        # mengirim hasil agar GUI tidak menampilkan angka confidence palsu.
        self.audio_confidence_values = {l: None for l in DAFTAR_LAJUR}
        self.audio_confidence_widgets = {}
        for lajur in DAFTAR_LAJUR:
            baris = ctk.CTkFrame(f_suara1, fg_color="transparent")
            baris.pack(fill="x", padx=10, pady=3)

            ctk.CTkLabel(
                baris,
                text=lajur,
                width=52,
                anchor="w",
                font=("Arial", 10),
            ).pack(side="left")

            progress = ctk.CTkProgressBar(
                baris,
                height=10,
                progress_color="#3498db",
            )
            progress.pack(side="left", fill="x", expand=True, padx=(4, 6))
            progress.set(0)

            persen = ctk.CTkLabel(
                baris,
                text="--%",
                width=38,
                anchor="e",
                font=("Arial", 10, "bold"),
            )
            persen.pack(side="right")

            self.audio_confidence_widgets[lajur] = {
                "progress": progress,
                "persen": persen,
            }

        self.lbl_status_audio = ctk.CTkLabel(
            f_suara1,
            text="Menunggu input audio",
            font=("Arial", 9),
            text_color="#aab7c4",
        )
        self.lbl_status_audio.pack(pady=(2, 7))

        f_akustik1 = ctk.CTkFrame(self, fg_color="#151a21")
        f_akustik1.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(f_akustik1, text="Data Akustik 1", text_color="gray").pack(expand=True)

        # Area ini disiapkan untuk grafik yang dikembangkan terpisah.
        f_suara2 = ctk.CTkFrame(self, fg_color=HEX_DARK_BG, border_width=1)
        f_suara2.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(
            f_suara2,
            text="[ Area Grafik Analisis ]\n(Dikembangkan terpisah)",
            text_color="#5dade2",
        ).pack(expand=True)

        f_akustik2 = ctk.CTkFrame(self, fg_color="#151a21")
        f_akustik2.grid(row=3, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(f_akustik2, text="Data Akustik 2", text_color="gray").pack(expand=True)

        # --- KOLOM 2 & 3: BLOK MONITOR KAMERA & DATA INPUT LAJUR ---
        self.kumpulan_panel = {}

        # Kamera 1 - Utara
        self.buat_box_kamera_simulasi(row=0, col=2, nomor=1, nama="Utara")
        self.kumpulan_panel["Utara"] = PanelInputLajur(self, "Utara")
        self.kumpulan_panel["Utara"].grid(row=1, column=2, sticky="nsew", padx=4, pady=4)

        # Kamera 2 - Timur
        self.buat_box_kamera_simulasi(row=0, col=3, nomor=2, nama="Timur")
        self.kumpulan_panel["Timur"] = PanelInputLajur(self, "Timur")
        self.kumpulan_panel["Timur"].grid(row=1, column=3, sticky="nsew", padx=4, pady=4)

        # Kamera 3 - Selatan
        self.buat_box_kamera_simulasi(row=2, col=2, nomor=3, nama="Selatan")
        self.kumpulan_panel["Selatan"] = PanelInputLajur(self, "Selatan")
        self.kumpulan_panel["Selatan"].grid(row=3, column=2, sticky="nsew", padx=4, pady=4)

        # Kamera 4 - Barat
        self.buat_box_kamera_simulasi(row=2, col=3, nomor=4, nama="Barat")
        self.kumpulan_panel["Barat"] = PanelInputLajur(self, "Barat")
        self.kumpulan_panel["Barat"].grid(row=3, column=3, sticky="nsew", padx=4, pady=4)

        # Pasang Tombol Paksa Manual di bagian bawah tiap-tiap blok data lajur
        for lajur in DAFTAR_LAJUR:
            btn_paksa = ctk.CTkButton(
                self.kumpulan_panel[lajur],
                text=f"Paksa Hijau {lajur}",
                width=120,
                height=22,
                fg_color="#34495e",
                command=lambda l=lajur: self.action_pemicu_manual_lajur(l)
            )
            btn_paksa.pack(pady=4)

    def buat_box_kamera_simulasi(self, row, col, nomor, nama):
        f = ctk.CTkFrame(self, fg_color="#242424", border_width=2, border_color="#555555")
        f.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        lbl = ctk.CTkLabel(f, text=f"CAMERA {nomor}\n(Lajur {nama})", font=("Arial", 14, "bold"))
        lbl.pack(expand=True)
        self.kotak_kamera[nama] = f      # [BARU]
        self.label_kamera[nama] = lbl    # [BARU]

    # -----------------------------------------------------------
    # PEMBARUAN AUDIO CONFIDENCE
    # -----------------------------------------------------------
    def perbarui_confidence_audio(self, hasil, deteksi=None):
        """
        Memperbarui skor model sirene untuk tiap simpang.
        Nilai berada di rentang 0.0 sampai 1.0. Skor YAMNet belum terkalibrasi
        sebagai probabilitas dan harus diperlakukan sebagai indikator eksperimen.
        """
        if not isinstance(hasil, dict):
            raise TypeError("hasil audio harus berupa dictionary per lajur")
        if deteksi is not None:
            self.audio_detection_status.update(deteksi)

        tersedia = []
        for lajur in DAFTAR_LAJUR:
            nilai = hasil.get(lajur)
            widget = self.audio_confidence_widgets[lajur]

            if nilai is None:
                self.audio_confidence_values[lajur] = None
                widget["progress"].set(0)
                widget["progress"].configure(progress_color="#3498db")
                widget["persen"].configure(text="--%", text_color="#aab7c4")
                continue

            try:
                confidence = float(nilai)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Skor audio untuk lajur {lajur} harus berupa angka 0.0 sampai 1.0 atau None"
                )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Skor audio untuk lajur {lajur} di luar rentang 0.0 sampai 1.0"
                )

            self.audio_confidence_values[lajur] = confidence
            is_detected = self.audio_detection_status.get(lajur, False)
            widget["progress"].set(confidence)
            widget["progress"].configure(
                progress_color="#e67e22" if is_detected else "#3498db"
            )
            widget["persen"].configure(
                text=f"{confidence * 100:.0f}%",
                text_color="#e67e22" if is_detected else "#ffffff",
            )
            tersedia.append((lajur, confidence))

        indikasi = [
            lajur for lajur in DAFTAR_LAJUR
            if self.audio_detection_status.get(lajur, False)
        ]
        if indikasi:
            self.lbl_status_audio.configure(
                text="Indikasi sirene: " + ", ".join(indikasi) + " | perlu validasi visual",
                text_color="#e67e22",
            )
        elif tersedia:
            lajur_tertinggi, skor_tertinggi = max(tersedia, key=lambda item: item[1])
            self.lbl_status_audio.configure(
                text=f"Skor tertinggi: {lajur_tertinggi} ({skor_tertinggi * 100:.0f}%)",
                text_color="#5dade2",
            )

    def mulai_deteksi_audio(self):
        """Jalankan classifier audio pada thread latar dan poll hasil pada thread GUI."""
        try:
            from siren_audio import SirenAudioEngine
        except ImportError as exc:
            self.lbl_status_audio.configure(
                text=f"Modul audio tidak tersedia: {exc}",
                text_color="#e74c3c",
            )
            return

        sumber_aktif = {
            lajur: sumber
            for lajur, sumber in SUMBER_AUDIO.items()
            if sumber is not None and str(sumber).strip()
        }
        if not sumber_aktif:
            self.lbl_status_audio.configure(
                text="Belum ada sumber audio. Atur SUMBER_AUDIO di Coba_kamera.py.",
                text_color="#aab7c4",
            )
            return

        try:
            self.audio_engine = SirenAudioEngine(sumber_aktif)
            self.audio_engine.start()
            self._id_audio_poll = self.after(250, self.poll_deteksi_audio)
        except Exception as exc:
            self.audio_engine = None
            self.lbl_status_audio.configure(
                text=f"Audio detector gagal dimulai: {exc}",
                text_color="#e74c3c",
            )

    def poll_deteksi_audio(self):
        """Ambil hasil audio dan perbarui panel pada thread utama Tkinter."""
        engine = self.audio_engine
        if engine is None:
            return

        try:
            events = engine.drain_events()
        except Exception as exc:
            self.lbl_status_audio.configure(
                text=f"Gagal membaca hasil audio: {exc}",
                text_color="#e74c3c",
            )
            self._id_audio_poll = self.after(500, self.poll_deteksi_audio)
            return

        for event in events:
            event_type = event.get("type")
            lajur = event.get("lane")

            if event_type == "status":
                pesan = str(event.get("message", "Status audio berubah"))
                if lajur in DAFTAR_LAJUR:
                    self.audio_lane_status[lajur] = pesan
                    # Pesan sumber hanya ditampilkan jika tidak sedang ada indikasi sirene.
                    if not any(self.audio_detection_status.values()):
                        self.lbl_status_audio.configure(
                            text=f"{lajur}: {pesan}",
                            text_color="#e74c3c" if (
                                "gagal" in pesan.casefold()
                                or "tidak ditemukan" in pesan.casefold()
                                or "bermasalah" in pesan.casefold()
                            ) else "#aab7c4",
                        )
                elif pesan:
                    self.lbl_status_audio.configure(
                        text=pesan,
                        text_color="#e74c3c" if (
                            "gagal" in pesan.casefold()
                            or "tidak ditemukan" in pesan.casefold()
                        ) else "#aab7c4",
                    )

            elif event_type == "confidence" and lajur in DAFTAR_LAJUR:
                self.audio_detection_status[lajur] = bool(event.get("detected", False))
                nilai_sekarang = dict(self.audio_confidence_values)
                nilai_sekarang[lajur] = event.get("confidence")
                self.perbarui_confidence_audio(
                    nilai_sekarang,
                    deteksi=self.audio_detection_status,
                )

        self._id_audio_poll = self.after(250, self.poll_deteksi_audio)

    # -----------------------------------------------------------
    # [BARU] INTEGRASI KAMERA
    # -----------------------------------------------------------
    def _teks_kamera(self, lajur, pesan):
        nomor = DAFTAR_LAJUR.index(lajur) + 1
        return f"CAMERA {nomor}\n(Lajur {lajur})\n{pesan}"

    def mulai_deteksi(self):
        self.lajur_berkamera = [l for l in DAFTAR_LAJUR if SUMBER_KAMERA[l] is not None]
        if not self.lajur_berkamera:
            return
        for l in self.lajur_berkamera:
            panel = self.kumpulan_panel[l]
            panel.set_jumlah({})            # mulai dari nol
            panel.kunci(True)               # isian diatur kamera
            self.label_kamera[l].configure(text=self._teks_kamera(l, "Memuat model YOLO..."))
        self.detektor = DetektorKamera()
        self.detektor.start()
        self.after(200, self.poll_kamera)

    def poll_kamera(self):
        """Dipanggil berkala di thread GUI: ambil hasil deteksi, perbarui feed dan angka."""
        det = self.detektor
        if det is None:
            return
        if det.siap:
            if det.error:
                for l in self.lajur_berkamera:
                    self.label_kamera[l].configure(image=None, text=self._teks_kamera(l, "Gagal memuat deteksi\n(input manual)"))
                    self.kumpulan_panel[l].kunci(False)
                print("[!] Detektor gagal:", det.error)
                return
            jumlah, frames = det.ambil()
            for l in self.lajur_berkamera:
                panel = self.kumpulan_panel[l]
                if l not in det.caps:
                    if l not in self._lajur_gagal:
                        self._lajur_gagal.add(l)
                        self.label_kamera[l].configure(text=self._teks_kamera(l, "Kamera tidak terbuka\n(input manual)"))
                        panel.kunci(False)
                    continue
                frame = frames[l]
                if frame is not None:
                    rgb = cv2.cvtColor(cv2.resize(frame, UKURAN_TAMPIL), cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    if l not in self.gambar_kamera:
                        self.gambar_kamera[l] = ctk.CTkImage(light_image=pil, dark_image=pil, size=UKURAN_TAMPIL)
                        self.label_kamera[l].configure(image=self.gambar_kamera[l], text="")
                    else:
                        self.gambar_kamera[l].configure(light_image=pil, dark_image=pil)
                if jumlah[l] != self.jumlah_terakhir.get(l):
                    self.jumlah_terakhir[l] = jumlah[l]
                    panel.set_jumlah(jumlah[l])
        self.after(120, self.poll_kamera)

    def tutup_aplikasi(self):
        if self._id_ticker is not None:
            self.after_cancel(self._id_ticker)
        if self._id_audio_poll is not None:
            self.after_cancel(self._id_audio_poll)
        if self.audio_engine is not None:
            self.audio_engine.stop()
        if self.detektor is not None:
            self.detektor.stop()
        self.destroy()

    # -----------------------------------------------------------
    # ENGINE CORE SIMULATION LOGIC
    # -----------------------------------------------------------
    def tentukan_lajur_paling_padat(self, lajur_kecuali):
        """Mencari lajur dengan antrean paling padat untuk giliran berikutnya.
        [PERBAIKAN] Jika skor sama, pilih yang paling lama menunggu (versi asli selalu
        memilih kandidat pertama sehingga Selatan/Barat bisa tidak pernah dapat giliran)."""
        kandidat = [l for l in DAFTAR_LAJUR if l != lajur_kecuali]
        now = time.time()
        return max(kandidat, key=lambda l: (self.kumpulan_panel[l].ambil_skor_bobot(),
                                            now - self.waktu_hijau_terakhir[l]))

    def hitung_durasi_fase_hijau(self, lajur_target):
        """
        Membagi total waktu hijau 120 detik secara proporsional terhadap skor,
        sambil menjaga setiap lajur di antara FASE_HIJAU_MIN dan FASE_HIJAU_MAX.
        Waktu kuning dihitung terpisah.
        """
        if lajur_target not in DAFTAR_LAJUR:
            raise ValueError(f"Lajur tidak dikenal: {lajur_target!r}")

        total_waktu = 120
        jumlah_lajur = len(DAFTAR_LAJUR)
        batas_min = FASE_HIJAU_MIN
        batas_max = FASE_HIJAU_MAX

        if (
            batas_min <= 0
            or batas_max < batas_min
            or jumlah_lajur * batas_min > total_waktu
            or jumlah_lajur * batas_max < total_waktu
        ):
            raise ValueError(
                "Batas durasi hijau tidak dapat memenuhi total waktu siklus 120 detik."
            )

        skor = {
            lajur: max(0.0, float(self.kumpulan_panel[lajur].ambil_skor_bobot()))
            for lajur in DAFTAR_LAJUR
        }

        # Setiap lajur mendapat waktu minimum terlebih dahulu.
        tambahan = {lajur: 0.0 for lajur in DAFTAR_LAJUR}
        sisa = float(total_waktu - jumlah_lajur * batas_min)
        kapasitas_per_lajur = float(batas_max - batas_min)
        aktif = list(DAFTAR_LAJUR)

        # Distribusikan sisa waktu ke lajur aktif. Jika lajur mencapai maksimum,
        # bekukan alokasinya lalu bagikan kembali sisa waktu kepada lajur lain.
        while aktif and sisa > 1e-9:
            total_skor_aktif = sum(skor[lajur] for lajur in aktif)
            if total_skor_aktif <= 0:
                bobot = {lajur: 1.0 for lajur in aktif}
            else:
                bobot = {lajur: skor[lajur] for lajur in aktif}
            total_bobot = sum(bobot.values())

            usulan = {
                lajur: sisa * bobot[lajur] / total_bobot
                for lajur in aktif
            }
            jenuh = [
                lajur for lajur in aktif
                if usulan[lajur] >= kapasitas_per_lajur - tambahan[lajur] - 1e-9
            ]

            if jenuh:
                waktu_dibagikan = 0.0
                for lajur in jenuh:
                    ruang = max(0.0, kapasitas_per_lajur - tambahan[lajur])
                    tambahan[lajur] += ruang
                    waktu_dibagikan += ruang
                sisa = max(0.0, sisa - waktu_dibagikan)
                aktif = [lajur for lajur in aktif if lajur not in jenuh]
            else:
                for lajur in aktif:
                    tambahan[lajur] += usulan[lajur]
                sisa = 0.0

        durasi_tepat = {
            lajur: batas_min + tambahan[lajur]
            for lajur in DAFTAR_LAJUR
        }
        hasil = {
            lajur: int(durasi_tepat[lajur])
            for lajur in DAFTAR_LAJUR
        }
        sisa_detik = total_waktu - sum(hasil.values())

        urutan = sorted(
            DAFTAR_LAJUR,
            key=lambda lajur: (
                durasi_tepat[lajur] - hasil[lajur],
                skor[lajur],
                -DAFTAR_LAJUR.index(lajur),
            ),
            reverse=True,
        )
        while sisa_detik > 0:
            kandidat = [lajur for lajur in urutan if hasil[lajur] < batas_max]
            if not kandidat:
                raise RuntimeError("Sisa waktu hijau tidak dapat dialokasikan.")
            for lajur in kandidat:
                if sisa_detik <= 0:
                    break
                hasil[lajur] += 1
                sisa_detik -= 1

        return hasil[lajur_target]

    def pilih_lajur_berikut(self):
        """[BARU] Menentukan lajur hijau berikutnya setelah fase kuning selesai."""
        if self.var_mode.get() == "MANUAL" or PILIH_LAJUR_BERIKUT == "BERGILIR":
            i = DAFTAR_LAJUR.index(self.lajur_aktif)
            return DAFTAR_LAJUR[(i + 1) % len(DAFTAR_LAJUR)]

        # OTOMATIS: anti-starvation dulu, lalu yang terpadat
        now = time.time()
        kandidat = [l for l in DAFTAR_LAJUR if l != self.lajur_aktif]
        terlama = max(kandidat, key=lambda l: now - self.waktu_hijau_terakhir[l])
        if now - self.waktu_hijau_terakhir[terlama] > BATAS_TUNGGU_MAKS:
            return terlama
        return self.tentukan_lajur_paling_padat(self.lajur_aktif)

    def siklus_transisi_fase(self):
        """[PERBAIKAN] Method ini dipanggil ticker tetapi tidak ada di kode asli
        (akan menyebabkan AttributeError saat hitung mundur habis)."""
        if self.fase_saat_ini == "HIJAU":
            self.fase_saat_ini = "KUNING"
            self.hitung_mundur = FASE_KUNING
        else:
            # kuning selesai -> pindah ke lajur berikutnya
            self.waktu_hijau_terakhir[self.lajur_aktif] = time.time()
            if self.lajur_terpaksa:
                lajur_baru = self.lajur_terpaksa
                self.lajur_terpaksa = None
            else:
                lajur_baru = self.pilih_lajur_berikut()
            self.lajur_aktif = lajur_baru
            self.fase_saat_ini = "HIJAU"
            self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)

    def sinkronisasi_tampilan_ui(self):
        """Memperbarui warna bulatan lampu, countdown, dan bingkai kamera."""
        for lajur, widget in self.visual_lampu.items():
            if lajur == self.lajur_aktif:
                warna = HEX_GREEN if self.fase_saat_ini == "HIJAU" else HEX_YELLOW
            else:
                warna = HEX_RED
            widget.configure(fg_color=warna)
            # [BARU] bingkai kamera mengikuti warna lampu
            self.kotak_kamera[lajur].configure(border_color=warna if self.simulasi_aktif else "#555555")

        if self.simulasi_aktif:
            warna_teks = HEX_GREEN if self.fase_saat_ini == "HIJAU" else HEX_YELLOW
            self.txt_status.configure(text=f"Status: {self.fase_saat_ini} @ {self.lajur_aktif}", text_color=warna_teks)
            self.txt_timer.configure(text=f"{max(self.hitung_mundur, 0)} dtk", text_color="#ffffff")
        else:
            self.txt_status.configure(text="Simulasi Dihentikan", text_color="orange")
            self.txt_timer.configure(text="-- dtk", text_color="gray")

    # -----------------------------------------------------------
    # INTERACTION & TIMING EVENTS
    # -----------------------------------------------------------
    def engine_simulation_ticker(self):
        """Ticker mandiri yang berjalan mundur 1 detik secara periodik."""
        if not self.simulasi_aktif:
            return

        self.hitung_mundur -= 1

        if self.hitung_mundur <= 0:
            self.siklus_transisi_fase()

        self.sinkronisasi_tampilan_ui()

        # [PERBAIKAN] id disimpan agar bisa dibatalkan (hindari ticker ganda saat stop lalu start cepat)
        self._id_ticker = self.after(1000, self.engine_simulation_ticker)

    def action_mulai_simulasi(self):
        if self.simulasi_aktif:
            return
        self.simulasi_aktif = True
        self.btn_mulai.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_acak.configure(state="disabled")

        # Reset fase awal dan hitung durasi hijau lajur aktif dari data saat ini
        self.fase_saat_ini = "HIJAU"
        self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)

        self.sinkronisasi_tampilan_ui()
        self._id_ticker = self.after(1000, self.engine_simulation_ticker)

    def action_hentikan_simulasi(self):
        self.simulasi_aktif = False
        if self._id_ticker is not None:
            self.after_cancel(self._id_ticker)
            self._id_ticker = None
        self.btn_mulai.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_acak.configure(state="normal")
        self.sinkronisasi_tampilan_ui()

    def action_acak_input_kendaraan(self):
        for panel in self.kumpulan_panel.values():
            panel.acak_jumlah_kendaraan()      # lajur berkamera dilewati
        if self.fase_saat_ini == "HIJAU" and self.simulasi_aktif:
            self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)
            self.txt_timer.configure(text=f"{self.hitung_mundur} dtk")

    def action_pemicu_manual_lajur(self, lajur_tujuan):
        """Memaksa sistem memindahkan lampu hijau ke lajur pilihan (hanya Mode Manual)."""
        if self.var_mode.get() != "MANUAL" or not self.simulasi_aktif:
            return

        # Jika lajur yang diklik sudah sedang hijau, abaikan
        if lajur_tujuan == self.lajur_aktif and self.fase_saat_ini == "HIJAU":
            return

        # Pemicu transisi kuning terlebih dahulu sebelum pindah
        self.fase_saat_ini = "KUNING"
        self.hitung_mundur = FASE_KUNING
        self.lajur_terpaksa = lajur_tujuan
        self.sinkronisasi_tampilan_ui()


# ===============================================================
# APLIKASI RUNNER (MAIN LAUNCHER)
# ===============================================================
if __name__ == "__main__":
    app = TrafficGUI()
    app.mainloop()
