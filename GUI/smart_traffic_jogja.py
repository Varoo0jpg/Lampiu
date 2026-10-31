"""
Launcher integrasi CCTV Jogja untuk Smart Traffic Lampiu.

File ini sengaja menggunakan ulang GUI, model, detektor YOLO, panel data,
dan logika simulasi dari smart_traffic_gui_100.py. Tidak ada bobot atau
pipeline existing yang disalin maupun dimodifikasi.

Jalankan dari root repository:
    python GUI/smart_traffic_jogja.py

Catatan:
- CCTV hanya dipakai sebagai input video untuk analisis dan simulasi.
- Tidak ada koneksi ke perangkat pengendali lampu fisik.
- Status alive di cameras.json adalah hasil pemeriksaan sebelumnya, bukan
  jaminan stream sedang aktif saat aplikasi dijalankan.
"""

import json
import os
import sys
from urllib.parse import urlparse

import tkinter as tk
from tkinter import StringVar, filedialog, messagebox, ttk

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERAS_FILE = os.path.join(GUI_DIR, "cameras.json")

# Pastikan modul GUI existing dapat diimpor saat file dijalankan dari root repo.
if GUI_DIR not in sys.path:
    sys.path.insert(0, GUI_DIR)

import smart_traffic_gui_100 as base_gui


HOST_CCTV_DIIZINKAN = "cctvjss.jogjakota.go.id"
OPSIONAL_NONAKTIF = "Nonaktif"
OPSIONAL_BAWAAN = "Sumber bawaan"
OPSIONAL_WEBCAM = ("Webcam 0", "Webcam 1", "Webcam 2")


def muat_daftar_cctv(path=CAMERAS_FILE):
    """Muat hanya URL HLS HTTPS dari host CCTV Jogja yang sudah dikenal."""
    if not os.path.isfile(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Gagal membaca {path}: {exc}") from exc

    if not isinstance(data, list):
        raise RuntimeError("Format cameras.json harus berupa daftar kamera.")

    hasil = []
    for item in data:
        if not isinstance(item, dict):
            continue

        url = str(item.get("url", "")).strip()
        parsed = urlparse(url)

        # Pembatasan host mencegah JSON lokal mengarahkan aplikasi ke host lain.
        if (
            parsed.scheme != "https"
            or parsed.hostname != HOST_CCTV_DIIZINKAN
            or not parsed.path.lower().endswith(".m3u8")
        ):
            continue

        # Detektor existing tidak memasang custom HTTP headers. Lewati entri
        # yang memerlukannya daripada mengabaikan autentikasi secara diam-diam.
        headers = item.get("headers") or {}
        if headers:
            continue

        if item.get("alive") is not True:
            continue

        camera_id = str(item.get("id", "")).strip()
        name = str(item.get("name", "Kamera tanpa nama")).strip()
        if not camera_id:
            continue

        hasil.append({
            "id": camera_id,
            "name": name,
            "url": url,
            "label": f'{camera_id} | {name}',
        })

    hasil.sort(key=lambda camera: (camera["name"].casefold(), camera["id"]))
    return hasil


class PemilihSumber(tk.Tk):
    """Dialog awal Tkinter standar; tidak membuat root CustomTkinter kedua."""

    def __init__(self, cameras):
        super().__init__()
        self.title("Lampiu — Pemilihan Sumber Video")
        self.geometry("940x560")
        self.minsize(780, 500)
        self.configure(bg="#171d26")

        self.cameras = cameras
        self.camera_by_label = {camera["label"]: camera for camera in cameras}
        self.variables = {}
        self.video_paths = {}
        self.selected_sources = None

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Lampiu.TCombobox",
            fieldbackground="#263241",
            background="#263241",
            foreground="#f2f5f8",
            arrowcolor="#f2f5f8",
            padding=6,
        )
        style.map(
            "Lampiu.TCombobox",
            fieldbackground=[("readonly", "#263241")],
            foreground=[("readonly", "#f2f5f8")],
            selectbackground=[("readonly", "#263241")],
            selectforeground=[("readonly", "#f2f5f8")],
        )
        style.configure(
            "Lampiu.TButton",
            background="#2878b8",
            foreground="#ffffff",
            padding=(10, 7),
            borderwidth=0,
        )
        style.map("Lampiu.TButton", background=[("active", "#1f6399")])
        style.configure(
            "Start.TButton",
            background="#218c54",
            foreground="#ffffff",
            padding=(12, 10),
            borderwidth=0,
        )
        style.map("Start.TButton", background=[("active", "#176b3f")])

        container = tk.Frame(self, bg="#171d26", padx=22, pady=18)
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="PILIH SUMBER VIDEO PER LAJUR",
            bg="#171d26",
            fg="#5dade2",
            font=("Arial", 19, "bold"),
        ).pack(anchor="w")
        tk.Label(
            container,
            text=(
                "Pilih CCTV Jogja, sumber bawaan, webcam, atau video lokal. "
                "Untuk pengujian pertama, disarankan memilih satu CCTV."
            ),
            bg="#171d26",
            fg="#b0b8c2",
            font=("Arial", 10),
            wraplength=850,
            justify="left",
        ).pack(anchor="w", pady=(6, 18))

        panel = tk.Frame(
            container,
            bg="#202a36",
            highlightbackground="#354354",
            highlightthickness=1,
            padx=14,
            pady=14,
        )
        panel.pack(fill="both", expand=True)

        tk.Label(
            panel,
            text="KONFIGURASI EMPAT LAJUR",
            bg="#202a36",
            fg="#e8edf2",
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        panel.grid_columnconfigure(1, weight=1)
        options = [
            OPSIONAL_NONAKTIF,
            OPSIONAL_BAWAAN,
            *OPSIONAL_WEBCAM,
            *(camera["label"] for camera in cameras),
        ]

        for row, lane in enumerate(base_gui.DAFTAR_LAJUR, start=1):
            tk.Label(
                panel,
                text=f"Lajur {lane}",
                bg="#202a36",
                fg="#e8edf2",
                font=("Arial", 11, "bold"),
                width=13,
                anchor="w",
            ).grid(row=row, column=0, padx=(2, 12), pady=9, sticky="w")

            variable = StringVar(master=self, value=OPSIONAL_NONAKTIF)
            self.variables[lane] = variable
            combo = ttk.Combobox(
                panel,
                textvariable=variable,
                values=options,
                state="readonly",
                style="Lampiu.TCombobox",
            )
            combo.grid(row=row, column=1, padx=6, pady=9, sticky="ew")

            ttk.Button(
                panel,
                text="Pilih video...",
                style="Lampiu.TButton",
                command=lambda selected_lane=lane: self.pilih_video_lokal(selected_lane),
            ).grid(row=row, column=2, padx=(8, 2), pady=9)

        status = (
            f"{len(cameras)} kamera CCTV tersedia di daftar lokal."
            if cameras
            else "Daftar CCTV kosong/tidak tersedia; webcam dan video lokal tetap dapat digunakan."
        )
        tk.Label(
            container,
            text=status,
            bg="#171d26",
            fg="#aab7c4",
            font=("Arial", 9),
        ).pack(anchor="w", pady=(12, 8))

        ttk.Button(
            container,
            text="Jalankan Smart Traffic Lampiu",
            style="Start.TButton",
            command=self.mulai_aplikasi,
        ).pack(fill="x")

        self.protocol("WM_DELETE_WINDOW", self.batal)

    def pilih_video_lokal(self, lane):
        path = filedialog.askopenfilename(
            parent=self,
            title=f"Pilih video untuk lajur {lane}",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("Semua file", "*.*"),
            ],
        )
        if path:
            self.video_paths[lane] = path
            self.variables[lane].set(f"Video lokal: {os.path.basename(path)}")

    def mulai_aplikasi(self):
        sources = {}
        for lane, variable in self.variables.items():
            choice = variable.get()

            if lane in self.video_paths and choice.startswith("Video lokal: "):
                sources[lane] = self.video_paths[lane]
            elif choice == OPSIONAL_NONAKTIF:
                sources[lane] = None
            elif choice == OPSIONAL_BAWAAN:
                sources[lane] = base_gui.SUMBER_KAMERA.get(lane)
            elif choice.startswith("Webcam "):
                try:
                    sources[lane] = int(choice.rsplit(" ", 1)[1])
                except ValueError:
                    sources[lane] = None
            elif choice in self.camera_by_label:
                sources[lane] = self.camera_by_label[choice]["url"]
            else:
                sources[lane] = None

        if not any(source is not None for source in sources.values()):
            proceed = messagebox.askyesno(
                "Tidak ada sumber video",
                "Semua lajur disetel Nonaktif. GUI tetap dapat dibuka dengan input manual. Lanjutkan?",
                parent=self,
            )
            if not proceed:
                return

        self.selected_sources = sources
        self.destroy()

    def batal(self):
        self.selected_sources = None
        self.destroy()


def main():
    camera_error = None
    try:
        cameras = muat_daftar_cctv()
    except RuntimeError as exc:
        camera_error = str(exc)
        cameras = []

    # Root Tk standar hanya dipakai untuk pemilih sumber. Setelah ditutup,
    # satu-satunya root CustomTkinter yang dibuat adalah TrafficGUI.
    picker = PemilihSumber(cameras)
    if camera_error:
        picker.after(
            150,
            lambda: messagebox.showerror(
                "Data CCTV tidak valid", camera_error, parent=picker
            ),
        )
    picker.mainloop()

    if picker.selected_sources is None:
        return

    # Hanya mengganti konfigurasi sumber pada proses launcher ini.
    # Model, detektor YOLO, dan mesin simulasi tetap berada di modul utama.
    base_gui.SUMBER_KAMERA = dict(picker.selected_sources)

    app = base_gui.TrafficGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
