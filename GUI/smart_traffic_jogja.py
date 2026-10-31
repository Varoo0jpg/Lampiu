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

import customtkinter as ctk
from tkinter import StringVar, filedialog, messagebox

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


class PemilihSumber(ctk.CTk):
    """Dialog awal untuk memilih sumber video masing-masing lajur."""

    def __init__(self, cameras):
        super().__init__()
        self.title("Lampiu — Pemilihan Sumber Video")
        self.geometry("900x720")
        self.minsize(760, 560)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.cameras = cameras
        self.camera_by_label = {camera["label"]: camera for camera in cameras}
        self.variables = {}
        self.video_paths = {}
        self.selected_sources = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        ctk.CTkLabel(
            header,
            text="PILIH SUMBER VIDEO PER LAJUR",
            font=("Arial", 19, "bold"),
            text_color="#5dade2",
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text=(
                "CCTV Jogja, sumber bawaan, webcam, atau video lokal. "
                "Untuk pengujian pertama, disarankan memilih satu CCTV."
            ),
            font=("Arial", 11),
            text_color="#b0b8c2",
            wraplength=820,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))

        body = ctk.CTkScrollableFrame(self, label_text="Konfigurasi empat lajur")
        body.grid(row=1, column=0, sticky="nsew", padx=18, pady=8)
        body.grid_columnconfigure(1, weight=1)

        options = [
            OPSIONAL_NONAKTIF,
            OPSIONAL_BAWAAN,
            *OPSIONAL_WEBCAM,
            *(camera["label"] for camera in cameras),
        ]

        for row, lane in enumerate(base_gui.DAFTAR_LAJUR):
            ctk.CTkLabel(
                body,
                text=lane,
                font=("Arial", 13, "bold"),
                width=90,
                anchor="w",
            ).grid(row=row, column=0, padx=(8, 12), pady=12, sticky="w")

            variable = StringVar(value=OPSIONAL_NONAKTIF)
            self.variables[lane] = variable

            combo = ctk.CTkComboBox(
                body,
                values=options,
                variable=variable,
                state="readonly",
                width=520,
            )
            combo.grid(row=row, column=1, padx=6, pady=12, sticky="ew")

            button = ctk.CTkButton(
                body,
                text="Pilih video...",
                width=120,
                command=lambda selected_lane=lane: self.pilih_video_lokal(selected_lane),
            )
            button.grid(row=row, column=2, padx=(6, 8), pady=12)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(8, 16))
        footer.grid_columnconfigure(0, weight=1)

        status = (
            f"{len(cameras)} kamera CCTV tersedia di daftar lokal."
            if cameras
            else "Daftar CCTV tidak tersedia atau tidak memiliki entri yang lolos validasi."
        )
        ctk.CTkLabel(
            footer,
            text=status,
            font=("Arial", 10),
            text_color="#aab7c4",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        ctk.CTkButton(
            footer,
            text="Jalankan Smart Traffic Lampiu",
            height=40,
            fg_color="#218c54",
            hover_color="#176b3f",
            command=self.mulai_aplikasi,
        ).grid(row=1, column=0, sticky="ew")

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
    try:
        cameras = muat_daftar_cctv()
    except RuntimeError as exc:
        messagebox.showerror("Data CCTV tidak valid", str(exc))
        cameras = []

    picker = PemilihSumber(cameras)
    picker.mainloop()

    if picker.selected_sources is None:
        return

    # Hanya mengganti konfigurasi sumber pada proses launcher ini.
    # smart_traffic_gui_100.py dan file lain di repository tidak diubah.
    base_gui.SUMBER_KAMERA = dict(picker.selected_sources)

    app = base_gui.TrafficGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
