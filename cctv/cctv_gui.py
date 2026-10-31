#!/usr/bin/env python3
"""GUI minimal: pilih kamera dari cameras.json -> tampilkan stream via OpenCV/FFmpeg.
pip install opencv-python pillow
cameras.json (isi HANYA dengan URL yang sudah diverifikasi):
[{"id":"cam01","name":"Nama kamera","url":"<STREAM_URL>","headers":{"Referer":"<jika browser memang mengirimnya>"}}]
"""
import json, os, threading, time, tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk


class StreamReader(threading.Thread):
    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam, self.stop_evt = cam, threading.Event()
        self.frame, self.status, self.lock = None, "Idle", threading.Lock()

    def _open(self):
        opts = ["rw_timeout;10000000"]  # 10 dtk (mikrodetik)
        hdr = self.cam.get("headers") or {}
        if hdr:
            opts.append("headers;" + "".join(f"{k}: {v}\r\n" for k, v in hdr.items()))
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)
        return cv2.VideoCapture(self.cam["url"], cv2.CAP_FFMPEG)

    def run(self):
        backoff = 1
        while not self.stop_evt.is_set():
            self.status = "Connecting..."
            cap = self._open()
            if not cap.isOpened():
                self.status = f"Gagal membuka stream, retry {backoff}s"
                cap.release()
                self.stop_evt.wait(backoff)
                backoff = min(backoff * 2, 30)
                continue
            backoff, fails = 1, 0
            self.status = "Live"
            while not self.stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                with self.lock:
                    self.frame = frame
            cap.release()
            self.status = "Terputus, reconnect..."
            self.stop_evt.wait(2)
        self.status = "Stopped"

    def latest(self):
        with self.lock:
            return self.frame


class App(tk.Tk):
    def __init__(self, cams):
        super().__init__()
        self.title("CCTV Viewer")
        self.cams = {f'{c["name"]} [{c.get("id","?")}]': c for c in cams if c.get("alive", True)}
        top = ttk.Frame(self); top.pack(fill="x", padx=8, pady=6)
        self.combo = ttk.Combobox(top, values=list(self.cams), state="readonly", width=45)
        self.combo.pack(side="left")
        if self.cams: self.combo.current(0)
        ttk.Button(top, text="Connect", command=self.connect).pack(side="left", padx=4)
        ttk.Button(top, text="Stop", command=self.disconnect).pack(side="left")
        self.status = ttk.Label(top, text="Idle"); self.status.pack(side="right")
        self.view = ttk.Label(self); self.view.pack(padx=8, pady=8)
        self.reader = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(33, self.tick)

    def connect(self):
        self.disconnect()
        cam = self.cams.get(self.combo.get())
        if cam:
            self.reader = StreamReader(cam); self.reader.start()

    def disconnect(self):
        if self.reader:
            self.reader.stop_evt.set(); self.reader = None
            self.view.configure(image="")

    def tick(self):
        if self.reader:
            self.status.configure(text=self.reader.status)
            f = self.reader.latest()
            if f is not None:
                f = cv2.resize(f, (960, int(f.shape[0] * 960 / f.shape[1])))
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                self.view.configure(image=img); self.view.image = img
        self.after(33, self.tick)

    def on_close(self):
        self.disconnect(); self.destroy()


if __name__ == "__main__":
    with open("cameras.json", encoding="utf-8") as fh:
        App(json.load(fh)).mainloop()
