#!/usr/bin/env python3
"""Cetak potongan kode situs yang memuat daftar kamera & pembentukan URL stream.
Hanya GET file publik. Jalankan: python dump_sources.py > hasil_C.txt"""
import requests
B = "https://cctv.jogjakota.go.id/"
H = {"User-Agent": "Mozilla/5.0 (analysis script)"}
def show(title, text, ranges=None):
    print(f"\n===== {title} =====")
    L = text.splitlines()
    for a, b in (ranges or [(1, len(L))]):
        print(f"--- baris {a}-{b} ---")
        for i in range(a - 1, min(b, len(L))):
            print(f"{i+1}: {L[i]}")
html = requests.get(B, headers=H, timeout=20).text
show("index.html (inline JS)", html, [(140, 215), (300, 330), (355, 385)])
show("assets/js/wowza.js", requests.get(B + "assets/js/wowza.js", headers=H, timeout=20).text)
