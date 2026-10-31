#!/usr/bin/env python3
"""TAHAP D: uji satu URL HLS secara bertahap (requests -> ffprobe -> OpenCV).
Pemakaian: python test_stream.py <URL_playlist.m3u8> [--origin URL] [--referer URL]
Tidak menyimpan video; hanya membaca 1 manifest, 1 chunklist, 1 segmen, 1 frame."""
import argparse, json, shutil, subprocess, urllib.parse as up
import requests

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("--origin"); ap.add_argument("--referer")
a = ap.parse_args()
H = {"User-Agent": "Mozilla/5.0 (stream-compat-test)"}
if a.origin: H["Origin"] = a.origin
if a.referer: H["Referer"] = a.referer
print("Header yang dikirim:", H)

def get(u, **kw):
    r = requests.get(u, headers=H, timeout=15, **kw)
    print(f"  GET {u}\n  -> {r.status_code} {r.headers.get('content-type')} {len(r.content)} bytes")
    return r

print("\n[1] Master playlist")
r = get(a.url)
if r.status_code != 200 or "#EXTM3U" not in r.text:
    raise SystemExit("GAGAL: bukan playlist HLS valid. Coba ulangi dengan --origin/--referer yang DIKIRIM BROWSER (lihat HAR).")
print("  ", r.text[:300].replace("\n", "\n   "))
lines = [l.strip() for l in r.text.splitlines()]
variants = [l for l in lines if l and not l.startswith("#")]
chunk_url = up.urljoin(a.url, variants[0]) if variants else a.url

print("\n[2] Chunklist (media playlist)")
r2 = get(chunk_url)
segs = [l.strip() for l in r2.text.splitlines() if l.strip() and not l.startswith("#")]
print(f"   jumlah segmen: {len(segs)} | ENDLIST: {'#EXT-X-ENDLIST' in r2.text} (False = live)")
if not segs: raise SystemExit("Tidak ada segmen di chunklist.")

print("\n[3] Segmen pertama")
r3 = get(up.urljoin(chunk_url, segs[-1]))
print("   sync byte MPEG-TS (0x47):", r3.content[:1] == b"\x47")

print("\n[4] ffprobe")
if shutil.which("ffprobe"):
    cmd = ["ffprobe", "-v", "error", "-rw_timeout", "10000000", "-show_streams", "-of", "json"]
    hdr = "".join(f"{k}: {v}\r\n" for k, v in H.items())
    p = subprocess.run(cmd + ["-headers", hdr, a.url], capture_output=True, text=True, timeout=40)
    if p.returncode == 0:
        for s in json.loads(p.stdout).get("streams", []):
            print("  ", {k: s.get(k) for k in ("codec_type", "codec_name", "width", "height", "avg_frame_rate")})
    else:
        print("   ffprobe gagal:", p.stderr[:300])
else:
    print("   ffprobe tidak ditemukan (pasang FFmpeg dulu), dilewati.")

print("\n[5] OpenCV: baca 1 frame")
import os, cv2
opts = ["rw_timeout;10000000"]
if len(H) > 1: opts.append("headers;" + "".join(f"{k}: {v}\r\n" for k, v in H.items()))
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)
cap = cv2.VideoCapture(a.url, cv2.CAP_FFMPEG)
ok, frame = (False, None)
if cap.isOpened():
    for _ in range(50):
        ok, frame = cap.read()
        if ok: break
cap.release()
print("   frame terbaca:", ok, "| ukuran:", None if frame is None else frame.shape)
print("\nHASIL:", "KOMPATIBEL dengan OpenCV" if ok else "OpenCV gagal (lihat langkah 1-4)")
