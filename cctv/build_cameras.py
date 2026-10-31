#!/usr/bin/env python3
"""Susun cameras.json dari endpoint publik yang dipakai situs: GET /home/getdata (JSON).
Hanya kamera berstatus PUBLIK (cctv_status == 0) yang dimasukkan. Status 1 (marker 'private')
dan 2 (maintenance) dilewati dan tidak diakses.
Pemakaian: python build_cameras.py [--check]
  --check : tes ringan tiap playlist (1 GET per kamera, jeda 0.5 dtk) lalu tandai hidup/mati."""
import argparse, json, time, collections
import requests

BASE = "https://cctv.jogjakota.go.id/"
H = {"User-Agent": "Mozilla/5.0 (cctv-client)"}
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
a = ap.parse_args()

s = requests.Session(); s.headers.update(H)
s.get(BASE, timeout=20)  # sesi normal, sama seperti membuka halaman di browser
r = s.get(BASE + "home/getdata", timeout=20, headers={"X-Requested-With": "XMLHttpRequest", "Referer": BASE})
print("getdata ->", r.status_code, r.headers.get("content-type"))
r.raise_for_status()
data = r.json()
print("jumlah record:", len(data), "| contoh field:", sorted(data[0].keys()) if data else None)
print("status:", dict(collections.Counter(str(d.get("cctv_status")) for d in data)))

cams, skipped = [], collections.Counter()
for d in data:
    st = str(d.get("cctv_status"))
    link = (d.get("cctv_link") or "").strip()
    if st != "0": skipped[f"status_{st}"] += 1; continue
    if not link.startswith("http"): skipped["tanpa_link"] += 1; continue
    cams.append({"id": str(d.get("cctv_id")), "name": (d.get("cctv_title") or "").strip(),
                 "url": link, "lat": d.get("cctv_latitude"), "lon": d.get("cctv_longitude"), "headers": {}})

if a.check:
    for c in cams:
        try:
            x = s.get(c["url"], timeout=10)
            c["alive"] = x.status_code == 200 and "#EXTM3U" in x.text
        except Exception:
            c["alive"] = False
        time.sleep(0.5)
    print("hidup:", sum(c["alive"] for c in cams), "/", len(cams))

json.dump(cams, open("cameras.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\ncameras.json ditulis: {len(cams)} kamera publik | dilewati: {dict(skipped)}")
for i, c in enumerate(cams[:15], 1):
    print(f"{i:>3}. [{c['id']}] {c['name']}  {c['url']}" + (f"  alive={c['alive']}" if a.check else ""))
if len(cams) > 15: print("   ...")
