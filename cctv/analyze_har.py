#!/usr/bin/env python3
"""TAHAP B: klasifikasi request dari HAR (DevTools > Network > Save all as HAR).
Cookie/Authorization TIDAK dicetak, hanya ditandai ada/tidak."""
import json, sys, collections, urllib.parse as up

har = json.load(open(sys.argv[1], encoding="utf-8"))
entries = har["log"]["entries"]

def kind(url, mime):
    p = up.urlparse(url).path.lower(); m = (mime or "").lower()
    if p.endswith(".m3u8") or "mpegurl" in m: return "HLS-MANIFEST"
    if p.endswith(".mpd") or "dash+xml" in m: return "DASH-MANIFEST"
    if "multipart/x-mixed-replace" in m or p.endswith((".mjpg", ".mjpeg")): return "MJPEG"
    if p.endswith((".ts", ".m4s", ".cmfv", ".cmfa")) or "mp2t" in m: return "SEGMENT"
    if p.endswith((".mp4", ".webm", ".flv")) or m.startswith(("video/", "audio/")): return "MEDIA"
    if "json" in m: return "JSON"
    if url.startswith(("ws://", "wss://")): return "WEBSOCKET"
    if m.startswith("image/"): return "IMAGE"
    return "OTHER"

rows = []
for e in entries:
    rq, rs = e["request"], e["response"]
    mime = rs["content"].get("mimeType", "")
    hd = {h["name"].lower(): h["value"] for h in rq["headers"]}
    rows.append(dict(
        k=kind(rq["url"], mime), method=rq["method"], status=rs["status"], mime=mime,
        url=rq["url"], referer=hd.get("referer"), origin=hd.get("origin"),
        has_cookie="cookie" in hd, has_auth="authorization" in hd,
        initiator=e.get("_initiator", {}).get("type"), size=rs["content"].get("size"),
        body=rs["content"].get("text"),
    ))

print("== Ringkasan tipe ==", dict(collections.Counter(r["k"] for r in rows)))
for want in ("HLS-MANIFEST", "DASH-MANIFEST", "MJPEG", "WEBSOCKET", "MEDIA", "JSON"):
    sel = [r for r in rows if r["k"] == want]
    if not sel: continue
    print(f"\n== {want} ({len(sel)}) ==")
    for r in sel[:40]:
        print(f"- {r['method']} {r['status']} {r['mime']}\n  {r['url']}\n  referer={r['referer']} origin={r['origin']} cookie={r['has_cookie']} auth={r['has_auth']}")
        if want == "JSON" and r["body"]: print("  body[:300]:", r["body"][:300].replace("\n", " "))
seg = [r for r in rows if r["k"] == "SEGMENT"]
if seg: print(f"\n== SEGMENT: {len(seg)} request, contoh: {seg[0]['url']}")
