#!/usr/bin/env python3
"""TAHAP A: inventaris JS & kandidat endpoint dari halaman publik. Read-only, hanya GET."""
import re, sys, json, urllib.parse as up
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://cctv.jogjakota.go.id/"
UA = {"User-Agent": "Mozilla/5.0 (analysis script)"}
PATTERNS = {
    "hls/dash/mjpeg": r"""[\w:/\.\-\?=&%@]+\.(?:m3u8|mpd|mjpg|mjpeg|flv)[\w\?=&%\-\.]*""",
    "rtsp/ws": r"""\b(?:rtsp|rtmp|wss?)://[^\s'"<>)]+""",
    "api-like": r"""['"`](/?(?:api|ajax|json|data|cctv|camera|stream|live|hls|player)[\w/\.\-\?=&%{}$]*)['"`]""",
    "abs-url": r"""https?://[^\s'"<>)\\]+""",
    "player-lib": r"(?i)\b(hls\.js|Hls\(|video\.js|videojs|flv\.js|mpegts|jwplayer|clappr|dash\.js|shaka|webrtc|RTCPeerConnection|plyr|WebSocket\()",
    "http-call": r"(?i)(fetch\(|\$\.(?:get|ajax|getJSON)\(|axios\.|XMLHttpRequest)",
}

def get(url):
    r = requests.get(url, headers=UA, timeout=20)
    return r

def scan(name, text):
    for label, pat in PATTERNS.items():
        seen = set()
        for m in re.finditer(pat, text):
            s = m.group(0)[:200]
            if s in seen: continue
            seen.add(s)
            line = text.count("\n", 0, m.start()) + 1
            print(f"  [{label}] {name}:{line}: {s}")

r = get(BASE)
print(f"# {BASE} -> {r.status_code} {r.headers.get('content-type')}")
print("# Server headers:", {k: v for k, v in r.headers.items() if k.lower() in ("server", "x-powered-by", "set-cookie", "access-control-allow-origin")})
html = r.text
srcs = re.findall(r"""<script[^>]+src=["']([^"']+)""", html, re.I)
css = re.findall(r"""<link[^>]+href=["']([^"']+\.css[^"']*)""", html, re.I)
print("\n## <script src>:"); [print("  ", up.urljoin(BASE, s)) for s in srcs]
print("\n## <link css>:"); [print("  ", up.urljoin(BASE, s)) for s in css]
print("\n## <video>/<iframe>/<source> tags:")
for t in re.findall(r"<(?:video|iframe|source|embed)[^>]*>", html, re.I): print("  ", t[:300])
print("\n## Scan HTML inline:"); scan("index.html", html)
for s in srcs:
    u = up.urljoin(BASE, s)
    try:
        js = get(u).text
    except Exception as e:
        print(f"\n## {u}: ERROR {e}"); continue
    print(f"\n## Scan {u} ({len(js)} bytes)")
    if len(js) > 400_000: print("  (file besar/vendor library, hanya scan pola spesifik)")
    scan(u.rsplit('/', 1)[-1], js)
