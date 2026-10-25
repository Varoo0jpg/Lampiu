"""Siren audio scoring for the Lampiu traffic simulation.

Audio input can be a local MP4/audio file or an HTTP(S) URL. YouTube URLs are
resolved to their audio stream with yt-dlp. FFmpeg decodes every source to
16 kHz mono PCM. YAMNet scores the AudioSet "Siren" class on short audio windows.

Important: YAMNet's score is a model score, not a calibrated probability. Tune
SIREN_THRESHOLD on representative recordings before using this for experiments.
"""
from __future__ import annotations

import csv
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SAMPLE_RATE = 16_000
WINDOW_SECONDS = 1.5
BYTES_PER_SAMPLE = 2
BYTES_PER_WINDOW = int(SAMPLE_RATE * WINDOW_SECONDS * BYTES_PER_SAMPLE)
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SIREN_THRESHOLD = 0.25
SMOOTHING_ALPHA = 0.45


class YamnetSirenClassifier:
    """Load the pretrained YAMNet model once and score the Siren class."""

    def __init__(self) -> None:
        import numpy as np
        import tensorflow_hub as hub

        self._np = np
        self._model = hub.load(YAMNET_URL)
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")

        with open(class_map_path, "r", encoding="utf-8", newline="") as class_file:
            rows = list(csv.DictReader(class_file))

        self._siren_indices = [
            int(row["index"])
            for row in rows
            if "siren" in row.get("display_name", "").strip().casefold()
        ]
        if not self._siren_indices:
            raise RuntimeError(
                "The loaded YAMNet class map does not contain a class with 'siren' in its name."
            )

    def score(self, waveform: Any) -> float:
        """Return the averaged YAMNet Siren score in the 0.0–1.0 range."""
        np = self._np
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size < int(SAMPLE_RATE * 0.975):
            return 0.0

        # YAMNet expects mono float samples at 16 kHz, generally in [-1, 1].
        samples = np.clip(samples, -1.0, 1.0)
        scores, _embeddings, _spectrogram = self._model(samples)
        scores_np = scores.numpy()
        siren_scores = scores_np[:, self._siren_indices]
        score = float(np.max(np.mean(siren_scores, axis=0)))
        return max(0.0, min(1.0, score))


class _AudioSourceWorker(threading.Thread):
    """Decode one configured input into fixed-size PCM windows."""

    def __init__(self, engine: "SirenAudioEngine", lane: str, source: str) -> None:
        super().__init__(daemon=True, name=f"LampiuAudio-{lane}")
        self.engine = engine
        self.lane = lane
        self.source = source
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()

    def stop(self) -> None:
        with self._process_lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass

    @staticmethod
    def _is_url(source: str) -> bool:
        return urlparse(source).scheme.casefold() in {"http", "https"}

    def _resolve_source(self) -> str:
        source = self.source.strip()
        lowered = source.casefold()
        if "youtube.com/" in lowered or "youtu.be/" in lowered:
            try:
                import yt_dlp
            except ImportError as exc:
                raise RuntimeError(
                    "URL YouTube membutuhkan yt-dlp. Pasang dependensi dari requirements-audio.txt."
                ) from exc

            options = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "format": "bestaudio/best",
            }
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(source, download=False)
            resolved = info.get("url")
            if not resolved:
                raise RuntimeError("yt-dlp tidak menemukan URL audio untuk sumber ini.")
            return str(resolved)

        if self._is_url(source):
            return source

        local_path = Path(source).expanduser()
        if not local_path.is_file():
            raise FileNotFoundError(f"File audio/video tidak ditemukan: {local_path}")
        return str(local_path.resolve())

    @staticmethod
    def _read_exact(stream: Any, size: int) -> bytes | None:
        """Read one full PCM window; return None if the stream ends mid-window."""
        parts: list[bytes] = []
        remaining = size
        while remaining > 0:
            part = stream.read(remaining)
            if not part:
                return None
            parts.append(part)
            remaining -= len(part)
        return b"".join(parts)

    def _ffmpeg_command(self, source: str) -> list[str]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg tidak ditemukan. Instal FFmpeg dan pastikan perintah ffmpeg tersedia di PATH."
            )

        local_source = not self._is_url(self.source.strip())
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if local_source:
            command += ["-stream_loop", "-1"]
        else:
            command += [
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
            ]

        command += [
            "-i", source,
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]
        return command

    def run(self) -> None:
        if not self.engine._is_url(self.source.strip()) and not Path(
            self.source.strip()
        ).expanduser().is_file():
            self.engine._emit_status(
                self.lane,
                f"File tidak ditemukan: {self.source}",
            )
            return

        if not shutil.which("ffmpeg"):
            self.engine._emit_status(
                self.lane,
                "FFmpeg tidak ditemukan. Instal FFmpeg dan tambahkan ke PATH.",
            )
            return

        while not self.engine.stopping:
            process = None
            try:
                resolved_source = self._resolve_source()
                command = self._ffmpeg_command(resolved_source)
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
                with self._process_lock:
                    self._process = process

                self.engine._emit_status(self.lane, "Sumber audio tersambung")
                if process.stdout is None:
                    raise RuntimeError("FFmpeg tidak menyediakan aliran audio.")

                while not self.engine.stopping:
                    data = self._read_exact(process.stdout, BYTES_PER_WINDOW)
                    if data is None:
                        break
                    self.engine._put_audio_window(self.lane, data)

                if not self.engine.stopping:
                    self.engine._emit_status(
                        self.lane,
                        "Aliran audio berakhir atau tidak memiliki audio. Mencoba menyambung ulang.",
                    )
            except Exception as exc:
                self.engine._emit_status(self.lane, f"Sumber audio bermasalah: {exc}")
            finally:
                if process is not None:
                    try:
                        if process.poll() is None:
                            process.terminate()
                        process.wait(timeout=1.0)
                    except (OSError, subprocess.TimeoutExpired):
                        try:
                            process.kill()
                        except OSError:
                            pass
                    with self._process_lock:
                        self._process = None

            if not self.engine.stopping:
                self.engine.stop_event.wait(3.0)


class SirenAudioEngine:
    """Read four audio sources, score them with YAMNet, and expose UI events."""

    def __init__(
        self,
        sources: dict[str, str | None],
        threshold: float = SIREN_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold harus berada di antara 0.0 dan 1.0")

        self.sources = {lane: source for lane, source in sources.items() if source}
        self.threshold = threshold
        self.stop_event = threading.Event()
        self.stopping = False
        self._audio_queue: queue.Queue[tuple[str, bytes]] = queue.Queue(maxsize=8)
        self._ui_events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=128)
        self._source_workers: list[_AudioSourceWorker] = []
        self._inference_thread: threading.Thread | None = None
        self._running = False
        self._smoothed: dict[str, float] = {}
        self._recent_positive: dict[str, list[bool]] = {}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if not self.sources:
            self._emit_status(None, "Belum ada sumber audio yang dikonfigurasi.")
            return

        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="LampiuSirenClassifier",
        )
        self._inference_thread.start()

    def stop(self) -> None:
        self.stopping = True
        self.stop_event.set()
        for worker in self._source_workers:
            worker.stop()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=1.5)
        for worker in self._source_workers:
            if worker.is_alive():
                worker.join(timeout=0.4)

    def drain_events(self, limit: int = 40) -> list[dict[str, Any]]:
        events = []
        for _ in range(max(1, limit)):
            try:
                events.append(self._ui_events.get_nowait())
            except queue.Empty:
                break
        return events

    def _emit_status(self, lane: str | None, message: str) -> None:
        self._emit_ui({"type": "status", "lane": lane, "message": message})

    def _emit_ui(self, event: dict[str, Any]) -> None:
        try:
            self._ui_events.put_nowait(event)
        except queue.Full:
            try:
                self._ui_events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._ui_events.put_nowait(event)
            except queue.Full:
                pass

    def _put_audio_window(self, lane: str, data: bytes) -> None:
        try:
            self._audio_queue.put_nowait((lane, data))
        except queue.Full:
            # Keep recent audio rather than building a long analysis delay.
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait((lane, data))
            except queue.Full:
                pass

    def _inference_loop(self) -> None:
        import numpy as np

        self._emit_status(None, "Memuat model audio YAMNet...")
        try:
            classifier = YamnetSirenClassifier()
        except Exception as exc:
            self._emit_status(
                None,
                f"Gagal memuat YAMNet: {exc}. Periksa instalasi audio dan koneksi internet.",
            )
            self._running = False
            return

        self._source_workers = [
            _AudioSourceWorker(self, lane, source)
            for lane, source in self.sources.items()
        ]
        self._emit_status(None, "Model YAMNet siap. Menghubungkan sumber audio...")
        for worker in self._source_workers:
            worker.start()

        try:
            while not self.stopping:
                try:
                    lane, raw_pcm = self._audio_queue.get(timeout=0.25)
                except queue.Empty:
                    continue

                waveform = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32)
                waveform /= 32768.0
                try:
                    raw_score = classifier.score(waveform)
                except Exception as exc:
                    self._emit_status(lane, f"Analisis audio gagal: {exc}")
                    continue

                previous = self._smoothed.get(lane, raw_score)
                smoothed = (
                    (1.0 - SMOOTHING_ALPHA) * previous
                    + SMOOTHING_ALPHA * raw_score
                )
                self._smoothed[lane] = max(0.0, min(1.0, smoothed))

                recent = self._recent_positive.setdefault(lane, [])
                recent.append(self._smoothed[lane] >= self.threshold)
                del recent[:-3]
                detected = sum(recent) >= 2

                self._emit_ui(
                    {
                        "type": "confidence",
                        "lane": lane,
                        "confidence": self._smoothed[lane],
                        "raw_score": raw_score,
                        "detected": detected,
                        "threshold": self.threshold,
                    }
                )
        finally:
            for worker in self._source_workers:
                worker.stop()
