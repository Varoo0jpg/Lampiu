"""YAMNet + Logistic Regression V2 inference for Lampiu."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import joblib
import librosa
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "audio_models" / "best_audio_classifier_v2.joblib"
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
EXPECTED_CLASSES = {"ambulance", "firetruck", "police", "traffic"}
EMERGENCY_CLASSES = {"ambulance", "firetruck", "police"}
SAMPLE_RATE = 16000

_lock = threading.Lock()
_yamnet = None
_classifier = None


def _load_models():
    global _yamnet, _classifier
    if _yamnet is not None and _classifier is not None:
        return _yamnet, _classifier
    with _lock:
        if _yamnet is not None and _classifier is not None:
            return _yamnet, _classifier
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Classifier tidak ditemukan: {MODEL_PATH}")
        import tensorflow_hub as hub
        classifier = joblib.load(MODEL_PATH)
        if not hasattr(classifier, "predict_proba") or not hasattr(classifier, "classes_"):
            raise TypeError("Classifier harus menyediakan predict_proba() dan classes_.")
        classes = {str(v).strip().lower() for v in classifier.classes_}
        if classes != EXPECTED_CLASSES:
            raise ValueError(f"Kelas classifier tidak sesuai: {sorted(classes)}")
        _classifier = classifier
        _yamnet = hub.load(YAMNET_HANDLE)
    return _yamnet, _classifier


def classify_audio_waveform(waveform, sample_rate=SAMPLE_RATE) -> dict:
    """Classify one mono waveform; training-compatible input is mono 16 kHz."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if waveform.ndim != 1 or waveform.size < int(0.5 * SAMPLE_RATE):
        raise ValueError("Audio harus mono dan berdurasi minimal 0,5 detik.")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("Audio mengandung nilai NaN/Inf.")
    yamnet, classifier = _load_models()
    _, embeddings, _ = yamnet(waveform)
    embedding = np.asarray(embeddings.numpy(), dtype=np.float32).mean(axis=0, keepdims=True)
    if embedding.shape != (1, 1024) or not np.all(np.isfinite(embedding)):
        raise RuntimeError(f"Embedding YAMNet tidak valid: {embedding.shape}")
    values = classifier.predict_proba(embedding)[0]
    names = [str(v).strip().lower() for v in classifier.classes_]
    probs = {name: float(value) for name, value in zip(names, values)}
    total = sum(probs.values())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Classifier menghasilkan probabilitas tidak valid.")
    probs = {name: value / total for name, value in probs.items()}
    predicted = max(probs, key=probs.get)
    return {
        "predicted_class": predicted,
        "probabilities": probs,
        "emergency_score": float(np.clip(sum(probs[k] for k in EMERGENCY_CLASSES), 0.0, 1.0)),
        "duration_seconds": float(waveform.size / SAMPLE_RATE),
        "sample_rate": SAMPLE_RATE,
    }


def classify_audio_file(audio_path: str | os.PathLike) -> dict:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"File audio tidak ditemukan: {path}")
    waveform, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return classify_audio_waveform(waveform, SAMPLE_RATE)
