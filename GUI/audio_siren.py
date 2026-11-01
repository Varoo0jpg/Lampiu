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
    """Load pretrained YAMNet and the saved classifier once."""
    global _yamnet, _classifier
    if _yamnet is not None and _classifier is not None:
        return _yamnet, _classifier

    with _lock:
        if _yamnet is not None and _classifier is not None:
            return _yamnet, _classifier
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Classifier tidak ditemukan: {MODEL_PATH}. "
                "Pastikan folder GUI/audio_models ikut diunduh."
            )

        import tensorflow_hub as hub

        classifier = joblib.load(MODEL_PATH)
        if not hasattr(classifier, "predict_proba") or not hasattr(classifier, "classes_"):
            raise TypeError("Classifier harus menyediakan predict_proba() dan classes_.")

        classes = {str(value).strip().lower() for value in classifier.classes_}
        if classes != EXPECTED_CLASSES:
            raise ValueError(
                "Kelas classifier tidak sesuai. "
                f"Diharapkan {sorted(EXPECTED_CLASSES)}, ditemukan {sorted(classes)}."
            )

        yamnet = hub.load(YAMNET_HANDLE)
        _classifier = classifier
        _yamnet = yamnet

    return _yamnet, _classifier


def classify_audio_file(audio_path: str | os.PathLike) -> dict:
    """Return predicted class, four class probabilities, and emergency score."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"File audio tidak ditemukan: {path}")

    waveform, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Audio kosong atau format waveform tidak valid.")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("Audio mengandung nilai NaN/Inf.")
    if waveform.size < int(0.5 * SAMPLE_RATE):
        raise ValueError("Durasi audio terlalu pendek; minimal 0,5 detik.")

    yamnet, classifier = _load_models()
    _, embeddings, _ = yamnet(waveform)
    embedding = np.asarray(embeddings.numpy(), dtype=np.float32).mean(axis=0, keepdims=True)

    if embedding.shape != (1, 1024) or not np.all(np.isfinite(embedding)):
        raise RuntimeError(f"Embedding YAMNet tidak valid: {embedding.shape}")

    raw_probabilities = classifier.predict_proba(embedding)[0]
    class_names = [str(value).strip().lower() for value in classifier.classes_]
    probability_map = {
        name: float(value)
        for name, value in zip(class_names, raw_probabilities)
    }
    total = sum(probability_map.values())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Classifier menghasilkan probabilitas tidak valid.")

    probability_map = {name: value / total for name, value in probability_map.items()}
    predicted_class = max(probability_map, key=probability_map.get)
    emergency_score = sum(probability_map[name] for name in EMERGENCY_CLASSES)

    return {
        "predicted_class": predicted_class,
        "probabilities": probability_map,
        "emergency_score": float(np.clip(emergency_score, 0.0, 1.0)),
        "duration_seconds": float(waveform.size / SAMPLE_RATE),
        "sample_rate": SAMPLE_RATE,
    }
