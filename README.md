# Lampiu

Lampiu is a research prototype for four-approach traffic-signal simulation using vehicle detection, PCU-weighted green-time allocation, and dual-modal emergency-vehicle indication.

**Current GUI entry point:** `GUI/smart_traffic_jogja.py`  
**Core implementation:** `GUI/smart_traffic_gui_100.py`

The application is for controlled software simulation and research. It is not connected to physical traffic-signal controllers and is not approved for deployment on public roads.

## Quick start

Use Python 3.10 or 3.11. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r GUI\requirements.txt
python GUI\smart_traffic_jogja.py
```

The launcher lets each approach (Utara, Timur, Selatan, Barat) use a CCTV Jogja stream, webcam, local video, built-in source, or no source. For the current emergency-audio scenario, select one local video containing an audio track and assign it to the intended approach.

## System overview

- **Vision:** YOLO26n 100-epoch checkpoint, eight classes, default confidence 0.15, NMS IoU 0.60, 640 inference size, five-frame median count smoothing.
- **Vehicle weighting:** simulator weights are Ambulance 1.00, Bus 2.00, Car 1.00, Fire engine 2.00, Military 2.00, Motorcycle 0.33, Police car 1.00, Truck 2.00. These are study-defined operational weights, not universally validated PCU standards.
- **Green allocation:** 120 seconds of green time distributed across four approaches in proportion to weighted counts, bounded to 10–40 seconds per approach. Yellow (3 s) and all-red (1 s) are additional transition-clearance times; one normal four-approach rotation therefore takes 136 seconds (120 s green allocation + 16 s clearance).
- **Audio:** local video soundtrack → FFmpeg decode → 3-second mono 16 kHz segments → YAMNet 1,024-dimensional embeddings → trained Logistic Regression classifier.
- **Emergency preemption:** visual confidence for Ambulance/Fire engine/Police car and aggregate audio emergency score must both exceed 0.65 on the same approach. These are initial thresholds, not field-calibrated values.

## Documentation

- [Main GUI documentation](GUI/README.md)
- [Audio model artifact and training provenance](GUI/audio_models/README.md)
- [Legacy Coba audio prototype notes](AUDIO_SETUP.md)

Chapter 4 / overall system experiment results are intentionally not included in this repository audit update. No unmeasured performance or effectiveness claims are made.
