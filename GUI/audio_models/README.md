# Audio classifier artifact — V2

## Runtime contract

The current GUI loads this file through `GUI/audio_siren.py`:

- Artifact: `best_audio_classifier_v2.joblib`
- Feature extractor: pretrained Google YAMNet (`https://tfhub.dev/google/yamnet/1`)
- Input to classifier: mean YAMNet embedding, 1,024 features per analysis window
- Class labels: `ambulance`, `firetruck`, `police`, `traffic`
- Inference output: per-class `predict_proba` scores; `emergency_score` is the sum of ambulance, firetruck, and police scores
- Audio waveform supplied by the GUI: mono, 16 kHz, in 3-second windows

## Training provenance

The project training discussion recorded Logistic Regression with `C=100` as the selected configuration after comparing candidate C values `0.1, 1, 10, 100` and an RBF SVM candidate set. The selected Logistic Regression configuration was reported with validation Macro F1 of 92.17%. The inference module does not retrain the classifier.

**Verification boundary:** the current repository contains the serialized estimator but not the complete audio-training notebook/script or a machine-readable training manifest. The inference module checks that the estimator exposes `predict_proba` and the expected four class labels; it does not independently assert the stored estimator's `C` parameter. Therefore, `C=100` is documented as the recorded training configuration, not as a value re-established by the inference code.

If the classifier is retrained, update this document with the new dataset version, split strategy, random seed, selected estimator and hyperparameters, class order, validation metrics, test metrics, and artifact checksum. Do not replace the artifact without updating its provenance.
