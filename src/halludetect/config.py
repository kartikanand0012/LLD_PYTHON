LABELS = {
    "non-hallucination": 0,
    "hallucination": 1,
}

LABEL_NAMES = {value: key for key, value in LABELS.items()}

DEFAULT_MODEL_PATH = "models/halludetect.joblib"
