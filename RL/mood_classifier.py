MOOD_LABELS = [
    "happy",
    "sad",
    "angry",
    "neutral",
    "excited",
    "dark",
    "unstable",
]


MOOD_PRESETS = {
    "happy": {
        "happy": 0.78,
        "sad": 0.03,
        "angry": 0.02,
        "neutral": 0.10,
        "excited": 0.05,
        "dark": 0.01,
        "unstable": 0.01,
    },
    "sad": {
        "happy": 0.03,
        "sad": 0.76,
        "angry": 0.02,
        "neutral": 0.08,
        "excited": 0.01,
        "dark": 0.09,
        "unstable": 0.01,
    },
    "angry": {
        "happy": 0.01,
        "sad": 0.05,
        "angry": 0.78,
        "neutral": 0.04,
        "excited": 0.04,
        "dark": 0.03,
        "unstable": 0.05,
    },
    "neutral": {
        "happy": 0.10,
        "sad": 0.08,
        "angry": 0.03,
        "neutral": 0.70,
        "excited": 0.03,
        "dark": 0.03,
        "unstable": 0.03,
    },
    "excited": {
        "happy": 0.25,
        "sad": 0.01,
        "angry": 0.04,
        "neutral": 0.05,
        "excited": 0.62,
        "dark": 0.01,
        "unstable": 0.02,
    },
    "dark": {
        "happy": 0.01,
        "sad": 0.22,
        "angry": 0.06,
        "neutral": 0.04,
        "excited": 0.01,
        "dark": 0.62,
        "unstable": 0.04,
    },
    "unstable": {
        "happy": 0.02,
        "sad": 0.08,
        "angry": 0.12,
        "neutral": 0.03,
        "excited": 0.05,
        "dark": 0.10,
        "unstable": 0.60,
    },
}


def _normalize_scores(scores):
    total = sum(float(scores.get(label, 0.0)) for label in MOOD_LABELS)
    if total <= 0:
        return {label: 1.0 / len(MOOD_LABELS) for label in MOOD_LABELS}

    return {label: float(scores.get(label, 0.0)) / total for label in MOOD_LABELS}


def classify_mood_mock(label="sad"):
    """Return a YOLO-compatible mock mood classifier result.

    The RL pipeline should depend on this output shape, not on a specific
    classifier implementation. The future YOLO module can replace this file
    as long as it returns the same keys.
    """
    if label not in MOOD_PRESETS:
        raise ValueError(f"Unknown mock mood label: {label}")

    scores = _normalize_scores(MOOD_PRESETS[label])
    predicted_label = max(scores, key=scores.get)

    return {
        "source": "mock",
        "label": predicted_label,
        "scores": scores,
    }
