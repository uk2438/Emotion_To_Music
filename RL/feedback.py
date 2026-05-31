import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_ROOT = PROJECT_ROOT / "results" / "feedback"
FEEDBACK_LOG_JSONL = FEEDBACK_ROOT / "feedback_log.jsonl"
FEEDBACK_LOG_CSV = FEEDBACK_ROOT / "feedback_log.csv"
REWARD_OVERRIDES_JSON = FEEDBACK_ROOT / "reward_weight_overrides.json"

FEEDBACK_FIELDS = [
    "emotion_match",
    "naturalness",
    "repetition_control",
    "richness",
    "overall",
]


def ensure_feedback_storage():
    FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)


def calculate_feedback_score(feedback):
    """오지선다 피드백을 selection_score에 더할 수 있는 보정 점수로 바꿉니다."""
    values = {
        field: int(feedback.get(field, 3))
        for field in FEEDBACK_FIELDS
    }
    centered = {field: value - 3 for field, value in values.items()}
    return round(
        6.0 * centered["overall"]
        + 4.0 * centered["emotion_match"]
        + 4.0 * centered["naturalness"]
        + 3.0 * centered["repetition_control"]
        + 3.0 * centered["richness"],
        4,
    )


def calculate_feedback_adjusted_selection_score(selection_score, feedback):
    return round(float(selection_score) + calculate_feedback_score(feedback), 4)


def load_reward_weight_overrides():
    ensure_feedback_storage()
    if not REWARD_OVERRIDES_JSON.exists():
        return {}
    with open(REWARD_OVERRIDES_JSON, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def get_reward_weight_overrides_for_mood(mood):
    return load_reward_weight_overrides().get(mood, {})


def _bounded_weight(value):
    return round(min(2.25, max(0.25, float(value))), 4)


def update_reward_weight_overrides(mood, feedback):
    """사용자 평가를 바탕으로 다음 GUI generation에 쓸 reward weight를 조금 조정합니다."""
    ensure_feedback_storage()
    overrides = load_reward_weight_overrides()
    mood_overrides = overrides.setdefault(mood, {})

    repetition_control = int(feedback.get("repetition_control", 3))
    naturalness = int(feedback.get("naturalness", 3))
    emotion_match = int(feedback.get("emotion_match", 3))
    richness = int(feedback.get("richness", 3))

    def bump(component, delta):
        current = mood_overrides.get(component, 1.0)
        mood_overrides[component] = _bounded_weight(current + delta)

    if repetition_control <= 2:
        bump("repetition", 0.08)
        bump("short_loop", 0.06)
        bump("diversity", 0.06)
        bump("pattern", 0.04)
    elif repetition_control >= 4:
        bump("repetition", -0.03)
        bump("short_loop", -0.02)

    if naturalness <= 2:
        bump("interval", 0.05)
        bump("duration_balance", 0.05)
        bump("stable_note", 0.03)
    elif naturalness >= 4:
        bump("interval", -0.02)

    if emotion_match <= 2:
        bump("mode", 0.06)
        bump("range_profile", 0.04)
        bump("velocity", 0.04)
    elif emotion_match >= 4:
        bump("mode", -0.02)

    if richness <= 2:
        bump("harmony", 0.05)
        bump("velocity", 0.03)

    overrides[mood] = mood_overrides
    with open(REWARD_OVERRIDES_JSON, "w", encoding="utf-8") as json_file:
        json.dump(overrides, json_file, ensure_ascii=False, indent=2)

    return mood_overrides


def append_feedback(record):
    ensure_feedback_storage()
    record = {
        **record,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    record["feedback_score"] = calculate_feedback_score(record["feedback"])
    record["feedback_adjusted_selection_score"] = calculate_feedback_adjusted_selection_score(
        record["selection_score"],
        record["feedback"],
    )
    record["updated_reward_overrides"] = update_reward_weight_overrides(
        record["music_mode"],
        record["feedback"],
    )

    with open(FEEDBACK_LOG_JSONL, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    flat_record = _flatten_feedback_record(record)
    write_header = not FEEDBACK_LOG_CSV.exists()
    with open(FEEDBACK_LOG_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(flat_record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat_record)

    return record


def _flatten_feedback_record(record):
    feedback = record.get("feedback", {})
    return {
        "timestamp": record.get("timestamp"),
        "generation_id": record.get("generation_id"),
        "music_mode": record.get("music_mode"),
        "raw_label": record.get("raw_label"),
        "midi_file": record.get("midi_file"),
        "selection_score": record.get("selection_score"),
        "feedback_score": record.get("feedback_score"),
        "feedback_adjusted_selection_score": record.get("feedback_adjusted_selection_score"),
        "emotion_match": feedback.get("emotion_match"),
        "naturalness": feedback.get("naturalness"),
        "repetition_control": feedback.get("repetition_control"),
        "richness": feedback.get("richness"),
        "overall": feedback.get("overall"),
        "comment": record.get("comment", ""),
    }
