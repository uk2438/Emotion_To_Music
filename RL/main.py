import argparse
from collections import Counter
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from dqn_agent import DQNAgent, FactorizedDQNAgent
from melody_env import MODE_PROFILES, MelodyEnv
from mood_classifier import classify_mood_mock
from q_learning_agent import QLearningAgent
from team_emotion_classifier import classify_mood_from_image

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
TEAM_CLASSIFIER_DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "Emotion_Classification" / "checkpoints" / "best.pt"
)
TEAM_CLASSIFIER_DEFAULT_IMAGE = PROJECT_ROOT / "Emotion_Classification" / "test.jpg"

NOTE_NAME_MAP = {
    57: "A3",
    59: "B3",
    60: "C4",
    61: "C#4",
    62: "D4",
    64: "E4",
    65: "F4",
    66: "F#4",
    67: "G4",
    69: "A4",
    70: "A#4/Bb4",
    71: "B4",
    72: "C5",
    74: "D5",
    76: "E5",
}

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def map_emotion_to_music_mode(emotion):
    """DeepFace 감정 label을 RL 음악 mode로 변환합니다."""
    emotion_map = {
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "neutral": "neutral",
        "surprise": "excited",
        "fear": "dark",
        "disgust": "unstable",
    }

    return emotion_map.get(emotion, "neutral")



def load_image_safely(image_path):
    """한글이 포함된 경로에서도 이미지를 안전하게 읽습니다.

    DeepFace는 문자열 경로에 non-english 문자가 있으면 에러를 낸다고합니다.
    그래서 cv2.imdecode와 np.fromfile을 이용해 이미지를 numpy array로 읽습니다.
    """
    import cv2

    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    return image


def analyze_face_emotion(image_path):
    """얼굴 사진을 분석해서 dominant emotion, music mode, 전체 score를 반환합니다."""
    from deepface import DeepFace

    image = load_image_safely(image_path)

    result = DeepFace.analyze(
        img_path=image,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
    )

    if isinstance(result, list):
        result = result[0]

    detected_emotion = result["dominant_emotion"]
    music_mode = map_emotion_to_music_mode(detected_emotion)
    emotion_scores = result["emotion"]

    return detected_emotion, music_mode, emotion_scores


def analyze_mood(source="teammate", mock_mood="sad", image_path=None, classifier_checkpoint=None):
    """Return a classifier-like result for the RL pipeline."""
    if source == "mock":
        result = classify_mood_mock(mock_mood)
        return result["label"], result["label"], result["scores"], result

    if source == "deepface":
        if image_path is None:
            raise ValueError("image_path is required when source='deepface'")

        detected_emotion, mode, emotion_scores = analyze_face_emotion(image_path)
        classifier_result = {
            "source": "deepface",
            "label": mode,
            "raw_label": detected_emotion,
            "scores": emotion_scores,
        }
        return detected_emotion, mode, emotion_scores, classifier_result

    if source == "teammate":
        if image_path is None:
            raise ValueError("image_path is required when source='teammate'")
        if classifier_checkpoint is None:
            classifier_checkpoint = TEAM_CLASSIFIER_DEFAULT_CHECKPOINT

        result = classify_mood_from_image(
            image_path=image_path,
            checkpoint_path=classifier_checkpoint,
        )
        return result["raw_label"], result["label"], result["scores"], result

    raise ValueError(f"Unknown mood source: {source}")


def train_agent(
    mode="happy",
    episodes=5000,
    melody_length=32,
    state_mode="table",
    mood_vector=None,
    octave_expansion=False,
    expansion_start_ratio=0.5,
    max_pitch_jump=12,
    action_masking=True,
    reward_weight_overrides=None,
):
    """지정한 감정 mode에 대해 Q-learning agent를 학습합니다."""
    if state_mode != "table":
        raise ValueError("QLearningAgent requires state_mode='table'. Use vector state with DQNAgent.")

    env = MelodyEnv(
        mode=mode,
        melody_length=melody_length,
        state_mode=state_mode,
        mood_vector=mood_vector,
        octave_expansion=octave_expansion,
        expansion_start_ratio=expansion_start_ratio,
        max_pitch_jump=max_pitch_jump,
        action_masking=action_masking,
        reward_weight_overrides=reward_weight_overrides,
    )
    agent = QLearningAgent(action_size=env.action_size)

    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, training=True, valid_actions=valid_actions)
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()

            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
                valid_next_actions=next_valid_actions,
            )

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 1000 == 0:
            recent_avg = sum(episode_rewards[-1000:]) / 1000
            print(
                f"Episode {episode + 1:5d} | "
                f"Recent Avg Reward: {recent_avg:6.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return env, agent, episode_rewards


def train_dqn_agent(
    mode="happy",
    episodes=5000,
    melody_length=32,
    mood_vector=None,
    batch_size=32,
    target_update_interval=100,
    hidden_size=128,
    learning_rate=0.0005,
    epsilon_decay=0.999,
    epsilon_min=0.05,
    replay_capacity=20000,
    use_double_dqn=True,
    octave_expansion=False,
    expansion_start_ratio=0.5,
    max_pitch_jump=12,
    action_masking=True,
    reward_weight_overrides=None,
):
    """지정한 감정 mode에 대해 DQN agent를 학습합니다."""
    env = MelodyEnv(
        mode=mode,
        melody_length=melody_length,
        state_mode="vector",
        mood_vector=mood_vector,
        octave_expansion=octave_expansion,
        expansion_start_ratio=expansion_start_ratio,
        max_pitch_jump=max_pitch_jump,
        action_masking=action_masking,
        reward_weight_overrides=reward_weight_overrides,
    )
    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.action_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        replay_capacity=replay_capacity,
        use_double_dqn=use_double_dqn,
    )

    episode_rewards = []
    episode_losses = []
    best_training_reward = None
    best_selection_score = None
    best_checkpoint_metrics = None
    best_model_state = None

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, training=True, valid_actions=valid_actions)
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
                next_valid_actions=next_valid_actions,
            )
            loss = agent.replay(batch_size)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_losses.append(sum(losses) / len(losses) if losses else None)

        selection_score, selection_metrics = calculate_checkpoint_selection_score(
            total_reward=total_reward,
            info=info,
            mode=mode,
            base_notes=env.base_notes,
        )
        if best_selection_score is None or selection_score > best_selection_score:
            best_training_reward = total_reward
            best_selection_score = selection_score
            best_checkpoint_metrics = selection_metrics
            best_model_state = agent.get_model_state()

        if (episode + 1) % target_update_interval == 0:
            agent.update_target_network()

        if (episode + 1) % 1000 == 0:
            recent_avg = sum(episode_rewards[-1000:]) / 1000
            recent_losses = [loss for loss in episode_losses[-1000:] if loss is not None]
            recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            print(
                f"Episode {episode + 1:5d} | "
                f"Recent Avg Reward: {recent_avg:6.2f} | "
                f"Recent Loss: {recent_loss:8.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.update_target_network()
    if best_model_state is not None:
        agent.load_model_state(best_model_state)
    training_metrics = {
        "episode_losses": episode_losses,
        "batch_size": batch_size,
        "target_update_interval": target_update_interval,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "replay_capacity": replay_capacity,
        "use_double_dqn": use_double_dqn,
        "best_training_reward": best_training_reward,
        "best_selection_score": best_selection_score,
        "best_checkpoint_metrics": best_checkpoint_metrics,
        "restored_best_checkpoint": best_model_state is not None,
        "octave_expansion": octave_expansion,
        "expansion_start_ratio": expansion_start_ratio,
        "max_pitch_jump": max_pitch_jump,
        "action_masking": action_masking,
    }

    return env, agent, episode_rewards, training_metrics


def train_factorized_dqn_agent(
    mode="happy",
    episodes=5000,
    melody_length=32,
    mood_vector=None,
    batch_size=32,
    target_update_interval=100,
    hidden_size=128,
    learning_rate=0.0005,
    epsilon_decay=0.999,
    epsilon_min=0.05,
    replay_capacity=20000,
    use_double_dqn=True,
    octave_expansion=False,
    expansion_start_ratio=0.5,
    max_pitch_jump=12,
    action_masking=True,
    reward_weight_overrides=None,
):
    """pitch, duration, velocity를 분리한 factorized DQN agent를 학습합니다."""
    env = MelodyEnv(
        mode=mode,
        melody_length=melody_length,
        state_mode="vector",
        mood_vector=mood_vector,
        octave_expansion=octave_expansion,
        expansion_start_ratio=expansion_start_ratio,
        max_pitch_jump=max_pitch_jump,
        action_masking=action_masking,
        reward_weight_overrides=reward_weight_overrides,
    )
    agent = FactorizedDQNAgent(
        state_size=env.get_state_size(),
        pitch_action_size=env.pitch_action_size,
        duration_action_size=env.duration_action_size,
        velocity_action_size=env.velocity_action_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        replay_capacity=replay_capacity,
        use_double_dqn=use_double_dqn,
    )

    episode_rewards = []
    episode_losses = []
    best_training_reward = None
    best_selection_score = None
    best_checkpoint_metrics = None
    best_model_state = None

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, training=True, valid_actions=valid_actions)
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
                next_valid_actions=next_valid_actions,
            )
            loss = agent.replay(batch_size)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_losses.append(sum(losses) / len(losses) if losses else None)

        selection_score, selection_metrics = calculate_checkpoint_selection_score(
            total_reward=total_reward,
            info=info,
            mode=mode,
            base_notes=env.base_notes,
        )
        if best_selection_score is None or selection_score > best_selection_score:
            best_training_reward = total_reward
            best_selection_score = selection_score
            best_checkpoint_metrics = selection_metrics
            best_model_state = agent.get_model_state()

        if (episode + 1) % target_update_interval == 0:
            agent.update_target_network()

        if (episode + 1) % 1000 == 0:
            recent_avg = sum(episode_rewards[-1000:]) / 1000
            recent_losses = [loss for loss in episode_losses[-1000:] if loss is not None]
            recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            print(
                f"Episode {episode + 1:5d} | "
                f"Recent Avg Reward: {recent_avg:6.2f} | "
                f"Recent Loss: {recent_loss:8.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.update_target_network()
    if best_model_state is not None:
        agent.load_model_state(best_model_state)
    training_metrics = {
        "episode_losses": episode_losses,
        "batch_size": batch_size,
        "target_update_interval": target_update_interval,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "replay_capacity": replay_capacity,
        "use_double_dqn": use_double_dqn,
        "best_training_reward": best_training_reward,
        "best_selection_score": best_selection_score,
        "best_checkpoint_metrics": best_checkpoint_metrics,
        "restored_best_checkpoint": best_model_state is not None,
        "octave_expansion": octave_expansion,
        "expansion_start_ratio": expansion_start_ratio,
        "max_pitch_jump": max_pitch_jump,
        "action_masking": action_masking,
    }

    return env, agent, episode_rewards, training_metrics


def generate_melody(env, agent):
    """학습된 agent를 이용해 탐험 없이 greedy 방식으로 멜로디를 생성합니다."""
    state = env.reset()
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, training=False, valid_actions=valid_actions)
        next_state, reward, done, info = env.step(action)
        state = next_state

    return (
        info["actions"],
        info["melody"],
        info.get("durations", [0.5] * len(info["melody"])),
        info.get("velocities", [100] * len(info["melody"])),
        info.get("pitch_actions", info["actions"]),
        info.get("duration_actions", []),
        info.get("velocity_actions", []),
        info.get("events", []),
        info.get("reward_breakdowns", []),
    )


def convert_to_note_names(melody):
    """MIDI pitch 숫자를 사람이 읽기 쉬운 음 이름으로 변환합니다."""
    note_names = []
    for note in melody:
        if note in NOTE_NAME_MAP:
            note_names.append(NOTE_NAME_MAP[note])
        else:
            octave = int(note) // 12 - 1
            note_names.append(f"{PITCH_CLASS_NAMES[int(note) % 12]}{octave}")
    return note_names


def export_melody_to_midi(
    melody,
    filename="generated_melody.mid",
    durations=None,
    velocities=None,
    note_duration=0.5,
    tempo=120,
    instrument_name="Acoustic Grand Piano",
    chord_progression=None,
    add_accompaniment=True,
):
    """생성된 MIDI pitch 리스트를 실제로 들을 수 있는 .mid 파일로 저장합니다."""
    if pretty_midi is None:
        raise ImportError(
            "pretty_midi가 설치되어 있지 않습니다. 터미널에서 `pip install pretty_midi`를 실행하세요."
        )

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    start_time = 0.0

    if durations is None:
        durations = [note_duration] * len(melody)
    if velocities is None:
        velocities = [100] * len(melody)

    for pitch, duration, velocity in zip(melody, durations, velocities):
        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=start_time,
            end=start_time + float(duration),
        )
        piano.notes.append(note)
        start_time += float(duration)

    midi.instruments.append(piano)

    if add_accompaniment and chord_progression:
        _add_accompaniment_tracks(
            midi=midi,
            chord_progression=chord_progression,
            total_duration=start_time,
            tempo=tempo,
        )

    midi.write(str(filename))


def _add_accompaniment_tracks(midi, chord_progression, total_duration, tempo):
    """Chord pad와 bass track을 추가해 단선율 MIDI를 더 풍성하게 만듭니다."""
    chord_track = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Pad 2 (warm)")
    )
    bass_track = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Bass")
    )

    chord_duration = max(1.0, total_duration / max(1, len(chord_progression)))
    current_time = 0.0

    while current_time < total_duration - 1e-6:
        chord_index = int(current_time // chord_duration) % len(chord_progression)
        chord_name = chord_progression[chord_index]
        chord_pitches = _chord_name_to_midi_pitches(chord_name, octave=4)
        bass_pitch = _chord_root_to_midi_pitch(chord_name, octave=2)
        end_time = min(total_duration, current_time + chord_duration)

        for pitch in chord_pitches:
            chord_track.notes.append(
                pretty_midi.Note(
                    velocity=54,
                    pitch=pitch,
                    start=current_time,
                    end=end_time,
                )
            )

        bass_track.notes.append(
            pretty_midi.Note(
                velocity=70,
                pitch=bass_pitch,
                start=current_time,
                end=end_time,
            )
        )

        current_time = end_time

    midi.instruments.append(chord_track)
    midi.instruments.append(bass_track)


def _chord_name_to_midi_pitches(chord_name, octave=4):
    root, quality = _parse_chord_name(chord_name)
    intervals = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "dim": [0, 3, 6],
    }[quality]
    root_pitch = _root_name_to_midi_pitch(root, octave)
    return [root_pitch + interval for interval in intervals]


def _chord_root_to_midi_pitch(chord_name, octave=2):
    root, _quality = _parse_chord_name(chord_name)
    return _root_name_to_midi_pitch(root, octave)


def _parse_chord_name(chord_name):
    if chord_name.endswith("dim"):
        return chord_name[:-3], "dim"
    if chord_name.endswith("m"):
        return chord_name[:-1], "minor"
    return chord_name, "major"


def _root_name_to_midi_pitch(root_name, octave):
    root_pitch_classes = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    if root_name not in root_pitch_classes:
        raise ValueError(f"Unknown chord root: {root_name}")
    return 12 * (octave + 1) + root_pitch_classes[root_name]


def set_seed(seed):
    """Make baseline runs easier to compare across future experiments."""
    random.seed(seed)
    np.random.seed(seed)


def save_reward_csv(rewards, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "reward"])
        for episode, reward in enumerate(rewards, start=1):
            writer.writerow([episode, reward])


def recent_average(values, window=100):
    if not values:
        return 0.0
    recent_values = values[-window:]
    return sum(recent_values) / len(recent_values)


def save_loss_csv(losses, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "loss"])
        for episode, loss in enumerate(losses, start=1):
            writer.writerow([episode, "" if loss is None else loss])


def save_reward_breakdown_csv(reward_breakdowns, filename):
    component_names = sorted(
        {component for breakdown in reward_breakdowns for component in breakdown}
    )
    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", *component_names, "total"])
        for step, breakdown in enumerate(reward_breakdowns, start=1):
            row = [step]
            row.extend(breakdown.get(component, 0.0) for component in component_names)
            row.append(sum(breakdown.values()))
            writer.writerow(row)


def summarize_reward_breakdowns(reward_breakdowns):
    if not reward_breakdowns:
        return {
            "component_totals": {},
            "component_averages": {},
            "total_reward": 0.0,
        }

    component_names = sorted(
        {component for breakdown in reward_breakdowns for component in breakdown}
    )
    totals = {
        component: sum(breakdown.get(component, 0.0) for breakdown in reward_breakdowns)
        for component in component_names
    }
    return {
        "component_totals": totals,
        "component_averages": {
            component: total / len(reward_breakdowns)
            for component, total in totals.items()
        },
        "total_reward": sum(totals.values()),
    }


def _count_direction_changes(melody):
    directions = []
    for index in range(1, len(melody)):
        interval = melody[index] - melody[index - 1]
        if interval > 0:
            directions.append(1)
        elif interval < 0:
            directions.append(-1)
        else:
            directions.append(0)

    return sum(
        1 for index in range(1, len(directions))
        if directions[index] != 0
        and directions[index - 1] != 0
        and directions[index] != directions[index - 1]
    )


def _phrase_similarity(first_phrase, second_phrase):
    if not first_phrase or not second_phrase:
        return 0.0
    compare_length = min(len(first_phrase), len(second_phrase))
    matches = sum(
        1 for index in range(compare_length)
        if first_phrase[index] == second_phrase[index]
    )
    return matches / compare_length


def _mode_contour_alignment(mode, melody):
    if len(melody) < 2:
        return 0.0

    upward = sum(1 for index in range(1, len(melody)) if melody[index] > melody[index - 1])
    downward = sum(1 for index in range(1, len(melody)) if melody[index] < melody[index - 1])
    moving = upward + downward
    if moving == 0:
        return 0.0

    if mode in ["happy", "excited"]:
        return upward / moving
    if mode in ["sad", "dark"]:
        return downward / moving
    if mode == "angry":
        large_moves = sum(
            1 for index in range(1, len(melody))
            if abs(melody[index] - melody[index - 1]) >= 5
        )
        return large_moves / max(1, len(melody) - 1)
    if mode == "neutral":
        small_moves = sum(
            1 for index in range(1, len(melody))
            if 0 < abs(melody[index] - melody[index - 1]) <= 4
        )
        return small_moves / max(1, len(melody) - 1)
    if mode == "unstable":
        chromatic_moves = sum(
            1 for index in range(1, len(melody))
            if abs(melody[index] - melody[index - 1]) in [1, 6]
        )
        return chromatic_moves / max(1, len(melody) - 1)
    return 0.0


def calculate_melody_metrics(actions, melody, durations=None, velocities=None, pitch_actions=None, mode=None, base_notes=None):
    if not actions:
        return {
            "length": 0,
            "unique_actions": 0,
            "unique_action_ratio": 0.0,
            "unique_pitch_actions": 0,
            "unique_duration_values": 0,
            "same_adjacent_count": 0,
            "same_adjacent_ratio": 0.0,
            "max_2gram_count": 0,
            "max_4gram_count": 0,
            "pitch_range": 0,
            "average_abs_pitch_interval": 0.0,
            "total_duration": 0.0,
            "average_duration": 0.0,
            "average_velocity": 0.0,
            "velocity_range": 0,
            "short_note_ratio": 0.0,
            "direction_change_count": 0,
            "phrase_repeat_ratio": 0.0,
            "cadence_stability": 0.0,
            "mode_contour_alignment": 0.0,
            "melodic_quality_score": 0.0,
        }

    if durations is None:
        durations = []
    if velocities is None:
        velocities = []
    if pitch_actions is None:
        pitch_actions = actions

    same_adjacent_count = sum(
        1 for index in range(1, len(pitch_actions))
        if pitch_actions[index] == pitch_actions[index - 1]
    )
    two_grams = Counter(
        tuple(pitch_actions[index:index + 2]) for index in range(len(pitch_actions) - 1)
    )
    four_grams = Counter(
        tuple(pitch_actions[index:index + 4]) for index in range(len(pitch_actions) - 3)
    )
    pitch_intervals = [
        abs(melody[index] - melody[index - 1])
        for index in range(1, len(melody))
    ]
    phrases = [
        pitch_actions[index:index + 4]
        for index in range(0, len(pitch_actions), 4)
        if len(pitch_actions[index:index + 4]) == 4
    ]
    adjacent_phrase_similarities = [
        _phrase_similarity(phrases[index], phrases[index + 1])
        for index in range(len(phrases) - 1)
    ]
    phrase_repeat_ratio = (
        sum(adjacent_phrase_similarities) / len(adjacent_phrase_similarities)
        if adjacent_phrase_similarities else 0.0
    )

    stable_pitch_classes = set()
    if base_notes:
        stable_pitch_classes = {base_notes[index] % 12 for index in [0, 2, 4, 7]}
    cadence_notes = [
        note for index, note in enumerate(melody, start=1)
        if index % 4 == 0
    ]
    cadence_stability = (
        sum(1 for note in cadence_notes if note % 12 in stable_pitch_classes)
        / len(cadence_notes)
        if cadence_notes and stable_pitch_classes else 0.0
    )
    same_adjacent_ratio = same_adjacent_count / max(1, len(pitch_actions) - 1)
    direction_change_count = _count_direction_changes(melody)
    mode_contour_alignment = _mode_contour_alignment(mode, melody)
    interval_smoothness = sum(1 for interval in pitch_intervals if 1 <= interval <= 7) / max(1, len(pitch_intervals))
    diversity_balance = min(1.0, len(set(pitch_actions)) / max(1, min(8, len(pitch_actions))))
    repetition_control = max(0.0, 1.0 - same_adjacent_ratio)
    phrase_balance = 1.0 - abs(phrase_repeat_ratio - 0.35)
    melodic_quality_score = round(
        100.0 * (
            0.25 * interval_smoothness
            + 0.20 * diversity_balance
            + 0.20 * repetition_control
            + 0.15 * cadence_stability
            + 0.10 * mode_contour_alignment
            + 0.10 * max(0.0, phrase_balance)
        ),
        2,
    )

    return {
        "length": len(actions),
        "unique_actions": len(set(actions)),
        "unique_action_ratio": len(set(actions)) / len(actions),
        "unique_pitch_actions": len(set(pitch_actions)),
        "unique_duration_values": len(set(durations)) if durations else 0,
        "same_adjacent_count": same_adjacent_count,
        "same_adjacent_ratio": same_adjacent_ratio,
        "max_2gram_count": max(two_grams.values()) if two_grams else 0,
        "max_4gram_count": max(four_grams.values()) if four_grams else 0,
        "pitch_range": max(melody) - min(melody) if melody else 0,
        "average_abs_pitch_interval": (
            sum(pitch_intervals) / len(pitch_intervals) if pitch_intervals else 0.0
        ),
        "total_duration": sum(float(duration) for duration in durations) if durations else 0.0,
        "average_duration": (
            sum(float(duration) for duration in durations) / len(durations) if durations else 0.0
        ),
        "average_velocity": (
            sum(float(velocity) for velocity in velocities) / len(velocities) if velocities else 0.0
        ),
        "velocity_range": max(velocities) - min(velocities) if velocities else 0,
        "short_note_ratio": (
            sum(1 for duration in durations if float(duration) <= 0.5) / len(durations)
            if durations else 0.0
        ),
        "direction_change_count": direction_change_count,
        "phrase_repeat_ratio": phrase_repeat_ratio,
        "cadence_stability": cadence_stability,
        "mode_contour_alignment": mode_contour_alignment,
        "interval_smoothness": interval_smoothness,
        "diversity_balance": diversity_balance,
        "repetition_control": repetition_control,
        "melodic_quality_score": melodic_quality_score,
    }


def calculate_selection_score(total_reward, metrics, mode=None):
    """최종 checkpoint/샘플 선택용 점수입니다.

    학습 reward만 보면 화성이나 종지 보상 때문에 반복적인 melody가 과대평가될 수 있습니다.
    그래서 사람이 듣는 품질과 가까운 지표를 함께 반영해 best model을 고릅니다.
    """
    short_note_excess = max(0.0, metrics.get("short_note_ratio", 0.0) - 0.85)
    if mode in ["sad", "dark"]:
        short_note_excess = max(0.0, metrics.get("short_note_ratio", 0.0) - 0.35)
    elif mode in ["neutral"]:
        short_note_excess = max(0.0, metrics.get("short_note_ratio", 0.0) - 0.65)

    return round(
        float(total_reward)
        + 0.45 * metrics.get("melodic_quality_score", 0.0)
        - 35.0 * metrics.get("same_adjacent_ratio", 0.0)
        - 18.0 * metrics.get("phrase_repeat_ratio", 0.0)
        - 28.0 * short_note_excess
        - 2.0 * max(0, metrics.get("max_4gram_count", 0) - 2),
        6,
    )


def calculate_checkpoint_selection_score(total_reward, info, mode, base_notes):
    metrics = calculate_melody_metrics(
        actions=info.get("actions", []),
        melody=info.get("melody", []),
        durations=info.get("durations", []),
        velocities=info.get("velocities", []),
        pitch_actions=info.get("pitch_actions", info.get("actions", [])),
        mode=mode,
        base_notes=base_notes,
    )
    return calculate_selection_score(total_reward, metrics, mode=mode), metrics


def save_experiment_summary(
    experiment_dir,
    classifier_result,
    detected_emotion,
    mode,
    emotion_scores,
    episodes,
    melody_length,
    rewards,
    actions,
    melody,
    durations,
    velocities,
    pitch_actions,
    duration_actions,
    velocity_actions,
    events,
    reward_breakdowns,
    note_names,
    agent,
    midi_filename,
    seed,
    state_mode,
    algorithm,
    training_metrics=None,
    env_config=None,
):
    if algorithm == "q_learning":
        agent_params = {
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "epsilon_final": agent.epsilon,
            "epsilon_decay": agent.epsilon_decay,
            "epsilon_min": agent.epsilon_min,
            "q_table_states": len(agent.q_table),
        }
    elif algorithm == "dqn":
        agent_params = {
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "hidden_size": agent.hidden_size,
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "epsilon_final": agent.epsilon,
            "epsilon_decay": agent.epsilon_decay,
            "epsilon_min": agent.epsilon_min,
            "use_double_dqn": agent.use_double_dqn,
            "replay_buffer_size": len(agent.memory),
        }
    elif algorithm == "factorized_dqn":
        agent_params = {
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "pitch_action_size": agent.pitch_action_size,
            "duration_action_size": agent.duration_action_size,
            "velocity_action_size": agent.velocity_action_size,
            "rhythm_action_size": agent.rhythm_action_size,
            "hidden_size": agent.hidden_size,
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "epsilon_final": agent.epsilon,
            "epsilon_decay": agent.epsilon_decay,
            "epsilon_min": agent.epsilon_min,
            "use_double_dqn": agent.use_double_dqn,
            "replay_buffer_size": len(agent.memory),
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    generated_metrics = calculate_melody_metrics(
        actions=actions,
        melody=melody,
        durations=durations,
        velocities=velocities,
        pitch_actions=pitch_actions,
        mode=mode,
        base_notes=(env_config or {}).get("base_notes"),
    )
    generated_reward_summary = summarize_reward_breakdowns(reward_breakdowns)
    generated_selection_score = calculate_selection_score(
        generated_reward_summary["total_reward"],
        generated_metrics,
        mode=mode,
    )

    summary = {
        "algorithm": algorithm,
        "classifier": classifier_result,
        "detected_emotion": detected_emotion,
        "music_mode": mode,
        "emotion_scores": emotion_scores,
        "episodes": episodes,
        "melody_length": melody_length,
        "state_mode": state_mode,
        "seed": seed,
        "env_config": env_config or {},
        "agent_params": agent_params,
        "reward_summary": {
            "last_reward": rewards[-1] if rewards else None,
            "last_100_avg": sum(rewards[-100:]) / min(len(rewards), 100) if rewards else None,
            "best_reward": max(rewards) if rewards else None,
            "worst_reward": min(rewards) if rewards else None,
        },
        "generated": {
            "actions": actions,
            "pitch_actions": pitch_actions,
            "duration_actions": duration_actions,
            "velocity_actions": velocity_actions,
            "events": events,
            "midi_notes": melody,
            "durations": durations,
            "velocities": velocities,
            "note_names": note_names,
            "midi_file": str(midi_filename),
            "selection_score": generated_selection_score,
            "reward_breakdown_summary": generated_reward_summary,
            "reward_breakdowns": reward_breakdowns,
            "metrics": generated_metrics,
        },
    }
    if training_metrics is not None:
        summary["training_metrics"] = {
            key: value for key, value in training_metrics.items() if key != "episode_losses"
        }

    summary_filename = experiment_dir / "summary.json"
    with open(summary_filename, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=2)

    reward_filename = experiment_dir / "episode_rewards.csv"
    save_reward_csv(rewards, reward_filename)

    loss_filename = None
    if training_metrics and "episode_losses" in training_metrics:
        loss_filename = experiment_dir / "episode_losses.csv"
        save_loss_csv(training_metrics["episode_losses"], loss_filename)

    reward_breakdown_filename = None
    if reward_breakdowns:
        reward_breakdown_filename = experiment_dir / "generated_reward_breakdown.csv"
        save_reward_breakdown_csv(reward_breakdowns, reward_breakdown_filename)

    return summary_filename, reward_filename, loss_filename, reward_breakdown_filename


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EmotionRL baseline or DQN agent.")
    parser.add_argument(
        "--algorithm",
        choices=["q_learning", "dqn", "factorized_dqn"],
        default="factorized_dqn",
        help="Training algorithm to run. The project default is factorized_dqn for event-factor learning.",
    )
    parser.add_argument(
        "--mood-source",
        choices=["mock", "deepface", "teammate"],
        default="teammate",
        help="Mood input source. The official project path uses the teammate EfficientNet classifier.",
    )
    parser.add_argument(
        "--mock-mood",
        default="sad",
        choices=["happy", "sad", "angry", "neutral", "excited", "dark", "unstable"],
        help="Legacy debug-only mood label for --mood-source mock.",
    )
    parser.add_argument(
        "--image",
        default=str(TEAM_CLASSIFIER_DEFAULT_IMAGE),
        help="Image path for the teammate classifier.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default=str(TEAM_CLASSIFIER_DEFAULT_CHECKPOINT),
        help="Checkpoint path for --mood-source teammate.",
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--melody-length", type=int, default=32)
    parser.add_argument(
        "--state-mode",
        choices=["table", "vector"],
        default="table",
        help="Use table state for Q-learning. Vector state is prepared for the future DQN trainer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--dqn-hidden-size", type=int, default=128)
    parser.add_argument("--dqn-learning-rate", type=float, default=0.0005)
    parser.add_argument("--dqn-epsilon-decay", type=float, default=0.999)
    parser.add_argument("--dqn-epsilon-min", type=float, default=0.05)
    parser.add_argument("--dqn-replay-capacity", type=int, default=20000)
    parser.add_argument(
        "--disable-double-dqn",
        action="store_true",
        help="Use the original DQN target instead of Double DQN target calculation.",
    )
    parser.add_argument(
        "--octave-expansion",
        action="store_true",
        help="After the expansion point, allow notes one octave below/above the base scale.",
    )
    parser.add_argument(
        "--expansion-start-ratio",
        type=float,
        default=0.5,
        help="Episode progress ratio where octave-expanded exploration starts.",
    )
    parser.add_argument(
        "--max-pitch-jump",
        type=int,
        default=12,
        help="Maximum allowed semitone jump when action masking is enabled.",
    )
    parser.add_argument(
        "--disable-action-masking",
        action="store_true",
        help="Allow the agent to choose any action without musical action masking.",
    )
    parser.add_argument(
        "--no-midi",
        action="store_true",
        help="Skip MIDI export when only checking training/logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    episodes = args.episodes
    melody_length = args.melody_length

    set_seed(args.seed)

    # 공식 입력 흐름:
    # image -> 팀원 EfficientNet classifier -> EmoSet 감정 확률
    # -> 수동 설계한 음악 mood 확률. 가장 높은 mood는 `mode`가 되고,
    # 전체 7-class mood vector는 RL 환경에 함께 전달됩니다.
    detected_emotion, mode, emotion_scores, classifier_result = analyze_mood(
        source=args.mood_source,
        mock_mood=args.mock_mood,
        image_path=image_path,
        classifier_checkpoint=Path(args.classifier_checkpoint),
    )

    print("=== EmotionRL Composer ===")
    print("Mood Source:", args.mood_source)
    if args.mood_source in ["deepface", "teammate"]:
        print("Image Path:", image_path)
    if args.mood_source == "teammate":
        print("Classifier Checkpoint:", args.classifier_checkpoint)
    print("Detected Emotion:", detected_emotion)
    print("Music Mode:", mode)
    print("Emotion Scores:")
    for emotion, score in emotion_scores.items():
        print(f"  {emotion}: {float(score):.2f}")
    print("Episodes:", episodes)
    print("Melody Length:", melody_length)
    print("Seed:", args.seed)
    print("Octave Expansion:", args.octave_expansion)
    print("Action Masking:", not args.disable_action_masking)
    print()

    if args.algorithm == "q_learning":
        # Q-learning은 주로 MelodyEnv의 mood별 reward/profile을 통해 `mode`를 사용합니다.
        # DQN 계열은 `mood_vector`를 state에 직접 포함할 수 있으므로,
        # 어떤 알고리즘을 쓰더라도 classifier의 전체 분포를 함께 넘깁니다.
        env, agent, rewards = train_agent(
            mode=mode,
            episodes=episodes,
            melody_length=melody_length,
            state_mode=args.state_mode,
            mood_vector=emotion_scores,
            octave_expansion=args.octave_expansion,
            expansion_start_ratio=args.expansion_start_ratio,
            max_pitch_jump=args.max_pitch_jump,
            action_masking=not args.disable_action_masking,
        )
        training_metrics = None
        state_mode = args.state_mode
    elif args.algorithm == "dqn":
        # 여기서는 classifier의 7-class mood vector가 DQN state의 일부가 됩니다.
        env, agent, rewards, training_metrics = train_dqn_agent(
            mode=mode,
            episodes=episodes,
            melody_length=melody_length,
            mood_vector=emotion_scores,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            hidden_size=args.dqn_hidden_size,
            learning_rate=args.dqn_learning_rate,
            epsilon_decay=args.dqn_epsilon_decay,
            epsilon_min=args.dqn_epsilon_min,
            replay_capacity=args.dqn_replay_capacity,
            use_double_dqn=not args.disable_double_dqn,
            octave_expansion=args.octave_expansion,
            expansion_start_ratio=args.expansion_start_ratio,
            max_pitch_jump=args.max_pitch_jump,
            action_masking=not args.disable_action_masking,
        )
        state_mode = "vector"
    else:
        # Factorized DQN도 같은 mood-conditioned state를 사용하지만,
        # pitch factor와 rhythm/velocity factor를 분리해서 예측합니다.
        env, agent, rewards, training_metrics = train_factorized_dqn_agent(
            mode=mode,
            episodes=episodes,
            melody_length=melody_length,
            mood_vector=emotion_scores,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            hidden_size=args.dqn_hidden_size,
            learning_rate=args.dqn_learning_rate,
            epsilon_decay=args.dqn_epsilon_decay,
            epsilon_min=args.dqn_epsilon_min,
            replay_capacity=args.dqn_replay_capacity,
            use_double_dqn=not args.disable_double_dqn,
            octave_expansion=args.octave_expansion,
            expansion_start_ratio=args.expansion_start_ratio,
            max_pitch_jump=args.max_pitch_jump,
            action_masking=not args.disable_action_masking,
        )
        state_mode = "vector"

    actions, melody, durations, velocities, pitch_actions, duration_actions, velocity_actions, events, reward_breakdowns = generate_melody(env, agent)
    note_names = convert_to_note_names(melody)

    print("\n=== 학습 완료 ===")
    print("Final Epsilon:", round(agent.epsilon, 3))
    print("Last 100 Avg Reward:", round(recent_average(rewards, 100), 2))

    print("\n=== Generated Melody ===")
    print("Actions:", actions)
    print("Pitch Actions:", pitch_actions)
    print("Durations:", durations)
    print("Velocities:", velocities)
    print("MIDI Notes:", melody)
    print("Note Names:", note_names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = EXPERIMENTS_ROOT / f"{timestamp}_{args.algorithm}_{mode}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    midi_filename = experiment_dir / f"generated_{mode}_melody.mid"

    if args.no_midi:
        print("\nMIDI export skipped.")
    else:
        try:
            export_melody_to_midi(
                melody,
                filename=midi_filename,
                durations=durations,
                velocities=velocities,
                tempo=MODE_PROFILES.get(mode, MODE_PROFILES["neutral"])["tempo"],
                instrument_name=MODE_PROFILES.get(mode, MODE_PROFILES["neutral"])["instrument"],
                chord_progression=MODE_PROFILES.get(mode, MODE_PROFILES["neutral"])["chord_progression"],
            )
            print("\nMIDI file saved:", str(midi_filename))
            print("Finder에서 이 파일을 더블클릭하면 생성된 멜로디를 들어볼 수 있습니다.")
        except ImportError as e:
            print("\nMIDI 저장 실패:", e)

    summary_filename, reward_filename, loss_filename, reward_breakdown_filename = save_experiment_summary(
        experiment_dir=experiment_dir,
        classifier_result=classifier_result,
        detected_emotion=detected_emotion,
        mode=mode,
        emotion_scores=emotion_scores,
        episodes=episodes,
        melody_length=melody_length,
        rewards=rewards,
        actions=actions,
        melody=melody,
        durations=durations,
        velocities=velocities,
        pitch_actions=pitch_actions,
        duration_actions=duration_actions,
        velocity_actions=velocity_actions,
        events=events,
        reward_breakdowns=reward_breakdowns,
        note_names=note_names,
        agent=agent,
        midi_filename=midi_filename,
        seed=args.seed,
        state_mode=state_mode,
        algorithm=args.algorithm,
        training_metrics=training_metrics,
        env_config={
            "octave_expansion": args.octave_expansion,
            "expansion_start_ratio": args.expansion_start_ratio,
            "max_pitch_jump": args.max_pitch_jump,
            "action_masking": not args.disable_action_masking,
            "mode_profile": MODE_PROFILES.get(mode, MODE_PROFILES["neutral"]),
            "reward_weights": env.reward_weights,
            "note_pool": env.notes,
            "base_notes": env.base_notes,
            "durations": env.durations,
            "velocities": env.velocities,
        },
    )

    print("\nExperiment summary saved:", str(summary_filename))
    print("Episode rewards saved:", str(reward_filename))
    if loss_filename is not None:
        print("Episode losses saved:", str(loss_filename))
    if reward_breakdown_filename is not None:
        print("Generated reward breakdown saved:", str(reward_breakdown_filename))


if __name__ == "__main__":
    main()
