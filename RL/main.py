



from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from melody_env import MelodyEnv
from q_learning_agent import QLearningAgent

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

    DeepFace는 문자열 경로에 non-english 문자가 있으면 에러를 낼 수 있습니다.
    그래서 cv2.imdecode와 np.fromfile을 이용해 이미지를 numpy array로 읽습니다.
    """
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    return image


def analyze_face_emotion(image_path):
    """얼굴 사진을 분석해서 dominant emotion, music mode, 전체 score를 반환합니다."""
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


def train_agent(mode="happy", episodes=5000, melody_length=32):
    """지정한 감정 mode에 대해 Q-learning agent를 학습합니다."""
    env = MelodyEnv(mode=mode, melody_length=melody_length)
    agent = QLearningAgent(action_size=env.action_size)

    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

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


def generate_melody(env, agent):
    """학습된 agent를 이용해 탐험 없이 greedy 방식으로 멜로디를 생성합니다."""
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        state = next_state

    return info["actions"], info["melody"]


def convert_to_note_names(melody):
    """MIDI pitch 숫자를 사람이 읽기 쉬운 음 이름으로 변환합니다."""
    return [NOTE_NAME_MAP.get(note, str(note)) for note in melody]


def export_melody_to_midi(melody, filename="generated_melody.mid", note_duration=0.5):
    """생성된 MIDI pitch 리스트를 실제로 들을 수 있는 .mid 파일로 저장합니다."""
    if pretty_midi is None:
        raise ImportError(
            "pretty_midi가 설치되어 있지 않습니다. 터미널에서 `pip install pretty_midi`를 실행하세요."
        )

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )

    start_time = 0.0

    for pitch in melody:
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=start_time,
            end=start_time + note_duration,
        )
        piano.notes.append(note)
        start_time += note_duration

    midi.instruments.append(piano)
    midi.write(str(filename))


def main():
    # 여기를 바꾸면 다른 얼굴 사진으로 테스트할 수 있습니다.
    image_path = PROJECT_ROOT / "face" / "disgust.jpg"

    detected_emotion, mode, emotion_scores = analyze_face_emotion(image_path)

    episodes = 5000
    melody_length = 32

    print("=== EmotionRL Composer ===")
    print("Image Path:", image_path)
    print("Detected Emotion:", detected_emotion)
    print("Music Mode:", mode)
    print("Emotion Scores:")
    for emotion, score in emotion_scores.items():
        print(f"  {emotion}: {float(score):.2f}")
    print("Episodes:", episodes)
    print("Melody Length:", melody_length)
    print()

    env, agent, rewards = train_agent(
        mode=mode,
        episodes=episodes,
        melody_length=melody_length,
    )

    actions, melody = generate_melody(env, agent)
    note_names = convert_to_note_names(melody)

    print("\n=== 학습 완료 ===")
    print("Final Epsilon:", round(agent.epsilon, 3))
    print("Last 100 Avg Reward:", round(sum(rewards[-100:]) / 100, 2))

    print("\n=== Generated Melody ===")
    print("Actions:", actions)
    print("MIDI Notes:", melody)
    print("Note Names:", note_names)

    midi_filename = PROJECT_ROOT / "results" / f"generated_{mode}_melody.mid"

    try:
        midi_filename.parent.mkdir(parents=True, exist_ok=True)
        export_melody_to_midi(melody, filename=midi_filename)
        print("\nMIDI file saved:", str(midi_filename))
        print("Finder에서 이 파일을 더블클릭하면 생성된 멜로디를 들어볼 수 있습니다.")
    except ImportError as e:
        print("\nMIDI 저장 실패:", e)


if __name__ == "__main__":
    main()