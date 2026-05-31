from deepface import DeepFace


# DeepFace가 반환하는 감정 label을 RL 모델에서 사용할 음악 mode로 변환합니다.
def map_emotion_to_music_mode(emotion):
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


# 테스트할 이미지 경로입니다.
# 현재 프로젝트 폴더 기준으로 face/laugh.jpg 파일이 있어야 합니다.
image_path = "face/sad.jpg"

try:
    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
    )

    # DeepFace 버전에 따라 결과가 list로 반환될 수 있으므로 첫 번째 결과만 사용합니다.
    if isinstance(result, list):
        result = result[0]

    detected_emotion = result["dominant_emotion"]
    music_mode = map_emotion_to_music_mode(detected_emotion)

    print("\n=== 감정 분석 결과 ===")
    print("Detected Emotion:", detected_emotion)
    print("Music Mode:", music_mode)
    print("All Scores:")

    for emotion, score in result["emotion"].items():
        print(f"  {emotion}: {float(score):.2f}")

except Exception as e:
    print("Error occured:", e)