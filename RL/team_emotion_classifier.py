from pathlib import Path


EMOSET_CLASSES = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
]

MOOD_LABELS = [
    "happy",
    "neutral",
    "sad",
    "dark",
    "angry",
    "excited",
    "unstable",
]


def _load_checkpoint(path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_model(num_classes, device):
    import torch.nn as nn
    from torchvision import models

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)


def _build_eval_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _reorder_emoset_probs(raw_probs, classes):
    import torch

    ordered_probs = torch.zeros(len(EMOSET_CLASSES))
    for index, class_name in enumerate(classes):
        target_index = EMOSET_CLASSES.index(class_name)
        ordered_probs[target_index] = raw_probs[index]
    return ordered_probs


def _emoset_to_music_mood(emoset_probs):
    """팀원 classifier 출력을 RL 음악 생성기가 사용하는 mood 공간으로 변환합니다.

    EfficientNet checkpoint는 EmoSet의 8개 시각 감정 class를 예측합니다.
    하지만 음악 생성기는 이 label을 그대로 쓰지 않고, MelodyEnv의 음계,
    tempo, rhythm, reward profile과 연결된 7개 음악 mood를 사용합니다.
    아래 weighted remapping은 이미지 감정 분류 결과를 RL 음악 생성용 감정
    벡터로 바꿔주는 수동 설계 bridge입니다.
    """
    import torch

    amusement = emoset_probs[0]
    anger = emoset_probs[1]
    awe = emoset_probs[2]
    contentment = emoset_probs[3]
    disgust = emoset_probs[4]
    excitement = emoset_probs[5]
    fear = emoset_probs[6]
    sadness = emoset_probs[7]

    # 출력 순서는 반드시 MOOD_LABELS와 같아야 합니다:
    # happy, neutral, sad, dark, angry, excited, unstable.
    # 하나의 시각 감정이 여러 음악 mood에 영향을 줄 수 있게 설계했습니다.
    # 예를 들어 amusement는 happy와 excited에 모두 반영되고,
    # fear/disgust/sadness는 dark 또는 unstable 분위기에 섞여 들어갑니다.
    mood_scores = torch.stack([
        amusement + 0.4 * contentment,
        contentment + 0.3 * awe,
        sadness,
        0.6 * fear + 0.5 * disgust + 0.4 * sadness + 0.3 * awe,
        anger + 0.4 * disgust,
        excitement + 0.4 * amusement + 0.3 * awe,
        0.6 * fear + 0.4 * anger + 0.3 * disgust + 0.3 * excitement,
    ])
    mood_probs = mood_scores / (mood_scores.sum() + 1e-8)
    top_mood = MOOD_LABELS[torch.argmax(mood_probs).item()]
    return mood_probs, top_mood


def classify_mood_from_image(image_path, checkpoint_path):
    """팀원 EfficientNet classifier를 실행하고 RL 입력 형식으로 반환합니다.

    처리 흐름:
    image -> EfficientNet logits -> EmoSet 확률 -> 음악 mood 확률
    -> top mood label. 반환값의 "scores" 필드는 MelodyEnv/RL 학습에
    전달되는 7-class 음악 mood vector입니다.
    """
    import torch
    from PIL import Image

    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)

    if not image_path.exists():
        raise FileNotFoundError(f"image file not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    classes = checkpoint.get("classes", EMOSET_CLASSES)

    model = _build_model(num_classes=len(classes), device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = _build_eval_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        raw_probs = torch.softmax(logits, dim=1)[0].cpu()

    # checkpoint 안의 class 순서가 다를 수 있으므로 먼저 EMOSET_CLASSES
    # 기준으로 확률 순서를 맞춘 뒤, 그 결과를 음악 mood로 다시 매핑합니다.
    emoset_probs = _reorder_emoset_probs(raw_probs, classes)
    mood_probs, top_mood = _emoset_to_music_mood(emoset_probs)

    return {
        "source": "teammate_efficientnet",
        "label": top_mood,
        "scores": {
            label: float(mood_probs[index])
            for index, label in enumerate(MOOD_LABELS)
        },
        "raw_label": EMOSET_CLASSES[int(torch.argmax(emoset_probs).item())],
        "raw_scores": {
            label: float(emoset_probs[index])
            for index, label in enumerate(EMOSET_CLASSES)
        },
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }
