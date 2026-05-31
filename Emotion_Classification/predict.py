import argparse
from pathlib import Path

import torch
from PIL import Image

from data_loaders import eval_transform
from model import EMOSET_CLASSES, build_model

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"

MOOD_CLASSES = [
    "HAPPY",
    "NEUTRAL",
    "SAD",
    "DARK",
    "ANGRY",
    "EXCITED",
    "UNSTABLE",
]


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def emoset_to_music_mood(
    emoset_probs: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    """
    EmoSet 8-class probability vector를
    음악 생성용 7-class mood vector로 변환합니다.

    Args:
        emoset_probs: shape [8], order = EMOSET_CLASSES

    Returns:
        mood_probs: shape [7], order = MOOD_CLASSES
        top_mood: str
    """
    amusement = emoset_probs[0]
    anger = emoset_probs[1]
    awe = emoset_probs[2]
    contentment = emoset_probs[3]
    disgust = emoset_probs[4]
    excitement = emoset_probs[5]
    fear = emoset_probs[6]
    sadness = emoset_probs[7]

    mood_scores = torch.stack([
        amusement + 0.4 * contentment,                          # HAPPY
        contentment + 0.3 * awe,                                # NEUTRAL
        sadness,                                                # SAD
        0.6 * fear + 0.5 * disgust + 0.4 * sadness + 0.3 * awe,  # DARK
        anger + 0.4 * disgust,                                  # ANGRY
        excitement + 0.4 * amusement + 0.3 * awe,               # EXCITED
        0.6 * fear + 0.4 * anger + 0.3 * disgust + 0.3 * excitement,  # UNSTABLE
    ])

    mood_probs = mood_scores / (mood_scores.sum() + 1e-8)
    top_mood = MOOD_CLASSES[torch.argmax(mood_probs).item()]

    return mood_probs, top_mood


def reorder_emoset_probs(
    emoset_probs: torch.Tensor,
    classes: list[str],
) -> torch.Tensor:
    """checkpoint class 순서 → EMOSET_CLASSES 고정 순서."""
    ordered_probs = torch.zeros(len(EMOSET_CLASSES))

    for i, class_name in enumerate(classes):
        target_idx = EMOSET_CLASSES.index(class_name)
        ordered_probs[target_idx] = emoset_probs[i]

    return ordered_probs


@torch.no_grad()
def predict(
    image_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    checkpoint = load_checkpoint(checkpoint_path, device)

    classes = checkpoint.get("classes", EMOSET_CLASSES)
    num_classes = len(classes)

    model = build_model(num_classes, device, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0).to(device)

    logits = model(image_tensor)
    raw_probs = torch.softmax(logits, dim=1)[0].cpu()
    ordered_probs = reorder_emoset_probs(raw_probs, classes)

    mood_probs, top_mood = emoset_to_music_mood(ordered_probs)

    return ordered_probs, mood_probs, top_mood


def print_results(
    ordered_probs: torch.Tensor,
    mood_probs: torch.Tensor,
    top_mood: str,
) -> None:
    print("\n==============================")
    print("EmoSet emotion probabilities")
    print("==============================")

    for idx in torch.argsort(ordered_probs, descending=True):
        class_name = EMOSET_CLASSES[idx]
        print(f"{class_name:12s}: {ordered_probs[idx].item():.4f}")

    print("\n==============================")
    print("Mapped music mood probabilities")
    print("==============================")

    for idx in torch.argsort(mood_probs, descending=True):
        print(f"{MOOD_CLASSES[idx]:9s}: {mood_probs[idx].item():.4f}")

    print("\nFinal mood:", top_mood)


def main() -> None:
    parser = argparse.ArgumentParser(description="EmoSet → music mood inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="추론할 이미지 경로",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="학습된 checkpoint 경로",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print(f"image: {image_path}")
    print(f"checkpoint: {checkpoint_path}")

    ordered_probs, mood_probs, top_mood = predict(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    print_results(ordered_probs, mood_probs, top_mood)


if __name__ == "__main__":
    main()
