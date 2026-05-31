"""
EmoSet-118K에서 클래스별 3000장(총 24K)을 샘플링해
train/val/test 폴더 구조로 이미지를 저장합니다.
"""

from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from tqdm import tqdm
import random

# =========================
# 설정
# =========================

DATASET_NAME = "Woleek/EmoSet-118K"

SAVE_ROOT = Path(__file__).resolve().parent / "datasets" / "EmoSet_24K_split"

NUM_PER_CLASS = 3000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMAGE_COL = "image"
LABEL_COL = "label"

CLASSES = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
]

random.seed(SEED)


def main() -> None:
    # =========================
    # 데이터셋 로드
    # =========================

    dataset = load_dataset(DATASET_NAME)

    full_dataset = concatenate_datasets([
        dataset["train"],
        dataset["val"],
        dataset["test"],
    ])

    print(full_dataset)
    print(full_dataset.column_names)
    print(full_dataset[0])

    # =========================
    # 라벨 정보 확인
    # =========================

    label_feature = full_dataset.features[LABEL_COL]

    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_names = label_feature.names
    else:
        label_names = sorted(set(full_dataset[LABEL_COL]))

    print("Label names:", label_names)

    # =========================
    # 클래스별 3000장 샘플링 후 split 저장
    # =========================

    for class_name in CLASSES:
        print(f"\nProcessing class: {class_name}")

        if isinstance(full_dataset[0][LABEL_COL], int):
            class_idx = label_names.index(class_name)

            indices = [
                i for i in range(len(full_dataset))
                if full_dataset[i][LABEL_COL] == class_idx
            ]
        else:
            indices = [
                i for i in range(len(full_dataset))
                if full_dataset[i][LABEL_COL] == class_name
            ]

        if len(indices) < NUM_PER_CLASS:
            raise ValueError(
                f"{class_name} 클래스 이미지 수가 부족합니다. "
                f"필요: {NUM_PER_CLASS}, 현재: {len(indices)}"
            )

        selected_indices = random.sample(indices, NUM_PER_CLASS)
        random.shuffle(selected_indices)

        n_train = int(NUM_PER_CLASS * TRAIN_RATIO)
        n_val = int(NUM_PER_CLASS * VAL_RATIO)

        split_indices = {
            "train": selected_indices[:n_train],
            "val": selected_indices[n_train : n_train + n_val],
            "test": selected_indices[n_train + n_val :],
        }

        for split_name, indices_for_split in split_indices.items():
            save_dir = SAVE_ROOT / split_name / class_name
            save_dir.mkdir(parents=True, exist_ok=True)

            for count, idx in enumerate(
                tqdm(indices_for_split, desc=f"{split_name}/{class_name}")
            ):
                item = full_dataset[idx]
                image = item[IMAGE_COL].convert("RGB")

                save_path = save_dir / f"{class_name}_{count:05d}.jpg"
                image.save(save_path, quality=95)

            print(
                f"{split_name}/{class_name}: "
                f"{len(indices_for_split)} images saved"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
