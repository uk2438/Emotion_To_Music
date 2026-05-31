from pathlib import Path

DATA_ROOT = Path("datasets/EmoSet_24K_split")

for split in ["train", "val", "test"]:
    print(f"\n[{split}]")
    total = 0

    split_path = DATA_ROOT / split
    if not split_path.exists():
        print(f"(경로 없음: {split_path})")
        continue

    for class_dir in sorted(split_path.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.glob("*.jpg")))
            total += n
            print(f"{class_dir.name}: {n}")

    print(f"total: {total}")
