from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent / "datasets" / "EmoSet_24K_split"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_dataset = datasets.ImageFolder(
        root=DATA_ROOT / "train",
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        root=DATA_ROOT / "val",
        transform=eval_transform,
    )
    test_dataset = datasets.ImageFolder(
        root=DATA_ROOT / "test",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, train_dataset


if __name__ == "__main__":
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders()

    print(train_dataset.classes)
    print(
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    images, labels = next(iter(train_loader))
    print("batch shape:", images.shape, labels.shape)
