import torch
import torch.nn as nn
from torchvision import models

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


def build_model(
    num_classes: int,
    device: torch.device,
    pretrained: bool = False,
) -> nn.Module:
    weights = (
        models.EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model.to(device)
