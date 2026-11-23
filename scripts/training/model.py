"""
MobileNetV3-Small with a new classifier head
"""

from torch import nn
from torchvision import models


def build_model(
    num_classes: int, freeze_backbone: bool = True, pretrained: bool = True
) -> nn.Module:
    """
    Builds the MobileNetV3-Small with a new classifier head.
    """
    model = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)

    last_layer = model.classifier[-1]
    assert isinstance(last_layer, nn.Linear)
    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)

    set_backbone_trainable(model, not freeze_backbone)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """
    Freezes or unfreezey the pretrained feature extractor. The head always trains.
    """
    assert isinstance(model.features, nn.Module)
    for param in model.features.parameters():
        param.requires_grad = trainable
