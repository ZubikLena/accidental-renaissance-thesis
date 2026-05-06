from typing import Any, Dict

import torch
from torch import nn
from torchvision import models


def build_model(config: Dict[str, Any], num_classes: int) -> nn.Module:
    model_config = config.get("model", {})
    name = model_config.get("name", "resnet50").lower()
    pretrained = bool(model_config.get("pretrained", True))
    hidden_units = model_config.get("head_hidden")
    dropout = float(model_config.get("dropout", 0.0))
    freeze_backbone = bool(model_config.get("freeze_backbone", True))

    if name in {"resnet50", "resnet"}:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        if hidden_units:
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, num_classes),
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)
        head_parameters = [param for param in model.fc.parameters()]
    elif name in {"vit", "vit_b_16", "vision_transformer", "vision-transformer"}:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        if hidden_units:
            model.heads.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, num_classes),
            )
        else:
            model.heads.head = nn.Linear(in_features, num_classes)
        head_parameters = [param for param in model.heads.head.parameters()]
    else:
        raise ValueError(f"Unsupported model name: {name}. Use resnet50 or vit_b_16.")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in head_parameters:
            param.requires_grad = True

    return model
