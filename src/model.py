from typing import Any, Dict

import torch
from torch import nn
from torchvision import models


def build_model(config: Dict[str, Any], num_classes: int) -> nn.Module:
    model_config = config.get("model", {})
    name = model_config.get("architecture", "resnet50").lower()
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", True))

    if name in {"resnet50", "resnet"}:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        if 'head' in model_config:
            head_config = model_config['head']
            layers = head_config['layers'] + [num_classes]
            seq = []
            for i, out_features in enumerate(layers):
                seq.append(nn.Linear(in_features, out_features))
                if i < len(layers) - 1:
                    seq.append(nn.ReLU())
                in_features = out_features
            model.fc = nn.Sequential(*seq)
        else:
            model.fc = nn.Linear(in_features, num_classes)
        head_parameters = [param for param in model.fc.parameters()]
    elif name in {"vit", "vit_b_16", "vision_transformer", "vision-transformer"}:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        if 'head' in model_config:
            head_config = model_config['head']
            layers = head_config['layers'] + [num_classes]
            seq = []
            for i, out_features in enumerate(layers):
                seq.append(nn.Linear(in_features, out_features))
                if i < len(layers) - 1:
                    seq.append(nn.ReLU())
                in_features = out_features
            model.heads.head = nn.Sequential(*seq)
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
            param.requires_grad = False
        for param in head_parameters:
            param.requires_grad = True

    return model
