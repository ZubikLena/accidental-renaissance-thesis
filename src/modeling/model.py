import torch.nn as nn
from torchvision import models


def get_model(model_name="resnet", num_classes=3):

    if model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model