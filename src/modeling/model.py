import torch.nn as nn
from torchvision import models


def get_renaissance_model(num_classes=3):

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model