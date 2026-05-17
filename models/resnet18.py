import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        return self.backbone(x)
