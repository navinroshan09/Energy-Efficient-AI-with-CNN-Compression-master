import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = mobilenet_v2(weights=None)
        in_feat = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        return self.net(x)
