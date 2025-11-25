import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNetCVM(nn.Module):
    def __init__(self, out_dim=384, pretrained=False):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-2])  # conv..layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
