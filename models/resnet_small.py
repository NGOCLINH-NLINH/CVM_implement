import torch
import torch.nn as nn
import torchvision.models as models


class SmallResNet18(nn.Module):
    def __init__(self, out_dim=384):
        super().__init__()
        base = models.resnet18(weights=None)
        # keep layers up to layer3 to make it smaller
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
