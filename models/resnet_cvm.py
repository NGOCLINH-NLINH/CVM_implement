import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class DeterministicResNetCVM(nn.Module):
    def __init__(self, out_dim=384, pretrained=True):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, out_dim)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        mu = self.fc(x)
        mu = F.normalize(mu, p=2, dim=1)
        return mu
