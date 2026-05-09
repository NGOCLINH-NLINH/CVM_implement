import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNetCVM(nn.Module):
    def __init__(self, out_dim=384, pretrained=False, sparsity_ratio=0.40, leak=0.05):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else models.resnet18(
            weights=None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

        self.sparsity_ratio = sparsity_ratio
        self.leak = leak

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.sparsity_ratio < 1.0:
            k = max(1, int(x.size(1) * self.sparsity_ratio))
            # _, idx = torch.topk(x, k, dim=1)
            # mask = torch.full_like(x, self.leak)
            # mask.scatter_(1, idx, 1.0)
            # x = x * mask
            val, idx = torch.topk(x, k, dim=1)
            sparse_x = torch.zeros_like(x).scatter_(1, idx, val)
            x = sparse_x

        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
