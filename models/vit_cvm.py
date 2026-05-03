import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTCVM(nn.Module):
    def __init__(self, out_dim=384, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=0)

        self.fc = nn.Linear(self.vit.num_features, out_dim)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        features = self.vit(x)
        mu = self.fc(features)
        mu = F.normalize(mu, p=2, dim=1)
        return mu