import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from copy import deepcopy

from models.vit_lora import VisionTransformer, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn, \
    Attention_LoRA


class ViT_lora_co(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x, *args, task=-1, get_cur_x=False, **kwargs):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x, task=task, get_cur_x=get_cur_x)

        x = self.norm(x)
        return x

    def forward(self, x, *args, task=-1, get_cur_x=False, **kwargs):
        x = self.forward_features(x, task=task, get_cur_x=get_cur_x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x


def create_vit_small_lora(pretrained=True, **kwargs):
    variant = 'vit_small_patch16_224_in21k'
    pretrained_cfg = resolve_pretrained_cfg(variant)
    model = build_model_with_cfg(
        ViT_lora_co, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model


class ViT_ACVM(nn.Module):
    def __init__(self, num_tasks=10, rank=64):
        super().__init__()
        model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                            n_tasks=num_tasks, rank=rank)

        self.image_encoder = create_vit_small_lora(pretrained=True, **model_kwargs)
        self.num_task = 0
        self.fea_in = defaultdict(dict)
        self.prototypes = {}

        for module in self.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

    def forward(self, x, get_cur_x=False):
        task_id = self.num_task - 1 if self.num_task > 0 else 0

        cls_feature = self.image_encoder(x, task=task_id, get_cur_x=get_cur_x)
        return F.normalize(cls_feature, p=2, dim=1)

    def update_task(self):
        self.num_task += 1

    def freeze_for_task(self):
        for name, param in self.named_parameters():
            param.requires_grad_(False)
            task_id = self.num_task - 1

            if f"lora_A_k.{task_id}" in name or f"lora_A_v.{task_id}" in name:
                param.requires_grad_(True)
            elif f"lora_B_k.{task_id}" in name or f"lora_B_v.{task_id}" in name:
                param.requires_grad_(True)

    def extract_fea_in(self, device):
        task_id = self.num_task - 1
        for module in self.modules():
            if isinstance(module, Attention_LoRA):
                self.fea_in[module.lora_A_k[task_id].weight] = deepcopy(module.cur_matrix).to(device)
                self.fea_in[module.lora_A_v[task_id].weight] = deepcopy(module.cur_matrix).to(device)
                self.fea_in[module.lora_B_k[task_id].weight] = deepcopy(module.cur_matrix).to(device)
                self.fea_in[module.lora_B_v[task_id].weight] = deepcopy(module.cur_matrix).to(device)
                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
        return self.fea_in

    @torch.no_grad()
    def build_prototypes(self, train_loader, class_indices, device):
        self.eval()
        class_features = {c: [] for c in class_indices}
        for images_aug, images_raw, labels in train_loader:
            emb = self(images_raw.to(device))
            for i, label in enumerate(labels):
                c = int(label.item())
                if c in class_features:
                    class_features[c].append(emb[i].cpu())
        for c in class_indices:
            if class_features[c]:
                proto = torch.stack(class_features[c]).mean(dim=0)
                self.prototypes[c] = F.normalize(proto, dim=0)