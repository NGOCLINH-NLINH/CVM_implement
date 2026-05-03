import torch
import torch.nn as nn


class OLoRAManager:
    def __init__(self, model, threshold=0.97):
        self.model = model
        self.threshold = threshold
        self.P_matrices = {}
        self.activation_cache = {}
        self.hooks = []

    def _forward_hook(self, name):
        def hook(module, input, output):
            x = input[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])

            if name not in self.activation_cache:
                self.activation_cache[name] = x
            else:
                self.activation_cache[name] = torch.cat([self.activation_cache[name], x], dim=0)

        return hook

    def collect_features_and_compute_P(self, dataloader, device):
        print("\n[O-LoRA] Calculating ortho space...")
        for name, module in self.model.named_modules():
            if 'lora_A' in name and isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self._forward_hook(name)))

        self.model.eval()
        with torch.no_grad():
            for images, _, _ in dataloader:
                self.model(images.to(device))
                if len(list(self.activation_cache.values())[0]) > 2000:
                    break

        for h in self.hooks: h.remove()
        self.hooks = []

        for name, acts in self.activation_cache.items():
            cov = acts.T @ acts / acts.shape[0]
            U, S, V = torch.svd(cov)

            total_var = S.sum()
            cum_var = torch.cumsum(S, dim=0)
            k = torch.searchsorted(cum_var, self.threshold * total_var).item() + 1
            U_k = U[:, :k]

            I = torch.eye(acts.shape[1], device=acts.device)
            P_new = I - U_k @ U_k.T

            weight_name = name + '.weight'
            if weight_name in self.P_matrices:
                self.P_matrices[weight_name] = self.P_matrices[weight_name] @ P_new
            else:
                self.P_matrices[weight_name] = P_new

        self.activation_cache = {}
        print(f"[O-LoRA] Done")

    def apply_gradient_projection(self):
        for name, param in self.model.named_parameters():
            if 'lora_A' in name and param.grad is not None:
                if name in self.P_matrices:
                    P = self.P_matrices[name]
                    param.grad.data = param.grad.data @ P