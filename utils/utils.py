# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import pickle

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, Dataset


class DualTransformDataset(Dataset):
    def __init__(self, dataset, transform_aug, transform_raw):
        self.dataset = dataset
        self.transform_aug = transform_aug
        self.transform_raw = transform_raw

    def __getitem__(self, index):
        img, label = self.dataset[index]

        img_aug = self.transform_aug(img)
        img_raw = self.transform_raw(img)

        return img_aug, img_raw, label

    def __len__(self):
        return len(self.dataset)


def load_anchors(path, device='cpu'):
    with open(path, 'rb') as f:
        anchors = pickle.load(f)
    keys = list(anchors.keys())
    mat = np.stack([anchors[k] for k in keys])
    tensor = torch.from_numpy(mat).float().to(device)
    tensor = F.normalize(tensor, p=2, dim=1)
    return keys, tensor


class ReservoirBuffer:
    def __init__(self, capacity=500, seed=42):
        self.capacity = capacity
        self.n_seen = 0
        self.buffer = []
        random.seed(seed)
        np.random.seed(seed)

    def add_batch(self, images, labels, task_id):
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        for i in range(images.shape[0]):
            item = (images[i].clone(), int(labels[i].item()), task_id)
            self.n_seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.capacity:
                    self.buffer[j] = item

    def get_all_data(self):
        if len(self.buffer) == 0:
            return [], [], []
        imgs = torch.stack([b[0] for b in self.buffer])
        labels = torch.tensor([b[1] for b in self.buffer], dtype=torch.long)
        task_ids = torch.tensor([b[2] for b in self.buffer], dtype=torch.long)
        return imgs, labels, task_ids


def triplet_loss_emb(emb, pos_emb, neg_emb, margin=0.1):
    d_pos = 1.0 - F.cosine_similarity(emb, pos_emb, dim=1)
    d_neg = 1.0 - F.cosine_similarity(emb, neg_emb, dim=1)
    loss = F.relu(d_pos - d_neg + margin).mean()
    return loss


def semantic_distance_loss(emb, emb_prev, old_anchor_matrix):
    if old_anchor_matrix is None or old_anchor_matrix.shape[0] == 0:
        return torch.tensor(0.0, device=emb.device)
    cos_t = emb @ old_anchor_matrix.t()
    cos_prev = emb_prev @ old_anchor_matrix.t()
    d_t = 1.0 - cos_t
    d_prev = 1.0 - cos_prev
    return F.mse_loss(d_t, d_prev)


def make_cifar100_tasks(num_tasks, batch_size, augment=True):
    if augment:
        transform_train_aug = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        transform_train_aug = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_full = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train_aug)
    test_full = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)

    classes = train_full.classes
    num_classes = len(classes)
    per_task = num_classes // num_tasks
    tasks = []

    all_targets = np.array(train_full.targets)
    all_test_targets = np.array(test_full.targets)

    for t in range(num_tasks):
        start = t * per_task
        end = start + per_task if t < num_tasks - 1 else num_classes

        train_idx = np.where((all_targets >= start) & (all_targets < end))[0]
        test_idx = np.where((all_test_targets >= start) & (all_test_targets < end))[0]

        train_subset = Subset(train_full, train_idx)
        test_subset = Subset(test_full, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        tasks.append((train_loader, test_loader, list(range(start, end))))

    return tasks, classes


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def triplet_loss_k_negs(emb, pos_emb, neg_embs, margin=0.1):
    cos_pos = (emb * pos_emb).sum(dim=1)
    d_pos = 1.0 - cos_pos

    cos_neg = (emb.unsqueeze(1) * neg_embs).sum(dim=2)
    d_neg = 1.0 - cos_neg
    loss = torch.clamp(d_pos.unsqueeze(1) - d_neg + margin, min=0.0)

    return loss.mean()


def triplet_loss_seen_negs(emb, pos_emb, labels, anchors_tensor, seen_indices, margin=0.1):
    device = emb.device
    anchors_seen = anchors_tensor[seen_indices].to(device)

    cos_pos = (emb * pos_emb).sum(dim=1)
    d_pos = 1.0 - cos_pos

    cos_seen = emb @ anchors_seen.t()
    d_seen = 1.0 - cos_seen

    seen_indices_tensor = torch.tensor(seen_indices, device=device).unsqueeze(0)
    mask = (seen_indices_tensor == labels.unsqueeze(1))

    loss_mat = torch.clamp(d_pos.unsqueeze(1) - d_seen + margin, min=0.0)

    loss_mat[mask] = 0.0

    num_negs = anchors_seen.size(0) - 1
    if num_negs <= 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_mat.sum() / (emb.size(0) * num_negs)


def adaptive_margin_triplet_loss_seen_negs(emb, pos_emb, labels, anchors_tensor, seen_indices, base_margin=0.1):
    device = emb.device
    anchors_seen = anchors_tensor[seen_indices].to(device)

    cos_pos = (emb * pos_emb).sum(dim=1)
    d_pos = 1.0 - cos_pos  # [Batch]

    cos_seen = emb @ anchors_seen.t()
    d_seen = 1.0 - cos_seen

    anchor_sim = pos_emb @ anchors_seen.t()
    adaptive_margin = base_margin * (1.0 - anchor_sim).clamp(min=0.0)
    loss_mat = torch.clamp(d_pos.unsqueeze(1) - d_seen + adaptive_margin, min=0.0)

    seen_indices_tensor = torch.tensor(seen_indices, device=device).unsqueeze(0)
    mask = (seen_indices_tensor == labels.unsqueeze(1))
    loss_mat[mask] = 0.0

    num_negs = anchors_seen.size(0) - 1
    if num_negs <= 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_mat.sum() / (emb.size(0) * num_negs)


def anchor_attraction_loss(emb, pos_emb):
    return (1.0 - (emb * pos_emb).sum(dim=1)).mean()


def image_side_prototype_spread_loss(emb, labels, anchors_tensor, seen_indices, delta=0.3):
    device = emb.device

    if len(seen_indices) <= 1:
        return torch.tensor(0.0, device=device)

    anchors_seen = anchors_tensor[seen_indices].to(device)

    cos_sim = emb @ anchors_seen.t()
    seen_idx_tensor = torch.tensor(seen_indices, device=device).unsqueeze(0)
    pos_mask = (seen_idx_tensor == labels.unsqueeze(1))

    neg_cos = cos_sim.masked_fill(pos_mask, -1.0)

    loss_mat = torch.clamp(neg_cos - delta, min=0.0)

    num_negs = anchors_seen.size(0) - 1
    return loss_mat.sum() / (emb.size(0) * num_negs)


def adaptive_margin_triplet_loss_k_negs(emb, pos, neg_k, base_margin=0.1, reduction="mean"):
    sim_pos = (emb * pos).sum(dim=1, keepdim=True)
    sim_neg = (emb.unsqueeze(1) * neg_k).sum(dim=2)

    anchor_sim = (pos.unsqueeze(1) * neg_k).sum(dim=2)

    adaptive_margin = base_margin * (1.0 - anchor_sim).clamp(min=0.0)
    loss = F.relu(sim_neg - sim_pos + adaptive_margin)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def standard_margin_loss(mu, labels, anchors_tensor, seen_inds, margin=0.5):
    pos_anchors = anchors_tensor[labels]
    L_pull = torch.mean(torch.sum((mu - pos_anchors) ** 2, dim=1))

    L_push = torch.tensor(0.0, device=mu.device)
    if len(seen_inds) > 1:
        seen_anchors = anchors_tensor[seen_inds]

        dists = torch.cdist(mu, seen_anchors, p=2.0)

        mask = torch.ones_like(dists, dtype=torch.bool)
        for i, lbl in enumerate(labels):
            idx = (torch.tensor(seen_inds, device=mu.device) == lbl).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                mask[i, idx[0]] = False

        neg_dists = dists[mask]

        if len(neg_dists) > 0:
            L_push = torch.mean(torch.clamp(margin - neg_dists, min=0.0))

    return L_pull + L_push


def kl_divergence_loss(mu, log_var, mu_prev, log_var_prev):
    var = torch.exp(log_var) + 1e-6
    var_prev = torch.exp(log_var_prev) + 1e-6
    term1 = log_var - log_var_prev
    term2 = (var_prev + (mu_prev - mu) ** 2) / var
    kl_div = 0.5 * torch.sum(term1 + term2 - 1.0, dim=1)

    return kl_div.mean()


def variance_regularization_loss(log_var, target_val=0.0):
    return torch.mean((log_var - target_val)**2)


def evaluate_all_seen_multi_lora(model, task_router, test_loader, anchors_tensor, device):
    model.eval()
    task_router.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            with model.disable_adapter():
                feats = model.base_model.features(images)
                feats = model.base_model.avgpool(feats)
                feats = torch.flatten(feats, 1)

            task_preds = task_router(feats).argmax(dim=1)
            unique_tasks = torch.unique(task_preds)

            for task_id in unique_tasks:
                mask = (task_preds == task_id)
                sub_images = images[mask]
                sub_labels = labels[mask]
                adapter_name = f"task_{task_id.item()}"

                if adapter_name in model.peft_config:
                    model.set_adapter(adapter_name)

                    sub_mu = model(sub_images)
                    sims = sub_mu @ anchors_tensor.t()
                    final_preds = sims.argmax(dim=1).cpu()

                    correct += (final_preds == sub_labels.cpu()).sum().item()
                total += len(sub_labels)

    return correct / total if total > 0 else 0.0