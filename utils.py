# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import pickle

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset


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

    def add_batch(self, images, labels):
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        for i in range(images.shape[0]):
            item = (images[i].clone(), int(labels[i].item()))
            self.n_seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.capacity:
                    self.buffer[j] = item

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None
        batch = random.sample(self.buffer, k=min(batch_size, len(self.buffer)))
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return imgs, labels

    def __len__(self):
        return len(self.buffer)


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
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_full = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
    test_full = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)
    classes = train_full.classes
    num_classes = len(classes)
    per_task = num_classes // num_tasks
    tasks = []
    for t in range(num_tasks):
        start = t * per_task
        end = start + per_task if t < num_tasks - 1 else num_classes
        train_idx = [i for i, (_, lbl) in enumerate(train_full) if start <= lbl < end]
        test_idx = [i for i, (_, lbl) in enumerate(test_full) if start <= lbl < end]
        train_subset = Subset(train_full, train_idx)
        test_subset = Subset(test_full, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        tasks.append((train_loader, test_loader, list(range(start, end))))
    return tasks, classes


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def triplet_loss_k_negs(emb, pos_emb, neg_embs, margin=0.1):
    """
    emb: [Batch, Dim]
    pos_emb: [Batch, Dim]
    neg_embs: [Batch, K, Dim]
    """
    # (B, D) * (B, D) -> sum(dim=1) -> (B)
    cos_pos = (emb * pos_emb).sum(dim=1)
    d_pos = 1.0 - cos_pos  # [Batch]

    # emb.unsqueeze(1): [Batch, 1, Dim]
    # (B, 1, D) * (B, K, D) -> sum(dim=2) -> (B, K)
    cos_neg = (emb.unsqueeze(1) * neg_embs).sum(dim=2)
    d_neg = 1.0 - cos_neg  # [Batch, K]

    # d_pos.unsqueeze(1) được broadcast để so sánh với từng giá trị trong d_neg
    loss = torch.clamp(d_pos.unsqueeze(1) - d_neg + margin, min=0.0)

    return loss.mean()


def triplet_loss_seen_negs(emb, pos_emb, labels, anchors_tensor, seen_indices, margin=0.1):
    """
    Chỉ tính Triplet Loss dựa trên các lớp đã học (seen_indices).
    """
    device = emb.device
    anchors_seen = anchors_tensor[seen_indices].to(device)  # [Num_Seen, Dim]

    # Distance to positive
    cos_pos = (emb * pos_emb).sum(dim=1)
    d_pos = 1.0 - cos_pos  # [Batch]

    # [Batch, Dim] @ [Dim, Num_Seen] -> [Batch, Num_Seen]
    cos_seen = emb @ anchors_seen.t()
    d_seen = 1.0 - cos_seen

    seen_indices_tensor = torch.tensor(seen_indices, device=device).unsqueeze(0)  # [1, Num_Seen]
    mask = (seen_indices_tensor == labels.unsqueeze(1))  # [Batch, Num_Seen]

    # Loss matrix: max(0, d_pos - d_neg + margin)
    loss_mat = torch.clamp(d_pos.unsqueeze(1) - d_seen + margin, min=0.0)

    loss_mat[mask] = 0.0

    num_negs = anchors_seen.size(0) - 1
    if num_negs <= 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_mat.sum() / (emb.size(0) * num_negs)
