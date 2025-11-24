# utils.py
import torch
import torch.nn.functional as F
import random
import pickle
import os
import numpy as np
from collections import deque


def load_anchors(path, device='gpu'):
    """
    Load pickled anchors: a dict {label: np.array}
    Returns: (anchor_keys, anchor_tensor (torch.FloatTensor on device))
    """
    with open(path, 'rb') as f:
        anchors = pickle.load(f)
    keys = list(anchors.keys())
    mat = np.stack([anchors[k] for k in keys])
    tensor = torch.from_numpy(mat).float().to(device)
    # L2-normalize just in case
    tensor = F.normalize(tensor, p=2, dim=1)
    return keys, tensor


# ---------------- Memory buffer (reservoir) ----------------
class ReservoirBuffer:
    """
    Reservoir sampling buffer for rehearsal.
    - capacity: total stored samples
    - store items as tuples (image_tensor, label)
    On Kaggle we may store indices + dataset reference to avoid large memory usage.
    For simplicity here we store (img, label) in-memory (CIFAR images are small).
    """

    def __init__(self, capacity=1000, seed=42):
        self.capacity = capacity
        self.n_seen = 0
        self.buffer = []
        random.seed(seed)
        np.random.seed(seed)

    def add_batch(self, images, labels):
        """
        images: tensor [B, C, H, W] (cpu)
        labels: tensor [B]
        """
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        for i in range(images.shape[0]):
            item = (images[i].clone(), int(labels[i].item()))
            self.n_seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                # reservoir sampling
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


# ---------------- Losses ----------------
def triplet_loss_emb(emb, pos_emb, neg_emb, margin=0.1):
    """
    emb: [B, D]  - image embedding
    pos_emb: [B, D] - positive anchor embedding (correct class)
    neg_emb: [B, D] - negative anchor embedding (random other class)
    Using cosine distance: d = 1 - cos
    loss = max(0, d_pos - d_neg + margin)
    """
    d_pos = 1.0 - F.cosine_similarity(emb, pos_emb, dim=1)
    d_neg = 1.0 - F.cosine_similarity(emb, neg_emb, dim=1)
    loss = F.relu(d_pos - d_neg + margin).mean()
    return loss


def distance_loss(emb, emb_prev, old_anchor_matrix):
    """
    Enforce that distances from sample to previous anchors remain similar across model versions.
    - emb: [B, D] current
    - emb_prev: [B, D] from previous model (detached)
    - old_anchor_matrix: [K, D], K previous classes
    Compute pairwise cosine distances: [B, K], compare by MSE.
    """
    if old_anchor_matrix is None or old_anchor_matrix.shape[0] == 0:
        return torch.tensor(0.0, device=emb.device)
    # compute distances: 1 - cos_sim
    # expand to [B, K, D] in cosine_similarity via broadcasting
    # Use torch.matmul for efficiency after normalization
    # Ensure all normalized
    emb = F.normalize(emb, p=2, dim=1)
    emb_prev = F.normalize(emb_prev, p=2, dim=1)
    anchor = F.normalize(old_anchor_matrix, p=2, dim=1)

    # cosine similarity: emb @ anchor.T  => [B, K]
    cos_t = emb @ anchor.t()
    cos_prev = emb_prev @ anchor.t()

    d_t = 1.0 - cos_t
    d_prev = 1.0 - cos_prev

    loss = F.mse_loss(d_t, d_prev)
    return loss


# ---------------- Evaluation helpers ----------------
def predict_from_anchors(model, dataloader, anchor_tensor, anchor_keys, device='cuda'):
    """
    Compute predictions for dataloader using nearest anchor (cosine).
    Returns: (y_true, y_pred_labels)
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        anchor = anchor_tensor.to(device)  # [K, D]
        for images, labels in dataloader:
            images = images.to(device)
            emb = model(images)  # [B, D]
            # compute cosine similarity
            sims = emb @ anchor.t()  # [B, K]
            idx = sims.argmax(dim=1).cpu().numpy()
            preds = [anchor_keys[i] for i in idx]
            y_true.extend([anchor_keys[l] if isinstance(l, int) is False else l for l in labels.numpy()])
            # Note: labels might be ints (class indices). In calls we will map indices -> label str externally.
            y_pred.extend(preds)
    return y_true, y_pred


def compute_accuracy_num(y_true_idx, y_pred_idx):
    """
    y_true_idx, y_pred_idx: arrays of integer class indices aligned to same anchor_keys
    """
    y_true_idx = np.array(y_true_idx)
    y_pred_idx = np.array(y_pred_idx)
    return (y_true_idx == y_pred_idx).mean()
