# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import pickle
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os


def load_anchors(path, device='cpu'):
    with open(path, 'rb') as f:
        anchors = pickle.load(f)
    keys = list(anchors.keys())
    mat = np.stack([anchors[k] for k in keys])
    tensor = torch.from_numpy(mat).float().to(device)
    tensor = F.normalize(tensor, p=2, dim=1)
    return keys, tensor


class ReservoirBuffer:
    def __init__(self, capacity=2000, seed=42):
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
    emb = F.normalize(emb, p=2, dim=1)
    emb_prev = F.normalize(emb_prev, p=2, dim=1)
    anchor = F.normalize(old_anchor_matrix, p=2, dim=1)
    cos_t = emb @ anchor.t()
    cos_prev = emb_prev @ anchor.t()
    d_t = 1.0 - cos_t
    d_prev = 1.0 - cos_prev
    return F.mse_loss(d_t, d_prev)


def nearest_anchor_predict(model, dataloader, anchors_tensor, anchor_keys, device='cuda'):
    model.eval()
    y_true = []
    y_pred = []
    idx_map = {k: i for i, k in enumerate(anchor_keys)}
    with torch.no_grad():
        anchors = anchors_tensor.to(device)
        for images, labels in dataloader:
            images = images.to(device)
            emb = model(images)
            sims = emb @ anchors.t()
            idx = sims.argmax(dim=1).cpu().numpy()
            preds = [anchor_keys[i] for i in idx]
            y_pred.extend(preds)
            # labels are integer indices global; convert to label string using anchor_keys
            y_true.extend([anchor_keys[l] for l in labels.numpy()])
    # compute accuracy by mapping to indices
    y_true_idx = [idx_map[y] for y in y_true]
    y_pred_idx = [idx_map[y] for y in y_pred]
    acc = accuracy_score(y_true_idx, y_pred_idx)
    return acc


def compute_forgetting(history_accs):
    T = len(history_accs)
    final = history_accs[-1]
    forgetting = 0.0
    for i in range(T - 1):
        acc_i_at_train = history_accs[i][i] if len(history_accs[i]) > i else 0.0
        acc_i_final = final[i] if i < len(final) else 0.0
        forgetting += (acc_i_at_train - acc_i_final)
    return forgetting / max(1, T - 1)


def linear_probing_score(model, train_features, train_labels, test_features, test_labels):
    # use sklearn logistic regression on L2-normalized features (numpy)
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='saga')
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)
    return accuracy_score(test_labels, preds)
