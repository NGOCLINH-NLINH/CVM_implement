import os

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


class ClassBalancedRandomBuffer:
    def __init__(self, capacity=500, seed=1234):
        self.capacity = capacity
        self.memory = {}
        self.rng = random.Random(seed)

    def update_buffer(self, dataloader, class_inds):
        all_imgs = {c: [] for c in class_inds}
        for _, raw_images, labels in dataloader:
            for i in range(len(labels)):
                lbl = int(labels[i].item())
                if lbl in class_inds:
                    all_imgs[lbl].append((raw_images[i].clone(), lbl))
        for c in class_inds:
            self.rng.shuffle(all_imgs[c])
            self.memory[c] = all_imgs[c]
        num_seen_classes = len(self.memory)
        m_per_class = self.capacity // num_seen_classes

        for c in self.memory.keys():
            self.memory[c] = self.memory[c][:m_per_class]

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return None, None

        all_items = []
        for c_items in self.memory.values():
            all_items.extend(c_items)

        if len(all_items) == 0:
            return None, None

        batch = random.sample(all_items, k=min(batch_size, len(all_items)))
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return imgs, labels

    def __len__(self):
        return sum([len(v) for v in self.memory.values()])


def get_active_mean(loss_slice):
    active = loss_slice[loss_slice > 0]
    if len(active) > 0:
        return active.mean()
    else:
        return loss_slice.sum() * 0.0
        # return torch.tensor(0.0, device=loss_slice.device, requires_grad=True)


def triplet_loss_emb(emb, pos_emb, neg_emb, margin=0.1):
    d_pos = 1.0 - F.cosine_similarity(emb, pos_emb, dim=1)
    d_neg = 1.0 - F.cosine_similarity(emb, neg_emb, dim=1)
    loss_tensor = F.relu(d_pos - d_neg + margin)
    return loss_tensor

    # loss = F.relu(d_pos - d_neg + margin).mean()
    # return loss

    # loss_tensor = F.relu(d_pos - d_neg + margin)
    # active_losses = loss_tensor[loss_tensor > 0]
    # if len(active_losses) > 0:
    #     return active_losses.mean()
    # else:
    #     return torch.tensor(0.0, device=emb.device, requires_grad=True)


def triplet_loss_hardest_neg(emb, pos_emb, labels, anchors_tensor, seen_indices, margin=0.1):
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
    hardest_losses, _ = loss_mat.max(dim=1)

    return hardest_losses.mean()


def semantic_distance_loss(emb, emb_prev, old_anchor_matrix, reduction='mean'):
    if old_anchor_matrix is None or old_anchor_matrix.shape[0] == 0:
        return torch.tensor(0.0, device=emb.device)

    cos_t = emb @ old_anchor_matrix.t()
    cos_prev = emb_prev @ old_anchor_matrix.t()
    d_t = 1.0 - cos_t
    d_prev = 1.0 - cos_prev

    # loss_matrix = torch.abs(d_t - d_prev)
    # loss_per_sample = loss_matrix.sum(dim=1)

    # loss_matrix = F.mse_loss(d_t, d_prev, reduction='none')
    # loss_per_sample = loss_matrix.mean(dim=1)

    loss_matrix = (d_t - d_prev) ** 2
    loss_per_sample = loss_matrix.sum(dim=1)
    if reduction == 'none':
        return loss_per_sample
    return loss_per_sample.mean()


def make_tinyimagenet_tasks(num_tasks, batch_size, augment=True, root="data/tiny-imagenet-200"):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if augment:
        transform_train_aug = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ])
    else:
        transform_train_aug = transforms.Compose([
            transforms.ToTensor(),
            norm
        ])

    transform_raw = transforms.ToTensor()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    train_full_raw = datasets.ImageFolder(os.path.join(root, 'train'), transform=None)
    test_full = datasets.ImageFolder(os.path.join(root, 'val/images_formatted'), transform=transform_test)

    train_dataset_wrapper = DualTransformDataset(train_full_raw, transform_train_aug, transform_raw)

    classes = train_full_raw.classes
    num_classes = len(classes)
    per_task = num_classes // num_tasks
    tasks = []

    all_targets = np.array(train_full_raw.targets)
    all_test_targets = np.array(test_full.targets)

    for t in range(num_tasks):
        start = t * per_task
        end = start + per_task if t < num_tasks - 1 else num_classes

        train_idx = np.where((all_targets >= start) & (all_targets < end))[0]
        test_idx = np.where((all_test_targets >= start) & (all_test_targets < end))[0]

        train_subset = Subset(train_dataset_wrapper, train_idx)
        test_subset = Subset(test_full, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        tasks.append((train_loader, test_loader, list(range(start, end))))

    return tasks, classes


def make_cifar100_tasks(num_tasks, batch_size, augment=True):
    if augment:
        transform_train_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        transform_train_aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    transform_raw = transforms.ToTensor()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_full_raw = datasets.CIFAR100(root="data", train=True, download=True, transform=None)
    test_full = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)

    train_dataset_wrapper = DualTransformDataset(train_full_raw, transform_train_aug, transform_raw)

    classes = train_full_raw.classes
    num_classes = len(classes)
    per_task = num_classes // num_tasks
    tasks = []

    all_targets = np.array(train_full_raw.targets)
    all_test_targets = np.array(test_full.targets)

    for t in range(num_tasks):
        start = t * per_task
        end = start + per_task if t < num_tasks - 1 else num_classes

        train_idx = np.where((all_targets >= start) & (all_targets < end))[0]
        test_idx = np.where((all_test_targets >= start) & (all_test_targets < end))[0]

        train_subset = Subset(train_dataset_wrapper, train_idx)
        test_subset = Subset(test_full, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        tasks.append((train_loader, test_loader, list(range(start, end))))
    return tasks, classes


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def triplet_loss_k_negs(emb, pos_emb, neg_embs, margin=0.1, reduction='none'):
    cos_pos = (emb * pos_emb).sum(dim=1)
    d_pos = 1.0 - cos_pos

    cos_neg = (emb.unsqueeze(1) * neg_embs).sum(dim=2)
    d_neg = 1.0 - cos_neg
    loss_matrix = torch.clamp(d_pos.unsqueeze(1) - d_neg + margin, min=0.0)
    # loss_per_sample = loss_mat.sum(dim=1)
    # return loss_per_sample.mean()
    # return loss.mean()
    loss_per_sample = loss_matrix.mean(dim=1)
    if reduction == 'none':
        return loss_per_sample
    return loss_per_sample.mean()


def triplet_loss_seen_negs(emb, pos_emb, labels, anchors_tensor, seen_indices, margin=0.1, reduction='none'):
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
        loss_per_sample = torch.zeros(emb.size(0), device=device, requires_grad=True)
    else:
        loss_per_sample = loss_mat.sum(dim=1) / num_negs
    if reduction == 'none':
        return loss_per_sample
    return loss_per_sample.mean()

    # if num_negs <= 0:
    #     return torch.tensor(0.0, device=device, requires_grad=True)
    #
    # return loss_mat.sum() / (emb.size(0) * num_negs)


def adaptive_margin_triplet_loss_seen_negs(emb, pos_emb, labels, anchors_tensor, seen_indices, base_margin=0.1, reduction='none'):
    device = emb.device
    anchors_seen = anchors_tensor[seen_indices].to(device)

    cos_pos = (emb * pos_emb).sum(dim=1)
    d_pos = 1.0 - cos_pos

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
        loss_per_sample = torch.zeros(emb.size(0), device=device, requires_grad=True)
    else:
        loss_per_sample = loss_mat.sum(dim=1) / num_negs

    if reduction == 'none':
        return loss_per_sample
    return loss_per_sample.mean()


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


def adaptive_margin_triplet_loss_k_negs(emb, pos, neg_k, base_margin=0.1, reduction="none", scale_factor=0.2):
    sim_pos = (emb * pos).sum(dim=1, keepdim=True)
    sim_neg = (emb.unsqueeze(1) * neg_k).sum(dim=2)
    anchor_sim = (pos.unsqueeze(1) * neg_k).sum(dim=2)

    # adaptive_margin = base_margin * (1.0 - anchor_sim).clamp(min=0.0)
    # loss_matrix = F.relu(sim_neg - sim_pos + adaptive_margin)
    # loss_per_sample = loss_matrix.mean(dim=1)

    scale_factor = scale_factor
    semantic_dist = (1.0 - anchor_sim).clamp(min=0.0)
    adaptive_margin = base_margin + (scale_factor * semantic_dist)
    loss_matrix = F.relu(sim_neg - sim_pos + adaptive_margin)
    loss_per_sample = loss_matrix.mean(dim=1)

    return loss_per_sample

def get_semantic_mask(batch_anchors, hash_mat, sparsity=0.4):
    h = torch.matmul(batch_anchors, hash_mat)
    k = max(1, int(512 * sparsity))
    _, idx = torch.topk(h, k, dim=-1)
    mask = torch.zeros_like(h)
    mask.scatter_(-1, idx, 1.0)

    batch_mask, _ = torch.max(mask, dim=0)
    return batch_mask


def cross_modal_mixup_loss(model, buf_imgs, buf_labels, anchors_tensor, alpha=0.4):
    device = buf_imgs.device
    batch_size = buf_imgs.size(0)

    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    index = torch.randperm(batch_size).to(device)
    mixed_imgs = lam * buf_imgs + (1 - lam) * buf_imgs[index]

    anchors_A = anchors_tensor[buf_labels]
    anchors_B = anchors_tensor[buf_labels[index]]

    mixed_anchors = lam * anchors_A + (1 - lam) * anchors_B
    mixed_anchors = F.normalize(mixed_anchors, p=2, dim=1)
    mixed_emb = model(mixed_imgs)
    cos_sim = (mixed_emb * mixed_anchors).sum(dim=1)
    loss = (1.0 - cos_sim).mean()

    return loss


class HerdingBuffer:
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.memory = {}
        self.seen_classes = []

    def icarl_herding(self, features, images, labels, nb_protos_cl):
        D = features.t()
        mu = D.mean(dim=1)

        w_t = mu.clone()
        selected_indices = []

        for _ in range(nb_protos_cl):
            tmp_t = torch.matmul(w_t, D)

            if len(selected_indices) > 0:
                tmp_t[selected_indices] = -float('inf')

            ind_max = torch.argmax(tmp_t).item()

            w_t = w_t + mu - D[:, ind_max]
            selected_indices.append(ind_max)

        exemplars = [(images[i].clone(), int(labels[i].item())) for i in selected_indices]
        return exemplars

    def update_buffer(self, model, dataloader, class_inds, device):

        model.eval()

        all_features = {c: [] for c in class_inds}
        all_images = {c: [] for c in class_inds}
        all_labels = {c: [] for c in class_inds}

        val_norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

        with torch.no_grad():
            for imgs, raw_imgs, lbls in dataloader:
                clean_imgs = torch.stack([val_norm(img) for img in raw_imgs]).to(device)
                feats = model(clean_imgs).cpu()

                for i in range(len(lbls)):
                    lbl = int(lbls[i].item())
                    if lbl in class_inds:
                        all_features[lbl].append(feats[i].unsqueeze(0))
                        all_images[lbl].append(raw_imgs[i])
                        all_labels[lbl].append(lbls[i])

        self.seen_classes.extend(class_inds)

        m = self.capacity // len(self.seen_classes)

        for c in self.memory.keys():
            self.memory[c] = self.memory[c][:m]

        for c in class_inds:
            feats_c = torch.cat(all_features[c], dim=0)
            imgs_c = torch.stack(all_images[c], dim=0)
            lbls_c = torch.tensor(all_labels[c], dtype=torch.long)

            exemplars_c = self.icarl_herding(feats_c, imgs_c, lbls_c, nb_protos_cl=m)
            self.memory[c] = exemplars_c

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return None, None

        all_exemplars = []
        for exemplars in self.memory.values():
            all_exemplars.extend(exemplars)

        if len(all_exemplars) == 0:
            return None, None

        batch = random.sample(all_exemplars, k=min(batch_size, len(all_exemplars)))
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return imgs, labels

    def __len__(self):
        return sum([len(v) for v in self.memory.values()])


def make_aircraft_tasks(num_tasks, batch_size, augment=True):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if augment:
        transform_train_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ])
    else:
        transform_train_aug = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm
        ])

    transform_raw = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm
    ])

    from torchvision.datasets import FGVCAircraft
    train_full_raw = FGVCAircraft(root="data", split='trainval', download=True, transform=None)
    test_full = FGVCAircraft(root="data", split='test', download=True, transform=transform_test)

    train_dataset_wrapper = DualTransformDataset(train_full_raw, transform_train_aug, transform_raw)

    classes = train_full_raw.classes
    num_classes = len(classes)
    per_task = num_classes // num_tasks
    tasks = []

    all_targets = np.array(train_full_raw._labels)
    all_test_targets = np.array(test_full._labels)

    for t in range(num_tasks):
        start = t * per_task
        end = start + per_task if t < num_tasks - 1 else num_classes

        train_idx = np.where((all_targets >= start) & (all_targets < end))[0]
        test_idx = np.where((all_test_targets >= start) & (all_test_targets < end))[0]

        train_subset = Subset(train_dataset_wrapper, train_idx)
        test_subset = Subset(test_full, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        tasks.append((train_loader, test_loader, list(range(start, end))))

    return tasks, classes


import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


def save_features_for_tsne(model, test_full, seen_indices, anchors_tensor, task_id, device, save_path):
    model.eval()

    idxs = [i for i, (_, lbl) in enumerate(test_full) if lbl in seen_indices]
    if len(idxs) == 0:
        return

    loader = DataLoader(Subset(test_full, idxs), batch_size=256, shuffle=False, num_workers=2)

    all_embs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            emb = model(images)
            all_embs.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    seen_anchors = anchors_tensor[seen_indices].cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez_compressed(
        save_path,
        embs=all_embs,
        labels=all_labels,
        anchors=seen_anchors,
        seen_indices=np.array(seen_indices)
    )
    print(f"[+] Saved features for t-SNE at {save_path}")


def make_gtsrb_tasks(num_tasks, batch_size, augment=True):
    norm = transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))

    if augment:
        transform_train_aug = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            norm
        ])
    else:
        transform_train_aug = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            norm
        ])

    transform_raw = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        norm
    ])

    from torchvision.datasets import GTSRB
    train_full_raw = GTSRB(root="data", split='train', download=True, transform=None)
    test_full = GTSRB(root="data", split='test', download=True, transform=transform_test)

    train_dataset_wrapper = DualTransformDataset(train_full_raw, transform_train_aug, transform_raw)

    num_classes = 43
    classes = [str(i) for i in range(num_classes)]
    per_task = num_classes // num_tasks
    tasks = []

    all_targets = np.array([lbl for _, lbl in train_full_raw._samples])
    all_test_targets = np.array([lbl for _, lbl in test_full._samples])

    for t in range(num_tasks):
        start = t * per_task
        end = start + per_task if t < num_tasks - 1 else num_classes

        train_idx = np.where((all_targets >= start) & (all_targets < end))[0]
        test_idx = np.where((all_test_targets >= start) & (all_test_targets < end))[0]

        train_subset = Subset(train_dataset_wrapper, train_idx)
        test_subset = Subset(test_full, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        tasks.append((train_loader, test_loader, list(range(start, end))))

    return tasks, classes