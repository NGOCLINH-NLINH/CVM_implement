import sys
import os

from sklearn.linear_model import LogisticRegression
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import copy
import json
from pathlib import Path
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm

from models.resnet_cvm import DeterministicResNetCVM, create_lora_cvm_model
from utils import load_anchors, ReservoirBuffer, triplet_loss_emb, semantic_distance_loss, make_cifar100_tasks, \
    set_seed, triplet_loss_k_negs, triplet_loss_seen_negs, anchor_attraction_loss, image_side_prototype_spread_loss, \
    adaptive_margin_triplet_loss_k_negs, standard_margin_loss, kl_divergence_loss, variance_regularization_loss

replay_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


def evaluate_all_seen(model, test_full, seen_indices, anchors_tensor, anchor_keys, device):
    idxs = [i for i, (_, lbl) in enumerate(test_full) if lbl in seen_indices]
    if len(idxs) == 0:
        return 0.0
    loader = DataLoader(Subset(test_full, idxs), batch_size=128, shuffle=False, num_workers=2)
    model.eval()
    anchors_seen = anchors_tensor[seen_indices].to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            emb = model(images)
            sims = emb @ anchors_seen.t()
            preds = sims.argmax(dim=1).cpu().numpy()
            global_preds = [seen_indices[p] for p in preds]
            true = labels.numpy()
            correct += sum([1 for i in range(len(true)) if global_preds[i] == true[i]])
            total += len(true)
    return correct / total if total > 0 else 0.0


def evaluate_task_full_anchors(model, test_full, task_class_inds, anchors_tensor, anchor_keys, device):
    idxs = [i for i, (_, lbl) in enumerate(test_full) if lbl in task_class_inds]
    if len(idxs) == 0:
        return 0.0
    loader = DataLoader(Subset(test_full, idxs), batch_size=128, shuffle=False, num_workers=2)
    model.eval()
    anchors = anchors_tensor.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            emb = model(images)
            sims = emb @ anchors.t()
            preds = sims.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def zero_shot_eval(model, anchors_tensor, unseen_indices, test_full, device):
    if len(unseen_indices) == 0:
        return 0.0
    loader = DataLoader(Subset(test_full, [i for i, (_, l) in enumerate(test_full) if l in unseen_indices]),
                        batch_size=128, shuffle=False, num_workers=2)
    anchors_unseen = anchors_tensor[unseen_indices].to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            emb = model(images)
            sims = emb @ anchors_unseen.t()
            preds = sims.argmax(dim=1).cpu().numpy()
            global_preds = [unseen_indices[p] for p in preds]
            true = labels.numpy()
            correct += sum([1 for i in range(len(true)) if global_preds[i] == true[i]])
            total += len(true)
    return correct / total if total > 0 else 0.0


def linear_probe_all(model, train_full, test_full, seen_indices, device, out_dim):
    model.eval()
    train_idx = [i for i, (_, l) in enumerate(train_full) if l in seen_indices]
    test_idx = [i for i, (_, l) in enumerate(test_full) if l in seen_indices]

    if len(train_idx) == 0 or len(test_idx) == 0:
        return 0.0

    loader_tr = DataLoader(Subset(train_full, train_idx), batch_size=256, shuffle=False, num_workers=2)
    loader_te = DataLoader(Subset(test_full, test_idx), batch_size=256, shuffle=False, num_workers=2)

    X_tr, y_tr = [], []
    X_te, y_te = [], []

    with torch.no_grad():
        for images, labels in loader_tr:
            images = images.to(device)
            feats, _ = model(images)
            X_tr.append(feats.cpu().numpy())
            y_tr.append(labels.numpy())
        for images, labels in loader_te:
            images = images.to(device)
            feats, _ = model(images)
            X_te.append(feats.cpu().numpy())
            y_te.append(labels.numpy())

    X_tr = np.concatenate(X_tr, axis=0)
    y_tr = np.concatenate(y_tr, axis=0)
    X_te = np.concatenate(X_te, axis=0)
    y_te = np.concatenate(y_te, axis=0)

    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    return (preds == y_te).mean()


def compute_forgetting(eval_history):
    T = len(eval_history)
    forgetting = []

    for i in range(T - 1):
        acc_i_over_time = [eval_history[t][i] for t in range(i, T)]
        max_acc = max(acc_i_over_time[:-1])
        final_acc = acc_i_over_time[-1]
        forgetting.append(max_acc - final_acc)

    if len(forgetting) == 0: return 0.0, []
    return sum(forgetting) / len(forgetting), forgetting


def main(cfg):
    set_seed(cfg.get('seed', 1234))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print(f"\n{'=' * 50}")
    print(f"EXPERIMENT: {cfg['exp_name']} | SEED: {cfg['seed']}")
    print(
        f"PARAMS: Beta={cfg['beta']}, Spread={cfg['spread_lambda']}, Adaptive={cfg['adaptive_margin']}, Margin={cfg['margin']}")
    print(f"DEVICE: {device}")
    print(f"{'=' * 50}\n")

    tasks, class_names = make_cifar100_tasks(cfg['num_tasks'], cfg['batch_size'], augment=True)
    train_full = datasets.CIFAR100(root="data", train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]))
    test_full = datasets.CIFAR100(root="data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]))

    if not os.path.exists(cfg['anchors_path']):
        if os.path.exists(os.path.join('..', cfg['anchors_path'])):
            cfg['anchors_path'] = os.path.join('..', cfg['anchors_path'])

    anchor_keys, anchors_tensor = load_anchors(cfg['anchors_path'], device=device)
    print("Loaded anchors:", len(anchor_keys))

    model = create_lora_cvm_model(out_dim=cfg['out_dim'], pretrained=cfg.get('pretrained_backbone', True), lora_rank=16).to(device)
    prev_model = None

    buffer = ReservoirBuffer(capacity=cfg['memory_size'])

    seen_inds = []

    seen_acc_history = []
    zero_shot_history = []
    linear_probe_history = []
    eval_history = []

    Path(cfg['checkpoints_dir']).mkdir(parents=True, exist_ok=True)

    for t, (train_loader, test_loader, class_inds) in enumerate(tasks):
        print(f"\n=== Training Task {t} (Classes: {min(class_inds)}-{max(class_inds)}) ===")
        cur_inds = class_inds
        old_inds = [i for i in seen_inds]
        seen_inds += cur_inds

        fc_params = [p for n, p in model.named_parameters() if 'fc' in n and p.requires_grad]
        lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]

        fc_lr = cfg['lr'] if t == 0 else cfg['lr'] * 0.1

        optimizer = optim.SGD([
            {'params': lora_params, 'lr': cfg['lr']},
            {'params': fc_params, 'lr': fc_lr}
        ], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.get('milestones', [50, 75]), gamma=0.1)

        total_steps = cfg['epochs_per_task'] * len(train_loader)
        pbar = tqdm(total=total_steps, desc=f"Task {t}", dynamic_ncols=True)

        for epoch in range(cfg['epochs_per_task']):
            model.train()

            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            for images, raw_images, labels in train_loader:
                images_cuda = images.to(device)
                labels_cuda = labels.to(device)

                # 1. Xử lý ảnh mới (Chỉ tính Margin Loss)
                mu = model(images_cuda)
                Lm = standard_margin_loss(mu, labels_cuda, anchors_tensor, seen_inds, margin=cfg['margin'])

                if cfg['spread_lambda'] > 0:
                    L_spread = image_side_prototype_spread_loss(mu, labels_cuda, anchors_tensor, seen_inds,
                                                                delta=cfg['spread_delta'])
                else:
                    L_spread = torch.tensor(0.0, device=device)

                loss = Lm + cfg['spread_lambda'] * L_spread

                if t > 0 and len(buffer) > 0 and cfg['replay_batch'] > 0 and cfg.get('replay_on', True):
                    buf_imgs_raw, buf_labels = buffer.sample(cfg['replay_batch'])
                    if buf_imgs_raw is not None:
                        buf_imgs_raw = buf_imgs_raw.to(device)
                        buf_labels = buf_labels.to(device)
                        buf_imgs_aug = replay_transform(buf_imgs_raw)

                        mu_buf = model(buf_imgs_aug)
                        Lm_buf = standard_margin_loss(mu_buf, buf_labels, anchors_tensor, seen_inds,
                                                      margin=cfg['margin'])

                        if cfg['spread_lambda'] > 0:
                            L_spread_buf = image_side_prototype_spread_loss(mu_buf, buf_labels, anchors_tensor,
                                                                            seen_inds, delta=cfg['spread_delta'])
                        else:
                            L_spread_buf = torch.tensor(0.0, device=device)

                        L_distill = torch.tensor(0.0, device=device)
                        if prev_model is not None and cfg['beta'] > 0:
                            with torch.no_grad():
                                mu_prev_buf = prev_model(buf_imgs_aug)

                            L_distill = (1.0 - (mu_buf * mu_prev_buf).sum(dim=1)).mean()

                        loss += cfg['replay_lambda'] * (
                                    Lm_buf + cfg['beta'] * L_distill + cfg['spread_lambda'] * L_spread_buf)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                buffer.add_batch(raw_images, labels)

                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.3f}"})

            scheduler.step()
        pbar.close()

        prev_model = copy.deepcopy(model).eval().to(device)
        for param in prev_model.parameters():
            param.requires_grad = False

        print(f"--- Evaluation after Task {t} ---")

        acc_all_seen = evaluate_all_seen(model, test_full, seen_inds, anchors_tensor, anchor_keys, device)
        seen_acc_history.append(acc_all_seen)
        print(f"Acc on all seen classes after task {t}: {acc_all_seen:.4f}")

        # lp_acc = linear_probe_all(model, train_full, test_full, seen_inds, device, cfg['out_dim'])
        # linear_probe_history.append(lp_acc)
        # print(f"Linear probe acc on seen classes after task {t}: {lp_acc:.4f}")

        unseen_inds = [i for i in range(len(anchor_keys)) if i not in seen_inds]
        zs = zero_shot_eval(model, anchors_tensor, unseen_inds, test_full, device)
        zero_shot_history.append(zs)
        print(f"Zero-shot acc on unseen classes after task {t}: {zs:.4f}")

        per_task_accs = []
        for i_task, (_, _, t_classes) in enumerate(tasks):
            if i_task > t:
                per_task_accs.append(None)
            else:
                acc_old_task = evaluate_task_full_anchors(model, test_full, t_classes, anchors_tensor, anchor_keys,
                                                          device)
                per_task_accs.append(acc_old_task)
                print(f"  -> Accuracy on Task {i_task}: {acc_old_task:.4f}")
        eval_history.append(per_task_accs)

    fw_score, _ = compute_forgetting(eval_history)
    avg_acc_final = np.mean(seen_acc_history)

    print(f"\n--- FINAL RESULTS (Seed {cfg['seed']}) ---")
    print(f"Avg Accuracy: {avg_acc_final:.4f}")
    print(f"Forgetting: {fw_score:.4f}")

    results = {
        "exp_name": cfg['exp_name'],
        "seed": cfg['seed'],
        "config": {
            "beta": cfg['beta'],
            "spread_lambda": cfg['spread_lambda'],
            "margin": cfg['margin'],
            "adaptive_margin": cfg['adaptive_margin'],
            "memory_size": cfg['memory_size']
        },
        "seen_acc_history": [float(x) for x in seen_acc_history],
        "linear_probe_history": [float(x) for x in linear_probe_history],
        "zero_shot_history": [float(x) for x in zero_shot_history],

        "avg_acc_over_time": float(avg_acc_final),
        "forgetting": float(fw_score),
        "eval_matrix": [[float(x) if x is not None else None for x in row] for row in eval_history]
    }

    log_filename = f"log_{cfg['exp_name']}_seed{cfg['seed']}.json"
    log_path = os.path.join(cfg['checkpoints_dir'], log_filename)

    with open(log_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved detailed logs to: {log_path}")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    mode_name = "ORIGINAL CVM" if cfg.get('original_cvm', False) else "PROBABILISTIC ACVM"
    print(f"\nSTARTING EXPERIMENT MODE: {mode_name}")

    main(cfg)
