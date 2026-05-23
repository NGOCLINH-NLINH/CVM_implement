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
from tqdm import tqdm

from models.resnet_cvm import ResNetCVM
from utils.utils import load_anchors, ReservoirBuffer, triplet_loss_emb, semantic_distance_loss, make_cifar100_tasks, \
    set_seed, triplet_loss_k_negs, triplet_loss_seen_negs, image_side_prototype_spread_loss, \
    adaptive_margin_triplet_loss_k_negs, get_semantic_mask, adaptive_margin_triplet_loss_seen_negs, \
    triplet_loss_hardest_neg, HerdingBuffer, get_active_mean, ClassBalancedRandomBuffer, cross_modal_mixup_loss

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


def evaluate_task_seen_anchors(model, test_full, task_class_inds, seen_indices, anchors_tensor, device):
    idxs = [i for i, (_, lbl) in enumerate(test_full) if lbl in task_class_inds]
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


def linear_probe_unseen(model, train_full, test_full, unseen_indices, device):
    model.eval()
    train_idx = [i for i, (_, l) in enumerate(train_full) if l in unseen_indices]
    test_idx = [i for i, (_, l) in enumerate(test_full) if l in unseen_indices]

    if len(train_idx) == 0 or len(test_idx) == 0:
        return 0.0

    loader_tr = DataLoader(Subset(train_full, train_idx), batch_size=256, shuffle=False, num_workers=2)
    loader_te = DataLoader(Subset(test_full, test_idx), batch_size=256, shuffle=False, num_workers=2)

    X_tr, y_tr = [], []
    X_te, y_te = [], []
    with torch.no_grad():
        for images, labels in loader_tr:
            images = images.to(device)
            feats = model(images).cpu().numpy()
            X_tr.append(feats)
            y_tr.append(labels.numpy())
        for images, labels in loader_te:
            images = images.to(device)
            feats = model(images).cpu().numpy()
            X_te.append(feats)
            y_te.append(labels.numpy())

    X_tr = np.concatenate(X_tr, axis=0)
    y_tr = np.concatenate(y_tr, axis=0)
    X_te = np.concatenate(X_te, axis=0)
    y_te = np.concatenate(y_te, axis=0)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
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

    model = ResNetCVM(out_dim=cfg['out_dim'], pretrained=cfg.get('pretrained_backbone', False)).to(device)
    prev_model = None

    buffer = ReservoirBuffer(capacity=cfg['memory_size'], seed=cfg.get('seed', 1234))

    seen_inds = []

    seen_acc_history = []
    zero_shot_history = []
    linear_probe_history = []
    eval_history = []

    Path(cfg['checkpoints_dir']).mkdir(parents=True, exist_ok=True)

    use_amp = cfg.get('use_amp', True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for t, (train_loader, test_loader, class_inds) in enumerate(tasks):
        print(f"\n=== Training Task {t} (Classes: {min(class_inds)}-{max(class_inds)}) ===")
        cur_inds = class_inds
        old_inds = [i for i in seen_inds]
        seen_inds += cur_inds

        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.get('milestones', [50, 75]), gamma=0.1)
        total_steps = cfg['epochs_per_task'] * len(train_loader)
        pbar = tqdm(total=total_steps, desc=f"Task {t}", dynamic_ncols=True)

        for epoch in range(cfg['epochs_per_task']):
            model.train()

            for images, raw_images, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)

                images_cuda = images.to(device)
                labels_cuda = labels.to(device)
                if prev_model is not None and len(old_inds) > 0:
                    old_anchor_mat = anchors_tensor[old_inds].to(device)
                else:
                    old_anchor_mat = None

                has_buffer = (t > 0 and len(buffer) > 0 and cfg['replay_batch'] > 0 and cfg.get('replay_on', True))
                if has_buffer:
                    buf_imgs_raw, buf_labels_cpu = buffer.sample(cfg['replay_batch'])
                    if buf_imgs_raw is not None:
                        buf_imgs_raw = buf_imgs_raw.to(device)
                        buf_labels = buf_labels_cpu.to(device)
                        buf_imgs_aug = torch.stack([replay_transform(img) for img in buf_imgs_raw])

                        combined_images = torch.cat([images_cuda, buf_imgs_aug], dim=0)
                        combined_labels = torch.cat([labels_cuda, buf_labels], dim=0)
                    else:
                        combined_images = images_cuda
                        combined_labels = labels_cuda
                else:
                    combined_images = images_cuda
                    combined_labels = labels_cuda

                with torch.amp.autocast('cuda', enabled=use_amp):
                    emb_combined = model(combined_images)
                    pos_combined = anchors_tensor[combined_labels].to(device)

                    if cfg.get('original_cvm', False):
                        neg_idx_list = []
                        for lbl in combined_labels.cpu().numpy():
                            choices = [c for c in seen_inds if c != lbl]
                            neg_idx = random.choice(choices) if len(choices) > 0 else lbl
                            neg_idx_list.append(neg_idx)

                        neg_tensor = anchors_tensor[torch.tensor(neg_idx_list, dtype=torch.long, device=device)]
                        Lm_unreduced = triplet_loss_emb(emb_combined, pos_combined, neg_tensor, margin=cfg['margin'])

                    else:
                        if cfg.get('use_all_seen_negs', False):
                            if cfg.get('adaptive_margin', False):
                                Lm_unreduced = adaptive_margin_triplet_loss_seen_negs(
                                    emb=emb_combined, pos_emb=pos_combined, labels=combined_labels,
                                    anchors_tensor=anchors_tensor, seen_indices=seen_inds,
                                    base_margin=cfg['margin'], reduction='none'
                                )
                            else:
                                Lm_unreduced = triplet_loss_seen_negs(
                                    emb=emb_combined, pos_emb=pos_combined, labels=combined_labels,
                                    anchors_tensor=anchors_tensor, seen_indices=seen_inds,
                                    margin=cfg['margin'], reduction='none'
                                )
                        else:
                            K_val = cfg.get('k_negs', 9)
                            neg_idx_list = []
                            for lbl in combined_labels.cpu().numpy():
                                choices = [c for c in seen_inds if c != lbl]
                                if len(choices) >= K_val:
                                    negs = random.sample(choices, k=K_val)
                                else:
                                    negs = random.choices(choices, k=K_val) if len(choices) > 0 else [lbl] * K_val
                                neg_idx_list.append(negs)

                            neg_k_tensor = anchors_tensor[torch.tensor(neg_idx_list, dtype=torch.long, device=device)]

                            if cfg.get('adaptive_margin', False):
                                Lm_unreduced = adaptive_margin_triplet_loss_k_negs(
                                    emb=emb_combined, pos=pos_combined, neg_k=neg_k_tensor,
                                    base_margin=cfg['margin'], reduction='none',
                                    scale_factor=cfg.get('adaptive_margin_scale', 0.2)
                                )
                            else:
                                Lm_unreduced = triplet_loss_k_negs(
                                    emb=emb_combined, pos_emb=pos_combined, neg_embs=neg_k_tensor,
                                    margin=cfg['margin'], reduction='none'
                                )

                    num_new = images_cuda.size(0)
                    use_active_mean = cfg.get('use_active_mean', True)
                    if use_active_mean:
                        Lm_new = get_active_mean(Lm_unreduced[:num_new])
                    else:
                        Lm_new = Lm_unreduced[:num_new].mean()

                    if has_buffer:
                        if use_active_mean:
                            Lm_buf = get_active_mean(Lm_unreduced[num_new:])
                        else:
                            Lm_buf = Lm_unreduced[num_new:].mean()
                        replay_weight = cfg.get('replay_lambda', 2.0)
                        Lm = Lm_new + replay_weight * Lm_buf
                    else:
                        Lm = Lm_new

                    Ld = torch.tensor(0.0, device=device)
                    if old_anchor_mat is not None and cfg.get('beta', 0.0) > 0:
                        num_new = images_cuda.size(0)
                        ld_mode = cfg.get('Ld_mode', 'buf')
                        replay_weight = cfg.get('replay_lambda', 2.0)

                        if ld_mode == 'buf' and has_buffer:
                            emb_buf_current = emb_combined[num_new:]
                            with torch.no_grad():
                                emb_prev_buf = prev_model(buf_imgs_aug)
                            Ld_unreduced = semantic_distance_loss(emb_buf_current, emb_prev_buf, old_anchor_mat,
                                                                  reduction='none')
                            Ld = Ld_unreduced.mean() * replay_weight

                        elif ld_mode == 'new':
                            emb_new_current = emb_combined[:num_new]
                            with torch.no_grad():
                                emb_prev_new = prev_model(images_cuda)

                            Ld_unreduced = semantic_distance_loss(emb_new_current, emb_prev_new, old_anchor_mat,
                                                                  reduction='none')
                            Ld = Ld_unreduced.mean()

                        elif ld_mode == 'both':
                            with torch.no_grad():
                                emb_prev_combined = prev_model(combined_images)
                            Ld_unreduced = semantic_distance_loss(emb_combined, emb_prev_combined, old_anchor_mat,
                                                                  reduction='none')
                            Ld_new = Ld_unreduced[:num_new].mean()
                            if has_buffer:
                                Ld_buf = Ld_unreduced[num_new:].mean()
                                Ld = Ld_new + replay_weight * Ld_buf
                            else:
                                Ld = Ld_new

                    loss = Lm + cfg['beta'] * Ld

                    use_cmm = cfg.get('use_cmm', False)
                    if has_buffer and use_cmm:
                        lambda_cmm = cfg.get('lambda_cmm', 1.0)
                        alpha_cmm = cfg.get('alpha_cmm', 0.4)
                        L_cmm = cross_modal_mixup_loss(
                            model=model,
                            buf_imgs=buf_imgs_aug,
                            buf_labels=buf_labels,
                            anchors_tensor=anchors_tensor,
                            alpha=alpha_cmm
                        )
                        loss = loss + lambda_cmm * L_cmm

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get('max_norm', 5.0))

                scaler.step(optimizer)
                scaler.update()

                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.3f}"})

            scheduler.step()
        pbar.close()

        print("Updating Replay Buffer...")
        for _, raw_images, labels in train_loader:
            buffer.add_batch(raw_images, labels)
        # buffer.update_buffer(train_loader, cur_inds)

        prev_model = copy.deepcopy(model).eval().to(device)

        print(f"--- Evaluation after Task {t} ---")

        acc_all_seen = evaluate_all_seen(model, test_full, seen_inds, anchors_tensor, anchor_keys, device)
        seen_acc_history.append(acc_all_seen)
        print(f"Acc on all seen classes after task {t}: {acc_all_seen:.4f}")

        unseen_inds = [i for i in range(len(anchor_keys)) if i not in seen_inds]
        if len(unseen_inds) > 0:
            zs = zero_shot_eval(model, anchors_tensor, unseen_inds, test_full, device)
            zero_shot_history.append(zs)
            print(f"Zero-shot acc on unseen classes after task {t}: {zs:.4f}")

            lp_acc = linear_probe_unseen(model, train_full, test_full, unseen_inds, device)
            linear_probe_history.append(lp_acc)
            print(f"Linear probe acc on unseen classes after task {t}: {lp_acc:.4f}")
        else:
            print(f"No unseen classes left to evaluate Forward Transfer after task {t}.")

        per_task_accs = []
        for i_task, (_, _, t_classes) in enumerate(tasks):
            if i_task > t:
                per_task_accs.append(None)
            else:
                acc_old_task = evaluate_task_seen_anchors(model, test_full, t_classes, seen_inds, anchors_tensor,
                                                          device)
                per_task_accs.append(acc_old_task)
                print(f"  -> Acc on Task {i_task} (Classes {min(t_classes)}-{max(t_classes)}): {acc_old_task:.4f}")
        eval_history.append(per_task_accs)

    forgetting_score, _ = compute_forgetting(eval_history)
    final_task_accs = eval_history[-1]
    avg_acc_final = np.mean([acc for acc in final_task_accs if acc is not None])

    print(f"\n--- FINAL RESULTS (Seed {cfg['seed']}) ---")
    print(f"Avg Accuracy: {avg_acc_final:.4f}")
    print(f"Forgetting: {forgetting_score:.4f}")

    if len(linear_probe_history) > 0:
        fw_score = np.mean(linear_probe_history)
        avg_zero_shot = np.mean(zero_shot_history)
    else:
        fw_score = 0.0
        avg_zero_shot = 0.0

    print(f"FW_Score (Linear Probe on Unseen): {fw_score:.4f}")
    print(f"Avg Zero-Shot on Unseen: {avg_zero_shot:.4f}")

    results = {
        "exp_name": cfg['exp_name'],
        "seed": cfg['seed'],
        "config": {
            "beta": cfg['beta'],
            "margin": cfg['margin'],
            "adaptive_margin": cfg.get('adaptive_margin', False),
            "memory_size": cfg['memory_size'],
            "use_all_seen_negs": cfg.get('use_all_seen_negs', False),
            "use_active_mean": cfg.get('use_active_mean', True),
            "use_cmm": cfg.get('use_cmm', False),
        },
        "seen_acc_history": [float(x) for x in seen_acc_history],
        "linear_probe_history": [float(x) for x in linear_probe_history],
        "zero_shot_history": [float(x) for x in zero_shot_history],

        "avg_acc_over_time": float(avg_acc_final),
        "forgetting_score": float(forgetting_score),
        "fw_score": float(fw_score),

        "eval_matrix": [[float(x) if x is not None else None for x in row] for row in eval_history]
    }

    log_filename = f"log_{cfg['exp_name']}_seed{cfg['seed']}.json"
    log_path = os.path.join(cfg['checkpoints_dir'], log_filename)

    with open(log_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved detailed logs to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar100_config.yaml')
    parser.add_argument('--exp-name', type=str, default='LoRA_ACVM')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg['exp_name'] = args.exp_name

    main(cfg)
