import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import json
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from tqdm import tqdm

from models.vit_cvm import ViT_ACVM
from utils.adam_proj import Adam
from utils.utils import (load_anchors, make_cifar100_tasks, set_seed, adaptive_margin_triplet_loss_k_negs,
                         adaptive_margin_triplet_loss_seen_negs)


def evaluate_all_seen(model, test_full, seen_indices, anchors_tensor, device):
    idxs = [i for i, (_, lbl) in enumerate(test_full) if lbl in seen_indices]
    if len(idxs) == 0:
        return 0.0
    loader = DataLoader(Subset(test_full, idxs), batch_size=128, shuffle=False, num_workers=2)
    model.eval()
    anchors_seen = anchors_tensor[seen_indices].to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            emb = model(images.to(device))
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
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            emb = model(images.to(device))
            sims = emb @ anchors_unseen.t()
            preds = sims.argmax(dim=1).cpu().numpy()
            global_preds = [unseen_indices[p] for p in preds]
            true = labels.numpy()
            correct += sum([1 for i in range(len(true)) if global_preds[i] == true[i]])
            total += len(true)
    return correct / total if total > 0 else 0.0


def evaluate_specific_task(model, test_full, eval_indices, seen_indices, anchors_tensor, device):
    idxs = [i for i, (_, lbl) in enumerate(test_full) if lbl in eval_indices]
    if len(idxs) == 0:
        return 0.0
    loader = DataLoader(Subset(test_full, idxs), batch_size=128, shuffle=False, num_workers=2)

    model.eval()
    anchors_seen = anchors_tensor[seen_indices].to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            emb = model(images.to(device))
            sims = emb @ anchors_seen.t()
            preds = sims.argmax(dim=1).cpu().numpy()

            global_preds = [seen_indices[p] for p in preds]
            true = labels.numpy()
            correct += sum([1 for i in range(len(true)) if global_preds[i] == true[i]])
            total += len(true)
    return correct / total if total > 0 else 0.0


def main(cfg):
    set_seed(cfg.get('seed', 1234))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 50}")
    print(f"EXPERIMENT: {cfg['exp_name']} | SEED: {cfg.get('seed', 1234)}")
    print(f"MODE: Exemplar-Free LoRA-ACVM (0 Memory Buffer)")
    print(f"DEVICE: {device}")
    print(f"{'=' * 50}\n")

    tasks, class_names = make_cifar100_tasks(cfg['num_tasks'], cfg['batch_size'], augment=True)

    test_full = datasets.CIFAR100(root="data", train=False, download=True, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]))

    anchor_keys, anchors_tensor = load_anchors(cfg['anchors_path'], device=device)

    model = ViT_ACVM(num_tasks=cfg['num_tasks'], rank=cfg.get('rank', 64)).to(device)

    seen_inds = []
    seen_acc_history = []
    zero_shot_history = []
    Path(cfg['checkpoints_dir']).mkdir(parents=True, exist_ok=True)

    task_classes_list = []

    for t, (train_loader, test_loader, class_inds) in enumerate(tasks):
        task_classes_list.append(class_inds)

        print(f"\n=== Training Task {t} (Classes: {min(class_inds)}-{max(class_inds)}) ===")
        cur_inds = class_inds
        seen_inds += cur_inds

        model.update_task()
        model.freeze_for_task()

        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable_params)
        print(f">>> DEBUG: Number of params being trained: {num_trainable}")
        if num_trainable == 0:
            raise ValueError("FATAL: None param is being trained")

        # params_svd = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_A' in n]
        # params_normal = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' in n]

        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        wd_svd = 0.0 if t > 0 else cfg.get('weight_decay_normal', 0.0005)

        # opt_groups = [
        #     {'params': params_svd, 'svd': True, 'thres': cfg.get('thres', 0.995), 'weight_decay': wd_svd, 'lr': cfg['lr'] * 3.0},
        #     {'params': params_normal, 'svd': False, 'weight_decay': cfg.get('weight_decay_normal', 0.0005), 'lr': cfg['lr']}
        # ]

        opt_groups = [
            {
                'params': trainable_params,
                'svd': True,
                'thres': cfg.get('thres', 0.99),
                'weight_decay': cfg.get('weight_decay_normal', 0.0005),
                'lr': cfg['lr'] * 2.0
            }
        ]
        optimizer = Adam(opt_groups, lr=cfg['lr'])

        if t > 0:
            print(">> [Stage 1] Calculating Drift-Resistant Space (DRS)...")
            model.eval()
            with torch.no_grad():
                for images_aug, images_raw, _ in train_loader:
                    _ = model(images_raw.to(device), get_cur_x=True)

            fea_in = model.extract_fea_in(device)
            optimizer.get_eigens(fea_in)
            optimizer.get_transforms()
            print(">> [Stage 1] Completed. Projection Matrices built.")

        print(">> [Stage 2] Training LoRA with Adaptive Triplet Loss...")
        model.train()
        total_steps = cfg['epochs_per_task'] * len(train_loader)
        scheduler = (torch.optim.lr_scheduler
                     .CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(cfg.get('eta_min', 1e-6))))

        pbar = tqdm(total=total_steps, desc=f"Task {t}", dynamic_ncols=True)

        for epoch in range(cfg['epochs_per_task']):
            for images_aug, _, labels in train_loader:
                images_aug, labels = images_aug.to(device), labels.to(device)

                emb = model(images_aug)
                pos = anchors_tensor[labels]

                # K = cfg.get('k_negs', 9)
                # neg_idx_list = []
                # for lbl in labels.cpu().numpy():
                #     choices = [c for c in seen_inds if c != lbl]
                #     if len(choices) >= K:
                #         negs = random.sample(choices, k=K)
                #     else:
                #         negs = random.choices(choices, k=K) if choices else [lbl] * K
                #     neg_idx_list.append(negs)
                #
                # neg_k_tensor = anchors_tensor[torch.tensor(neg_idx_list, dtype=torch.long, device=device)]

                # loss_trip = adaptive_margin_triplet_loss_k_negs(emb, pos, neg_k_tensor, base_margin=cfg['margin'])

                # loss_trip = adaptive_margin_triplet_loss_seen_negs(emb, pos, labels, anchors_tensor, seen_inds,
                #                                                    base_margin=cfg['margin'])
                # loss_attr = (1.0 - (emb * pos).sum(dim=1)).mean()
                # loss = loss_trip + cfg.get('attr_loss_weight', 0.1) * loss_attr

                sims = emb @ anchors_tensor.t()
                logits = sims / cfg['temperature']
                loss = torch.nn.functional.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get('clip_grad', 5.0))
                optimizer.step()

                scheduler.step()

                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.3f}"})
        pbar.close()

        task_accs = []
        for i in range(t + 1):
            acc_i = evaluate_specific_task(model, test_full, task_classes_list[i], seen_inds, anchors_tensor, device)
            task_accs.append(acc_i)
            print(
                f"  -> Acc on Task {i} (Classes {min(task_classes_list[i])}-{max(task_classes_list[i])}): {acc_i:.4f}")

        acc_all_seen = np.mean(task_accs)
        seen_acc_history.append(acc_all_seen)
        print(f"[*] Average Accuracy on all SEEN tasks: {acc_all_seen:.4f}")

        unseen_inds = [i for i in range(len(anchor_keys)) if i not in seen_inds]
        zs = zero_shot_eval(model, anchors_tensor, unseen_inds, test_full, device)
        zero_shot_history.append(zs)
        print(f"[*] Zero-shot accuracy on UNSEEN classes: {zs:.4f}")

    print(f"\n{'=' * 50}")
    print(f"FINAL RESULTS")
    print(f"Average Accuracy over {cfg['num_tasks']} tasks: {np.mean(seen_acc_history):.4f}")

    log_path = os.path.join(cfg['checkpoints_dir'], f"log_{cfg['exp_name']}_seed{cfg.get('seed', 1234)}.json")
    with open(log_path, 'w') as f:
        json.dump({
            "exp_name": cfg['exp_name'],
            "seen_acc_history": [float(x) for x in seen_acc_history],
            "zero_shot_history": [float(x) for x in zero_shot_history],
            "avg_acc": float(np.mean(seen_acc_history))
        }, f, indent=4)
    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar100_config.yaml')
    parser.add_argument('--exp-name', type=str, default='LoRA_ACVM')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg['exp_name'] = args.exp_name

    if 'rank' not in cfg:
        cfg['rank'] = 64

    main(cfg)
