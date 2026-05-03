import sys
import os

from peft import LoraConfig, get_peft_model
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

from models.resnet_cvm import DeterministicResNetCVM
from utils import load_anchors, ReservoirBuffer, triplet_loss_emb, semantic_distance_loss, make_cifar100_tasks, \
    set_seed, triplet_loss_k_negs, triplet_loss_seen_negs, anchor_attraction_loss, image_side_prototype_spread_loss, \
    adaptive_margin_triplet_loss_k_negs, standard_margin_loss, kl_divergence_loss, variance_regularization_loss, evaluate_all_seen_multi_lora

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
    # train_full = datasets.CIFAR100(root="data", train=True, download=True, transform=transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    # ]))
    test_full = datasets.CIFAR100(root="data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]))

    if not os.path.exists(cfg['anchors_path']):
        if os.path.exists(os.path.join('..', cfg['anchors_path'])):
            cfg['anchors_path'] = os.path.join('..', cfg['anchors_path'])

    anchor_keys, anchors_tensor = load_anchors(cfg['anchors_path'], device=device)
    print("Loaded anchors:", len(anchor_keys))

    base_model = DeterministicResNetCVM(out_dim=cfg['out_dim'], pretrained=True).to(device)

    for param in base_model.parameters():
        param.requires_grad = False

    task_router = nn.Linear(512, cfg['num_tasks']).to(device)
    lora_config = LoraConfig(
        r=16,
        target_modules=["conv1", "conv2"],
        modules_to_save=["fc"],
        bias="none"
    )

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

        adapter_name = f"task_{t}"
        if t == 0:
            global model
            model = get_peft_model(base_model, lora_config, adapter_name=adapter_name)
        else:
            model.add_adapter(adapter_name, lora_config)

        model.set_adapter(adapter_name)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.get('milestones', [50, 75]), gamma=0.1)
        total_steps = cfg['epochs_per_task'] * len(train_loader)
        pbar = tqdm(total=total_steps, desc=f"Task {t}", dynamic_ncols=True)

        for epoch in range(cfg['epochs_per_task']):
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            for images, raw_images, labels in train_loader:
                images_cuda, labels_cuda = images.to(device), labels.to(device)

                mu = model(images_cuda)
                loss = standard_margin_loss(mu, labels_cuda, anchors_tensor, seen_inds, margin=cfg['margin'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                buffer.add_batch(raw_images, labels, task_id=t)
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.3f}"})

            scheduler.step()

        print("Training Task Router...")
        router_optimizer = optim.Adam(task_router.parameters(), lr=0.001)
        criterion_router = nn.CrossEntropyLoss()

        buf_imgs, _, buf_task_ids = buffer.get_all_data()

        if len(buf_imgs) > 0:
            task_router.train()
            buf_dataset = torch.utils.data.TensorDataset(buf_imgs, buf_task_ids)
            buf_loader = DataLoader(buf_dataset, batch_size=128, shuffle=True)

            for _ in range(cfg.get('router_epochs', 10)):
                for b_img, b_tid in buf_loader:
                    b_img, b_tid = b_img.to(device), b_tid.to(device)

                    with torch.no_grad():
                        with model.disable_adapter():
                            feats = model.base_model.features(b_img)
                            feats = model.base_model.avgpool(feats)
                            feats = torch.flatten(feats, 1)

                    router_preds = task_router(feats)
                    router_loss = criterion_router(router_preds, b_tid)

                    router_optimizer.zero_grad()
                    router_loss.backward()
                    router_optimizer.step()
        pbar.close()

        prev_model = copy.deepcopy(model).eval().to(device)
        for param in prev_model.parameters():
            param.requires_grad = False

        print(f"--- Evaluation after Task {t} ---")

        seen_test_idx = [i for i, lbl in enumerate(test_full.targets) if lbl in seen_inds]
        seen_test_loader = DataLoader(Subset(test_full, seen_test_idx), batch_size=128, shuffle=False, num_workers=2)

        acc_all_seen = evaluate_all_seen_multi_lora(model, task_router, seen_test_loader, anchors_tensor, device)
        seen_acc_history.append(acc_all_seen)
        print(f"Acc on all seen classes after task {t}: {acc_all_seen:.4f}")

        per_task_accs = []
        for i_task, (_, test_loader_task, t_classes) in enumerate(tasks):
            if i_task > t:
                per_task_accs.append(None)
            else:
                adapter_name = f"task_{i_task}"
                model.set_adapter(adapter_name)
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
