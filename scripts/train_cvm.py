import sys
import os

from sklearn.linear_model import LogisticRegression

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
from utils import load_anchors, ReservoirBuffer, triplet_loss_emb, semantic_distance_loss, make_cifar100_tasks, \
    set_seed, triplet_loss_k_negs, triplet_loss_seen_negs, anchor_attraction_loss, image_side_prototype_spread_loss, \
    adaptive_margin_triplet_loss_k_negs

replay_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


def evaluate_all_seen(model, test_full, seen_indices, anchors_tensor, anchor_keys, device):
    # Build a test loader for all seen classes using test_full Subset
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
            global_preds = [seen_indices[p] for p in preds]  # convert index in seen list -> global class idx
            true = labels.numpy()
            correct += sum([1 for i in range(len(true)) if global_preds[i] == true[i]])
            total += len(true)
    return correct / total if total > 0 else 0.0


def evaluate_task_full_anchors(model, test_full, task_class_inds, anchors_tensor, anchor_keys, device):
    # Evaluate accuracy of ONE task using FULL anchor set
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
    # For zero-shot CVM: build anchor embeddings for unseen classes and test on their test examples using nearest anchor
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

    # Extract features
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

    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    return (preds == y_te).mean()


def compute_forgetting(eval_history):
    """
    eval_history[t][i]: accuracy of task i after training task t
    """
    T = len(eval_history)
    forgetting = []

    for i in range(T - 1):  # last task has no forgetting
        acc_i_over_time = [eval_history[t][i] for t in range(i, T)]
        max_acc = max(acc_i_over_time[:-1])  # before final
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

    # tasks
    tasks, class_names = make_cifar100_tasks(cfg['num_tasks'], cfg['batch_size'], augment=True)
    # Lưu ý: root="data" sẽ tạo folder data trong thư mục hiện hành khi chạy
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

    # anchors
    if not os.path.exists(cfg['anchors_path']):
        # Fallback check for relative path issues
        if os.path.exists(os.path.join('..', cfg['anchors_path'])):
            cfg['anchors_path'] = os.path.join('..', cfg['anchors_path'])

    anchor_keys, anchors_tensor = load_anchors(cfg['anchors_path'], device=device)
    print("Loaded anchors:", len(anchor_keys))

    # model
    model = ResNetCVM(out_dim=cfg['out_dim'], pretrained=cfg.get('pretrained_backbone', False)).to(device)
    prev_model = None

    buffer = ReservoirBuffer(capacity=cfg['memory_size'])

    seen_inds = []

    seen_acc_history = []  # Accuracy on all seen classes after each task
    zero_shot_history = []  # Zero-shot accuracy on unseen classes after each task
    linear_probe_history = []  # Linear probe accuracy after each task
    eval_history = []  # Detailed per-task accuracy matrix (for forgetting)

    Path(cfg['checkpoints_dir']).mkdir(parents=True, exist_ok=True)

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
            K = 9

            for images, raw_images, labels in train_loader:
                images_cuda = images.to(device)
                labels_cuda = labels.to(device)
                emb = model(images_cuda)

                if prev_model is not None and len(old_inds) > 0:
                    old_anchor_mat = anchors_tensor[old_inds].to(device)
                else:
                    old_anchor_mat = None

                pos = anchors_tensor[labels_cuda].to(device)

                neg_idx_list = []
                for lbl in labels.numpy():
                    choices = [c for c in cur_inds if c != lbl]
                    if len(choices) >= K:
                        negs = random.sample(choices, k=K)
                    else:
                        negs = random.choices(choices, k=K)
                    neg_idx_list.append(negs)
                neg_k_tensor = anchors_tensor[torch.tensor(neg_idx_list, dtype=torch.long, device=device)]

                if cfg['adaptive_margin']:
                    Lm = adaptive_margin_triplet_loss_k_negs(emb, pos, neg_k_tensor, base_margin=cfg['margin'])
                else:
                    Lm = triplet_loss_k_negs(emb, pos, neg_k_tensor, margin=cfg['margin'])

                if cfg['spread_lambda'] > 0:
                    L_spread = image_side_prototype_spread_loss(emb, labels_cuda, anchors_tensor, seen_inds,
                                                                delta=cfg['spread_delta'])
                else:
                    L_spread = torch.tensor(0.0, device=device)

                if old_anchor_mat is not None and cfg['beta'] > 0:
                    with torch.no_grad():
                        emb_prev = prev_model(images_cuda)
                    Ld = semantic_distance_loss(emb, emb_prev, old_anchor_mat)
                else:
                    Ld = torch.tensor(0.0, device=device)

                loss = Lm + cfg['beta'] * Ld + cfg['spread_lambda'] * L_spread

                if len(buffer) > 0 and cfg['replay_batch'] > 0 and cfg['replay_on']:
                    buf_imgs_raw, buf_labels = buffer.sample(cfg['replay_batch'])
                    if buf_imgs_raw is not None:
                        buf_imgs_raw = buf_imgs_raw.to(device);
                        buf_labels = buf_labels.to(device)
                        buf_imgs_aug = replay_transform(buf_imgs_raw)
                        emb_buf = model(buf_imgs_aug)
                        pos_buf = anchors_tensor[buf_labels].to(device)

                        Lm_buf = triplet_loss_seen_negs(emb_buf, pos_buf, buf_labels, anchors_tensor, seen_inds,
                                                        margin=cfg['margin'])

                        Ld_buf = torch.tensor(0.0, device=device)
                        if old_anchor_mat is not None and cfg['beta'] > 0:
                            with torch.no_grad(): emb_prev_buf = prev_model(buf_imgs_aug)
                            Ld_buf = semantic_distance_loss(emb_buf, emb_prev_buf, old_anchor_mat)

                        loss += cfg['replay_lambda'] * (Lm_buf + cfg['beta'] * Ld_buf)

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

        print(f"--- Evaluation after Task {t} ---")

        # 1. ACCURACY ON ALL SEEN CLASSES
        acc_all_seen = evaluate_all_seen(model, test_full, seen_inds, anchors_tensor, anchor_keys, device)
        seen_acc_history.append(acc_all_seen)
        print(f"Acc on all seen classes after task {t}: {acc_all_seen:.4f}")

        # 2. LINEAR PROBING SCORE FOR SEEN CLASSES
        lp_acc = linear_probe_all(model, train_full, test_full, seen_inds, device, cfg['out_dim'])
        linear_probe_history.append(lp_acc)
        print(f"Linear probe acc on seen classes after task {t}: {lp_acc:.4f}")

        # 3. ZERO-SHOT ON UNSEEN CLASSES
        unseen_inds = [i for i in range(len(anchor_keys)) if i not in seen_inds]
        zs = zero_shot_eval(model, anchors_tensor, unseen_inds, test_full, device)
        zero_shot_history.append(zs)
        print(f"Zero-shot acc on unseen classes after task {t}: {zs:.4f}")

        # 4. FORGETTING HISTORY
        per_task_accs = []
        for i_task, (_, _, t_classes) in enumerate(tasks):
            if i_task > t:
                per_task_accs.append(None)
            else:
                acc_old_task = evaluate_task_full_anchors(model, test_full, t_classes, anchors_tensor, anchor_keys,
                                                          device)
                per_task_accs.append(acc_old_task)
        eval_history.append(per_task_accs)

    # Final Metrics
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar100_config.yaml')
    parser.add_argument('--exp-name', type=str, default='default')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--anchors-path', type=str, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--spread-lambda', type=float, default=None)
    parser.add_argument('--margin', type=float, default=None)
    parser.add_argument('--memory-size', type=int, default=None)
    parser.add_argument('--no-adaptive', action='store_true')

    args = parser.parse_args()

    # Load YAML
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg['exp_name'] = args.exp_name

    if args.seed is not None: cfg['seed'] = args.seed
    if args.anchors_path: cfg['anchors_path'] = args.anchors_path

    if args.beta is not None: cfg['beta'] = args.beta
    if args.spread_lambda is not None: cfg['spread_lambda'] = args.spread_lambda
    if args.margin is not None: cfg['margin'] = args.margin
    if args.memory_size is not None: cfg['memory_size'] = args.memory_size

    if args.no_adaptive:
        cfg['adaptive_margin'] = False
    else:
        cfg.setdefault('adaptive_margin', True)

    cfg['lr'] = float(cfg['lr'])
    cfg['momentum'] = float(cfg['momentum'])
    cfg['weight_decay'] = float(cfg['weight_decay'])
    cfg['batch_size'] = int(cfg['batch_size'])

    cfg['margin'] = float(cfg['margin'])
    cfg['beta'] = float(cfg['beta'])
    cfg['spread_lambda'] = float(cfg['spread_lambda'])
    cfg['spread_delta'] = float(cfg['spread_delta'])
    cfg['replay_lambda'] = float(cfg['replay_lambda'])

    main(cfg)