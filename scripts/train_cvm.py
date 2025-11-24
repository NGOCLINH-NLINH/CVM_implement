import argparse
import os
import copy
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from models.resnet_small import SmallResNet18
from utils import load_anchors, ReservoirBuffer, triplet_loss_emb, distance_loss
import numpy as np
from tqdm import tqdm
import pickle


# ---------- Helper: map dataset class idx -> label name (for CIFAR100) ----------
def save_cifar100_labels(out_file="data/cifar100_labels.txt"):
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=".", train=True, download=True)
    labels = ds.classes
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write("\n".join(labels))
    print("Saved CIFAR100 labels to", out_file)


# ---------- Simple dataset wrapper to produce tasks (class-incremental) ----------
def make_class_il_dataloaders(dataset_name="CIFAR100", num_tasks=10, batch_size=64, shuffle_train=True):
    """
    For CIFAR100:
      - split 100 classes into `num_tasks` experiences evenly
      - returns list of (train_loader, val_loader, class_list)
    """
    if dataset_name.lower().startswith("cifar100"):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_full = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
        test_full = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)
        classes = train_full.classes  # list 100
        num_classes = len(classes)
        classes_per_task = num_classes // num_tasks
        dataloaders = []
        for t in range(num_tasks):
            start = t * classes_per_task
            end = start + classes_per_task if t < num_tasks - 1 else num_classes
            task_classes = classes[start:end]  # list of class names
            # build subset indices for train/test
            train_idx = [i for i, (_, label) in enumerate(train_full) if label in list(range(start, end))]
            test_idx = [i for i, (_, label) in enumerate(test_full) if label in list(range(start, end))]
            # But above using label numbers is easier: label numbers correspond to 0..99
            from torch.utils.data import Subset
            train_subset = Subset(train_full, [i for i, (_, label) in enumerate(train_full) if start <= label < end])
            test_subset = Subset(test_full, [i for i, (_, label) in enumerate(test_full) if start <= label < end])
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_train, num_workers=2)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
            dataloaders.append((train_loader, test_loader, list(range(start, end))))
        return dataloaders, classes
    else:
        raise NotImplementedError("Only CIFAR100 implemented in this script.")


# ---------- Training loop ----------
def train_cvm(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # prepare dataloaders/tasks
    dataloaders_tasks, class_names = make_class_il_dataloaders(
        dataset_name="CIFAR100",
        num_tasks=cfg['num_tasks'],
        batch_size=cfg['batch_size']
    )

    # save labels file (for build_anchors)
    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/cifar100_labels.txt", "w") as f:
        f.write("\n".join(class_names))

    # load anchors
    keys, anchor_tensor = load_anchors(cfg['anchors_path'], device=device)
    print("Loaded anchors:", len(keys))

    # model (trainable)
    model = SmallResNet18(out_dim=cfg['out_dim']).to(device)
    prev_model = None

    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=cfg.get('weight_decay', 0.0))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.get('milestones', [30, 45]), gamma=0.1)

    # memory buffer
    buffer = ReservoirBuffer(capacity=cfg['memory_size'])

    # bookkeeping
    seen_class_indices = []
    checkpoints_dir = Path(cfg.get('checkpoints_dir', 'checkpoints'))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # global evaluation store
    task_accs = []

    for t, (train_loader, test_loader, task_class_indices) in enumerate(dataloaders_tasks):
        print(f"\n=== Training Task {t} with classes {task_class_indices} ===")
        # Build anchor indices for current task
        # anchor keys are class names in order from CIFAR100; anchor order must match labels file
        # The anchor_keys list (keys) corresponds to class_names earlier
        cur_anchor_idxs = task_class_indices  # because we saved labels in same order

        # old anchors (classes seen before)
        old_anchor_idxs = [i for i in seen_class_indices]

        seen_class_indices.extend(cur_anchor_idxs)

        # put anchor tensors on device
        anchors_all = anchor_tensor  # [100, D] on device

        # Train epochs for the current task
        model.train()
        for epoch in range(cfg['epochs_per_task']):
            pbar = tqdm(train_loader, desc=f"Task {t} Epoch {epoch}")
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)  # labels are 0..99 global

                # forward
                emb = model(images)  # [B, D]

                # positive anchor: gather per-sample anchor vector by label index
                pos = anchors_all[labels]  # [B, D]

                # negative anchor: choose random anchor from current task classes but not same as label
                # build a tensor of random indices
                neg_indices = []
                num_cur = len(cur_anchor_idxs)
                for lbl in labels.cpu().numpy():
                    # sample a negative class id from current task different than lbl
                    choices = cur_anchor_idxs.copy()
                    if lbl in choices:
                        choices.remove(lbl)
                    if len(choices) == 0:
                        neg_idx = lbl  # fallback
                    else:
                        neg_idx = np.random.choice(choices)
                    neg_indices.append(neg_idx)
                neg = anchors_all[torch.tensor(neg_indices, device=device)]

                # Lm
                Lm = triplet_loss_emb(emb, pos, neg, margin=cfg['margin'])

                # Ld: compare to prev_model distances to old anchors
                if prev_model is None or len(old_anchor_idxs) == 0:
                    Ld = torch.tensor(0.0, device=device)
                else:
                    with torch.no_grad():
                        emb_prev = prev_model(images)
                    old_anchor_mat = anchors_all[old_anchor_idxs]  # [K, D]
                    Ld = distance_loss(emb, emb_prev, old_anchor_mat)

                loss = Lm + cfg['beta'] * Ld

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # add to replay buffer (store CPU tensors)
                buffer.add_batch(images.detach().cpu(), labels.detach().cpu())

                pbar.set_postfix({"loss": float(loss.detach().cpu().numpy()), "Lm": float(Lm.detach().cpu().numpy()),
                                  "Ld": float(Ld.detach().cpu().numpy())})

            scheduler.step()

        # After finishing task training, set prev_model copy
        prev_model = copy.deepcopy(model).eval().to(device)
        # Save checkpoint
        ckpt_path = checkpoints_dir / f"model_task_{t}.pth"
        torch.save({
            'task': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'seen_class_indices': seen_class_indices
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # Evaluate on all seen classes so far (global test). For simplicity, we evaluate by creating a test loader of test samples from seen classes
        # Build dataset of test samples whose labels in seen_class_indices
        # For speed, we use the provided test loader of current task only for quick check
        model.eval()
        # Basic accuracy on this task (compare nearest anchor among seen anchors)
        import torch.nn.functional as F
        correct = 0
        total = 0
        anchors_seen = anchors_all[seen_class_indices]  # [S, D]
        with torch.no_grad():
            for images, labels in test_loader:  # only current task test
                images = images.to(device)
                labels = labels.to(device)
                emb = model(images)  # [B, D]
                sims = emb @ anchors_seen.t()  # [B, S]
                preds = sims.argmax(dim=1)
                # map preds (0..S-1) to global class idx:
                pred_global = [seen_class_indices[i] for i in preds.cpu().numpy()]
                correct += sum([1 for i, l in enumerate(labels.cpu().numpy()) if pred_global[i] == l])
                total += images.shape[0]
        acc = correct / total if total > 0 else 0.0
        print(f"Task {t} accuracy on current task test: {acc:.4f}")
        task_accs.append(acc)

    print("\nTraining finished. Task accuracies:", task_accs)
    # save final model
    torch.save(model.state_dict(), checkpoints_dir / "model_final.pth")
    print("Saved final model.")
    return model, anchor_tensor, keys


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar100_config.yaml')
    parser.add_argument('--anchors-path', type=str, default='anchors/cifar100_anchors.pkl')
    args = parser.parse_args()

    # load config
    cfg_path = args.config
    if os.path.exists(cfg_path):
        cfg = yaml.safe_load(open(cfg_path))
    else:
        # default inline config
        cfg = {
            'num_tasks': 10,
            'batch_size': 64,
            'epochs_per_task': 2,  # set small for quick test; increase for real runs
            'out_dim': 384,
            'lr': 0.1,
            'margin': 0.1,
            'beta': 0.5,
            'memory_size': 1000,
            'anchors_path': args.anchors_path,
            'checkpoints_dir': 'checkpoints'
        }
    # ensure anchors_path in cfg
    cfg['anchors_path'] = args.anchors_path
    train_cvm(cfg)
