import os
import argparse
import yaml
import copy
from pathlib import Path
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
from models.resnet_cvm import ResNetCVM
from utils import load_anchors, ReservoirBuffer, triplet_loss_emb, semantic_distance_loss, make_cifar100_tasks, \
    set_seed, triplet_loss_k_negs, triplet_loss_seen_negs, anchor_attraction_loss
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import random


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


def linear_probe_all(model, train_full, test_full, seen_indices, device, out_dim):
    # freeze model, compute features for train and test over seen classes
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
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='saga')
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    return (preds == y_te).mean()


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

    return sum(forgetting) / len(forgetting), forgetting


def main(cfg):
    set_seed(cfg.get('seed', 1234))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # tasks
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

    # anchors
    anchor_keys, anchors_tensor = load_anchors(cfg['anchors_path'], device=device)
    print("Loaded anchors:", len(anchor_keys))

    # model
    model = ResNetCVM(out_dim=cfg['out_dim'], pretrained=cfg.get('pretrained_backbone', False)).to(device)
    prev_model = None

    buffer = ReservoirBuffer(capacity=cfg['memory_size'])

    seen_inds = []
    task_accs = []
    eval_history = []  # list of per-task arrays for forgetting calculation

    for t, (train_loader, test_loader, class_inds) in enumerate(tasks):
        print(f"\n=== Training Task {t} with classes {class_inds} ===")
        cur_inds = class_inds
        old_inds = [i for i in seen_inds]
        seen_inds += cur_inds

        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.get('milestones', [30, 45]), gamma=0.1)

        total_steps = cfg['epochs_per_task'] * len(train_loader)
        pbar = tqdm(
            total=total_steps,
            desc=f"Task {t}",
            dynamic_ncols=True
        )

        # Training loop per epoch
        for epoch in range(cfg['epochs_per_task']):
            model.train()

            K = 9  # number of negatives
            for images, labels in train_loader:
                images_cuda = images.to(device)
                labels_cuda = labels.to(device)  # global labels 0..99

                # current batch embeddings
                emb = model(images_cuda)

                # positive anchors per sample
                pos = anchors_tensor[labels_cuda].to(device)

                # negative: sample random label within current task but != label
                neg_idx_list = []
                for lbl in labels.numpy():
                    choices = [c for c in cur_inds if c != lbl]
                    if len(choices) >= K:
                        negs = random.sample(choices, k=K)
                    else:
                        negs = random.choices(choices, k=K)
                    neg_idx_list.append(negs)

                neg_idx = torch.tensor(neg_idx_list, dtype=torch.long, device=device)
                neg_k_tensor = anchors_tensor[neg_idx]

                Lm = triplet_loss_k_negs(emb, pos, neg_k_tensor, margin=cfg['margin'])

                L_anchor = anchor_attraction_loss(emb, pos)

                # compute Ld if prev_model exists and old anchors exist
                if prev_model is None or len(old_inds) == 0:
                    Ld = torch.tensor(0.0, device=device)
                else:
                    with torch.no_grad():
                        emb_prev = prev_model(images_cuda)
                    old_anchor_mat = anchors_tensor[old_inds].to(device)
                    Ld = semantic_distance_loss(emb, emb_prev, old_anchor_mat)

                loss = Lm + cfg['beta'] * Ld + cfg['anchor_lambda'] * L_anchor

                # replay mixing: sample buffer and compute loss on replay items and mix
                if len(buffer) > 0 and cfg['replay_batch'] > 0:
                    buf_imgs, buf_labels = buffer.sample(cfg['replay_batch'])
                    if buf_imgs is not None:
                        buf_imgs = buf_imgs.to(device)
                        buf_labels = buf_labels.to(device)
                        emb_buf = model(buf_imgs)
                        pos_buf = anchors_tensor[buf_labels].to(device)
                        # choose negatives from all seen classes for buffer items
                        # neg_idx_buf_list = []
                        # for lbl in buf_labels.cpu().numpy():
                        #     choices = [i for i in seen_inds if i != lbl]
                        #     if len(choices) >= K:
                        #         negs = random.sample(choices, k=K)
                        #     else:
                        #         negs = random.choices(choices, k=K)
                        #     neg_idx_buf_list.append(negs)
                        #
                        # neg_idx_buf = torch.tensor(neg_idx_buf_list, dtype=torch.long, device=device)
                        # neg_buf_k_tensor = anchors_tensor[neg_idx_buf]
                        # Lm_buf = triplet_loss_k_negs(emb_buf, pos_buf, neg_buf_k_tensor, margin=cfg['margin'])

                        Lm_buf = triplet_loss_seen_negs(emb_buf, pos_buf, buf_labels, anchors_tensor, seen_inds,
                                                        margin=cfg['margin'])
                        L_anchor_buf = anchor_attraction_loss(emb_buf, pos_buf)
                        # no Ld for replay for simplicity, or could compute with prev_model
                        loss = loss + cfg['replay_lambda'] * (Lm_buf + + cfg['anchor_lambda'] * L_anchor_buf)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # add to buffer
                buffer.add_batch(images, labels)

                pbar.update(1)
                pbar.set_postfix({
                    "e": epoch,
                    "a": f"{L_anchor.item():.3f}",
                    "m": f"{Lm.item():.3f}",
                    "d": f"{Ld.item():.3f}"
                })

            scheduler.step()

        pbar.close()

        # after finishing task t
        prev_model = copy.deepcopy(model).eval().to(device)

        # Save checkpoint
        Path(cfg['checkpoints_dir']).mkdir(parents=True, exist_ok=True)
        ckpt = {
            'task': t,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'seen_inds': seen_inds
        }
        torch.save(ckpt, os.path.join(cfg['checkpoints_dir'], f"model_task_{t}.pth"))
        print("Saved checkpoint for task", t)

        # Evaluate: accuracy on current task test loader using nearest anchor among seen anchors
        acc_t = evaluate_all_seen(model, test_full, cur_inds, anchors_tensor, anchor_keys, device)
        print(f"Task {t} accuracy on current test: {acc_t:.4f}")
        task_accs.append(acc_t)

        # Evaluate performance on all seen classes (full test set)
        acc_all_seen = evaluate_all_seen(model, test_full, seen_inds, anchors_tensor, anchor_keys, device)
        print(f"Acc on all seen classes after task {t}: {acc_all_seen:.4f}")

        # Linear probing score for seen classes
        lp_acc = linear_probe_all(model, train_full, test_full, seen_inds, device, cfg['out_dim'])
        print(f"Linear probe acc on seen classes after task {t}: {lp_acc:.4f}")

        # Zero-shot on unseen classes
        unseen_inds = [i for i in range(len(anchor_keys)) if i not in seen_inds]
        zs = zero_shot_eval(model, anchors_tensor, unseen_inds, test_full, device)
        print(f"Zero-shot acc on unseen classes after task {t}: {zs:.4f}")

        # store evaluation snapshot (for forgetting computation)
        # compute accuracy per task on their test sets
        per_task_accs = []
        for i_task, (_, _, t_classes) in enumerate(tasks):
            if i_task > t:
                per_task_accs.append(None)
                continue

            acc_i = evaluate_task_full_anchors(
                model,
                test_full,
                t_classes,
                anchors_tensor,
                anchor_keys,
                device
            )
            per_task_accs.append(acc_i)
        eval_history.append(per_task_accs)

    # final save
    torch.save(model.state_dict(), os.path.join(cfg['checkpoints_dir'], "model_final.pth"))
    print("Training finished. Task accs (per-task):", task_accs)

    # compute forgetting approx
    # convert eval_history into matrix: eval_history[after_task_t][task_i]
    print(
        "Eval history (summary):",
        [[round(x, 3) if x is not None else "-" for x in row]
         for row in eval_history]
    )

    fw_score, fw_per_task = compute_forgetting(eval_history)
    print("Forgetting score (FW):", round(fw_score, 4))
    print("Per-task forgetting:", [round(f, 4) for f in fw_per_task])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar100_config.yaml')
    parser.add_argument('--anchors-path', type=str, default='anchors/cifar100_anchors.pkl')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['lr'] = float(cfg['lr'])
    cfg['momentum'] = float(cfg['momentum'])
    cfg['weight_decay'] = float(cfg['weight_decay'])
    cfg['margin'] = float(cfg['margin'])
    cfg['beta'] = float(cfg['beta'])
    cfg['anchor_lambda'] = float(cfg['anchor_lambda'])

    cfg['batch_size'] = int(cfg['batch_size'])
    cfg['epochs_per_task'] = int(cfg['epochs_per_task'])
    cfg['num_tasks'] = int(cfg['num_tasks'])
    cfg['memory_size'] = int(cfg['memory_size'])
    cfg['replay_batch'] = int(cfg['replay_batch'])

    cfg['replay_on'] = bool(cfg['replay_on'])
    cfg['pretrained_backbone'] = bool(cfg['pretrained_backbone'])
    main(cfg)
