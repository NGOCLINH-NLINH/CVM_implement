import argparse
import pickle
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def get_anchor_similarity(class_a, class_b, anchors_path):
    print(f"[*] Reading anchors file from: {anchors_path}")
    try:
        with open(anchors_path, 'rb') as f:
            anchors = pickle.load(f)
    except FileNotFoundError:
        print(f"[-] Error: Can't find anchors file at {anchors_path}")
        return None

    if class_a not in anchors:
        print(f"[-] Error: Class '{class_a}' not found")
        return None
    if class_b not in anchors:
        print(f"[-] Error: Class '{class_b}' not found")
        return None

    vec_a = torch.tensor(anchors[class_a]).float().unsqueeze(0)
    vec_b = torch.tensor(anchors[class_b]).float().unsqueeze(0)

    sim = F.cosine_similarity(vec_a, vec_b, dim=1).item()

    print(f"[+] Cosine Similarity between '{class_a}' and '{class_b}': {sim:.4f}")
    return sim


def visualize_full_matrix(anchors_path, out_img_path):
    print(f"[*] Reading anchors file from: {anchors_path}")
    with open(anchors_path, 'rb') as f:
        anchors = pickle.load(f)

    labels = list(anchors.keys())

    mat = torch.tensor(np.stack(list(anchors.values()))).float()
    mat = torch.nn.functional.normalize(mat, p=2, dim=1)

    print("[*] Calculating Cosine Similarity Matrix...")
    sim_matrix = (mat @ mat.t()).numpy()

    print(f"[*] Drawing Heatmap for {len(labels)} classes...")
    plt.figure(figsize=(30, 24))

    ax = sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="magma",
        annot=False,
        square=True,
        cbar_kws={"shrink": .8}
    )

    plt.title("Cosine Similarity Matrix of All CIFAR100 Anchors", fontsize=24, pad=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img_path, dpi=300)
    print(f"[+] Save graph {out_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchors', type=str, default='anchors/cifar100_anchors.pkl')
    parser.add_argument('--out', type=str, default='anchors_similarity_heatmap.png')
    parser.add_argument('--c1', type=str, default=None)
    parser.add_argument('--c2', type=str, default=None)
    parser.add_argument('--heatmap', action='store_true')

    args = parser.parse_args()

    if args.c1 and args.c2:
        get_anchor_similarity(args.c1, args.c2, args.anchors)
    elif args.c1 or args.c2:
        print("[-] Missing --c1 or --c2")
    if args.heatmap or (not args.c1 and not args.c2):
        visualize_full_matrix(args.anchors, args.out)