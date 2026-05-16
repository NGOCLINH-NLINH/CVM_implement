import argparse

import torch
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import CIFAR100
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_labels_file(labels_file):
    p = Path(labels_file)
    if not p.exists():
        print("Labels file not found. Creating from CIFAR100 dataset...")
        data_dir = str(PROJECT_ROOT / "data")
        ds = CIFAR100(root=data_dir, train=True, download=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("\n".join(ds.classes))
        print("Saved labels to", labels_file)


def build_anchors(labels_file, out_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    ensure_labels_file(labels_file)
    labels = [l.strip() for l in open(labels_file).read().splitlines() if l.strip()]

    print(f"\n[*] Initialize language model: '{model_name}'...")
    model = SentenceTransformer(model_name)

    print("[*] Encoding features...")

    anchors = {}
    for lbl in tqdm(labels, desc="Building anchors"):
        prompt = f"This is an image of {lbl}"
        vec = model.encode(prompt)
        anchors[lbl] = vec

    mat = np.array(list(anchors.values()))
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)

    for i, lbl in enumerate(anchors.keys()):
        anchors[lbl] = mat[i]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(anchors, f)
    print(f"\n[+] Saved {len(anchors)} anchors at: {out_path}")


def analyze_anchors(pkl_path):
    print(f"[*] Reading anchors file from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        anchors = pickle.load(f)

    keys = list(anchors.keys())
    mat = torch.tensor([anchors[k] for k in keys])
    sim_matrix = mat @ mat.t()

    print("\n=== TOP 3 NEARESR ANCHORS ===")
    sim_matrix.fill_diagonal_(-1.0)

    for i in range(3):
        max_val = torch.max(sim_matrix)
        idx = (sim_matrix == max_val).nonzero(as_tuple=True)
        class_A = keys[idx[0][0]]
        class_B = keys[idx[1][0]]

        print(f"{i + 1}. '{class_A}' and '{class_B}' - Similarity: {max_val.item():.4f}")
        sim_matrix[idx[0][0], idx[1][0]] = -1.0
        sim_matrix[idx[1][0], idx[0][0]] = -1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-file', type=str, default=str(PROJECT_ROOT / 'data/cifar100_labels.txt'))
    parser.add_argument('--out', type=str, default=str(PROJECT_ROOT / 'anchors/cifar100_anchors.pkl'))
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--analyze-only', action='store_true', help="khong can build lai")
    args = parser.parse_args()
    if not args.analyze_only:
        build_anchors(args.labels_file, args.out, args.model)
    analyze_anchors(args.out)
