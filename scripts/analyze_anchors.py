import argparse
import pickle
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import numpy as np
from torchvision.datasets import CIFAR100
from tqdm import tqdm


def analyze_anchors(anchors, selected_labels):
    print("\n--- ANALYZE COSINE SIMILARITY---")

    selected_anchors = {lbl: anchors[lbl] for lbl in selected_labels if lbl in anchors}
    labels = list(selected_anchors.keys())

    vectors = [torch.from_numpy(selected_anchors[lbl]).float() for lbl in labels]
    num_labels = len(labels)

    similarity_matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(i, num_labels):

            sim = cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0)).item()
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    try:
        import pandas as pd
        df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
        df = df.apply(lambda x: pd.Series([f'{v:.4f}' for v in x], index=df.columns))
        print("Cosine Similarity Matrix:")
        print(df)
    except ImportError:
        print("Installing pandas for better display...")
        print(labels)
        print(similarity_matrix)

def ensure_labels_file(labels_file):
    p = Path(labels_file)
    if not p.exists():
        print("Labels file not found. Creating from CIFAR100 dataset (download may be required).")
        ds = CIFAR100(root="data", train=True, download=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("\n".join(ds.classes))
        print("Saved labels to", labels_file)


def build_anchors(labels_file, out_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    ensure_labels_file(labels_file)
    labels = [l.strip() for l in open(labels_file).read().splitlines() if l.strip()]
    model = SentenceTransformer(model_name)
    anchors = {}
    for lbl in tqdm(labels, desc="Building anchors"):
        prompt = f"This is an image of {lbl}"
        vec = model.encode(prompt)
        anchors[lbl] = vec
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(anchors, f)
    print(f"Saved anchors to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-file', type=str, default='data/cifar100_labels.txt')
    parser.add_argument('--out', type=str, default='anchors/cifar100_anchors.pkl')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--analyze', action='store_true', help='Set this flag to analyze the resulting anchors.')
    args = parser.parse_args()

    build_anchors(args.labels_file, args.out, args.model)

    if args.analyze:
        try:
            with open(args.out, 'rb') as f:
                anchors = pickle.load(f)

            selected_labels_group = [
                'wolf', 'fox', 'tiger', 'bus', 'train', 'apple', 'orange'
            ]

            analyze_anchors(anchors, selected_labels_group)

        except FileNotFoundError:
            print(f"Error: Can't find anchors file at {args.out}.")