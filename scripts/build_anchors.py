import argparse

import torch
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import CIFAR100, FGVCAircraft
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import os
import shutil


def prepare_tinyimagenet(root="data/tiny-imagenet-200"):
    val_dir = os.path.join(root, 'val')
    formatted_dir = os.path.join(val_dir, 'images_formatted')
    labels_path = os.path.join(os.path.dirname(root), 'tiny_imagenet_labels.txt')
    train_dir = os.path.join(root, 'train')

    if not os.path.exists(labels_path):
        print("Creating TinyImageNet labels file...")
        words_file = os.path.join(root, 'words.txt')

        if os.path.exists(words_file) and os.path.exists(train_dir):
            words_dict = {}
            with open(words_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        words_dict[parts[0]] = parts[1].split(',')[0]

            classes = sorted(os.listdir(train_dir))
            human_labels = [words_dict.get(c, c) for c in classes]

            with open(labels_path, 'w') as f:
                f.write('\n'.join(human_labels))
            print(f"Saved 200 clean labels to {labels_path}")
    if not os.path.exists(formatted_dir):
        print("Formatting TinyImageNet validation directory...")
        val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')

        if os.path.exists(val_annotations_path):
            os.makedirs(formatted_dir, exist_ok=True)
            classes = sorted(os.listdir(train_dir))
            for c in classes:
                os.makedirs(os.path.join(formatted_dir, c), exist_ok=True)
            with open(val_annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_file, cls = parts[0], parts[1]
                        src = os.path.join(val_dir, 'images', img_file)
                        dst = os.path.join(formatted_dir, cls, img_file)
                        if os.path.exists(src):
                            shutil.move(src, dst)
            print("Formatted Validation Directory Successfully!")
        else:
            print("val_annotations.txt not found, skip formatting.")


def ensure_labels_file(labels_file):
    p = Path(labels_file)
    if not p.exists():
        print(f"Labels file not found at {labels_file}.")
        p.parent.mkdir(parents=True, exist_ok=True)

        if 'tiny' in labels_file.lower():
            print("Creating from TinyImageNet dataset...")
            data_dir = str(PROJECT_ROOT / "data/tiny-imagenet-200")
            prepare_tinyimagenet(data_dir)

        elif 'aircraft' in labels_file.lower():
            print("Creating from FGVC Aircraft dataset...")
            data_dir = str(PROJECT_ROOT / "data")
            ds = FGVCAircraft(root=data_dir, split='train', download=True)
            aircraft_classes = [f"{c} aircraft" for c in ds.classes]
            with open(p, "w") as f:
                f.write("\n".join(aircraft_classes))

        elif 'gtsrb' in labels_file.lower():
            print("Creating from GTSRB dataset...")
            base_gtsrb = [
                "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60",
                "speed limit 70", "speed limit 80", "end of speed limit 80", "speed limit 100",
                "speed limit 120", "no passing", "no passing for vehicles over 3.5 metric tons",
                "right-of-way at the next intersection", "priority road", "yield", "stop",
                "no vehicles", "vehicles over 3.5 metric tons prohibited", "no entry",
                "general caution", "dangerous curve to the left", "dangerous curve to the right",
                "double curve", "bumpy road", "slippery road", "road narrows on the right",
                "road work", "traffic signals", "pedestrians", "children crossing",
                "bicycles crossing", "beware of ice or snow", "wild animals crossing",
                "end of all speed and passing limits", "turn right ahead", "turn left ahead",
                "ahead only", "go straight or right", "go straight or left", "keep right",
                "keep left", "roundabout mandatory", "end of no passing",
                "end of no passing by vehicles over 3.5 metric tons"
            ]
            gtsrb_classes = [f"{lbl} traffic sign" for lbl in base_gtsrb]
            with open(p, "w") as f:
                f.write("\n".join(gtsrb_classes))

        else:
            print("Creating from CIFAR100 dataset...")
            data_dir = str(PROJECT_ROOT / "data")
            ds = CIFAR100(root=data_dir, train=True, download=False)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-file', type=str, default=str(PROJECT_ROOT / 'data/cifar100_labels.txt'))
    parser.add_argument('--out', type=str, default=str(PROJECT_ROOT / 'anchors/cifar100_anchors.pkl'))
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    build_anchors(args.labels_file, args.out, args.model)
