#!/usr/bin/env bash
set -e
# Run in Kaggle notebook or local bash
pip install -r requirements.txt

# build labels file + anchors
python - <<'PY'
from scripts.build_anchors import build_anchors
# Build CIFAR100 labels file
from pathlib import Path
from torchvision.datasets import CIFAR100
ds = CIFAR100(root="data", train=True, download=True)
Path("data").mkdir(parents=True, exist_ok=True)
with open("data/cifar100_labels.txt", "w") as f:
    f.write("\n".join(ds.classes))
print("Saved labels")
build_anchors("data/cifar100_labels.txt", "anchors/cifar100_anchors.pkl")
PY

python scripts/train_cvm.py --config configs/cifar100_config.yaml
