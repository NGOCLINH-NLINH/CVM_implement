# Training Instruction
## 1. Install dependencies
## 2. Training
### 2.1. Training in local
#### A. CIFAR-100
* Download the CIFAR-100 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and extract it to `data/`.
* Build anchors (run for first time training):
```bash
 python scripts/build_anchors.py --labels-file data/cifar100_labels.txt --out anchors/cifar100_anchors.pkl
```
* Train (config files are available in `configs/`):
```bash
 python scripts/train_cvm.py --config configs/[config file name].yaml
```
#### B. Tiny ImageNet
* Download the Tiny ImageNet dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and extract it to `data/`.
* Build anchors (run for first time training):
```bash
 python scripts/build_anchors.py --labels-file data/tiny_imagenet_labels.txt --out anchors/tiny_anchors.pkl
```
* Train (config files are available at `configs/`):
```bash
 python scripts/train_cvm_imgnet.py --config configs/[config file name].yaml
```

### 2.2. Training on Kaggle
#### A. CIFAR-100
* Upload dataset to Kaggle notebook.
* Notebook content:
```bash
# Clone repository
!git clone --branch testtesttest https://github.com/NGOCLINH-NLINH/SCALA.git
%cd SCALA
!pip install -q timm==0.6.13 sentence-transformers

# Prepare data directory
!mkdir -p /kaggle/working/SCALA/data/
!mkdir -p /kaggle/working/data/
!cp -r /kaggle/input/datasets/[kaggle username]/cifar100/cifar-100-python /kaggle/working/SCALA/data/
!cp -r /kaggle/input/datasets/[kaggle username]/cifar100/cifar-100-python /kaggle/working/data/

# Build anchors
!python scripts/build_anchors.py \
  --labels-file data/cifar100_labels.txt \
  --out anchors/cifar100_anchors.pkl

# Prepare config file (write in a seperate cell). For example:
%%writefile /kaggle/working/temp_config.yaml
exp_name: "Test"
seed: 1234
num_tasks: 10
batch_size: 32
epochs_per_task: 50
checkpoints_dir: /kaggle/working/checkpoints
pretrained_backbone: False
anchors_path: anchors/cifar100_anchors.pkl
lr: 0.05  
momentum: 0.9
weight_decay: 0.0001
margin: 0.15                 
k_negs: 9                 
beta: 1.0
out_dim: 384
replay_batch: 32
replay_lambda: 1.0
milestones: [35, 45]
memory_size: 2000
replay_on: True
original_cvm: False
Ld_mode: new
use_active_mean: True
use_all_seen_negs: False
adaptive_margin: True
adaptive_margin_scale: 0.3
max_norm: 30.0
use_cmm: True
lambda_cmm: 1.0
alpha_cmm: 0.4

# Train
!python scripts/train_cvm.py --config /kaggle/working/temp_config.yaml
```

#### B. Tiny ImageNet
* Upload dataset to Kaggle notebook.
* Notebook content:
```bash
# Clone repository
!git clone --branch testtesttest https://github.com/NGOCLINH-NLINH/SCALA.git
%cd SCALA
!pip install -q timm==0.6.13 sentence-transformers

# Prepare data directory
!mkdir -p /kaggle/working/SCALA/data/
!mkdir -p /kaggle/working/data/
!cp -r /kaggle/input/datasets/[kaggle username]/[dataset name]/tiny-imagenet-200 /kaggle/working/SCALA/data/
!cp -r /kaggle/input/datasets/[kaggle username]/[dataset name]/tiny-imagenet-200 /kaggle/working/data/

# Build anchors
!python scripts/build_anchors.py \
  --labels-file data/tiny_imagenet_labels.txt \
  --out anchors/tiny_anchors.pkl

# Prepare config file (write in a seperate cell). For example:
%%writefile /kaggle/working/temp_config.yaml
exp_name: "Test"
seed: 1234
num_tasks: 10
batch_size: 32
epochs_per_task: 50
checkpoints_dir: /kaggle/working/checkpoints
pretrained_backbone: False
anchors_path: anchors/tiny_anchors.pkl
lr: 0.05  
momentum: 0.9
weight_decay: 0.0001
margin: 0.15                 
k_negs: 5                 
beta: 1.0
out_dim: 384
replay_batch: 32
replay_lambda: 1.0
milestones: [35, 45]
memory_size: 2000
replay_on: True
original_cvm: False
Ld_mode: new
use_active_mean: True
use_all_seen_negs: False
adaptive_margin: True
adaptive_margin_scale: 0.3
max_norm: 30.0
use_cmm: True
lambda_cmm: 1.0
alpha_cmm: 0.4

# Train
!python scripts/train_cvm_imgnet.py --config /kaggle/working/temp_config.yaml
```
