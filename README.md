# PointNet 3D Classification (ModelNet40)

PointNet is a classic neural network for learning directly from *unordered* 3D point clouds by applying shared MLPs to each point and using a symmetric aggregation (global max pooling) to obtain a permutation-invariant global feature. This project implements PointNet from scratch in PyTorch and trains it on ModelNet40 using `torch_geometric.datasets.ModelNet`.

## Project Structure
```
pointnet-3d-classification/
├── data/           # data loading utilities
├── models/         # PointNet architecture
├── utils/          # helper functions
├── train.py        # training script
├── evaluate.py     # evaluation script
├── requirements.txt
└── README.md
```

## Architecture (Text / ASCII Diagram)
```
          Point Cloud Input
            x: [N, 3]
                 |
          +---------------+
          |  Input T-Net  |  predicts A_in: [3, 3]
          +---------------+
                 |
         x = A_in * x
                 |
          +---------------+
          | Shared MLP #1 |  Conv1d(1x1): 3 -> 64
          +---------------+
                 |
          +---------------+
          | Feature T-Net |  predicts A_feat: [64, 64]
          +---------------+
                 |
         x = A_feat * x
                 |
          +---------------+
          | Shared MLP #2 |  Conv1d(1x1): 64 -> 128 -> 1024
          +---------------+
                 |
        Global Max Pool over N points
                 |
               [1024]
                 |
          FC(1024->512) -> Dropout
                 |
          FC(512->256)  -> Dropout
                 |
             FC(256->K)
                 |
           logits: [K=40]
```

## How to Install

### 1) Python + virtual environment
Use Python `3.10+`:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install PyTorch
Install a PyTorch build appropriate for your machine (CPU or CUDA) from the official site.

### 3) Install PyTorch Geometric (torch-geometric)
`torch-geometric` has binary dependencies. Follow the official installation guide for your PyTorch version:
https://pytorch-geometric.readthedocs.io/

### 4) Install project dependencies
```bash
pip install -r requirements.txt
```

## Dataset Download + Preprocessing
The code automatically downloads ModelNet40 via `torch_geometric.datasets.ModelNet`.
ModelNet is stored as meshes; we preprocess it into a fixed-size point cloud using:
`torch_geometric.transforms.SamplePoints(num=<num_points>)`

## Train
```bash
python train.py \
  --data-root data/modelnet \
  --num-points 1024 \
  --epochs 50 \
  --lr 0.001 \
  --batch-size 32 \
  --output-dir checkpoints/pointnet-modelnet40
```

During training, the script logs per-epoch training/validation loss and accuracy and saves the best model checkpoint to:
`checkpoints/pointnet-modelnet40/best_model.pth`
It also writes a CSV log to:
`checkpoints/pointnet-modelnet40/training_log.csv`

## Evaluate
```bash
python evaluate.py \
  --data-root data/modelnet \
  --num-points 1024 \
  --checkpoint checkpoints/pointnet-modelnet40/best_model.pth
```

The evaluation prints per-class accuracy and overall test accuracy.

## Results (Example)
PointNet accuracy varies with training settings and preprocessing. After you train, run `evaluate.py` to fill in your exact numbers.

| Num Points | Overall Accuracy (Test) |
|---:|---:|
| 1024 | ~90.0% (example) |

## Why PointNet Matters for 3D Understanding
3D point clouds have no inherent order, but many standard neural layers assume ordered inputs. PointNet solves this by using:
1) shared weights across points (so each point is encoded the same way), and
2) a symmetric pooling function (global max pooling) to aggregate point features into a permutation-invariant representation.

This makes PointNet a foundational architecture for tasks like classification and part segmentation in 3D.

