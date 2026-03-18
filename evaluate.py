from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.dataset import get_modelnet40_datasets
from models.pointnet import PointNetClassifier


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    model.eval()

    num_classes = model.num_classes
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        y_true = batch.y.view(-1).long()

        logits, _ = model(batch.pos, batch.batch)
        y_pred = logits.argmax(dim=1).long()

        for c in range(num_classes):
            mask = y_true == c
            cnt = int(mask.sum().item())
            if cnt == 0:
                continue
            total_per_class[c] += cnt
            correct_per_class[c] += int((y_pred[mask] == c).sum().item())

    overall_correct = int(correct_per_class.sum().item())
    overall_total = int(total_per_class.sum().item())
    overall_acc = overall_correct / max(overall_total, 1)

    acc_per_class = (correct_per_class.float() / total_per_class.clamp(min=1).float()).cpu().numpy()
    return overall_acc, acc_per_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PointNet on ModelNet40")
    parser.add_argument("--data-root", type=str, default="data/modelnet")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--checkpoint", type=str, default="checkpoints/pointnet-modelnet40/best_model.pth")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args: Dict = checkpoint.get("args", {})

    num_classes = int(model_args.get("num_classes", 40))
    dropout = float(model_args.get("dropout", 0.3))

    model = PointNetClassifier(num_classes=num_classes, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_dataset = get_modelnet40_datasets(
        root=args.data_root,
        num_points=args.num_points,
        force_reload=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    overall_acc, per_class_acc = evaluate(model, test_loader, device)
    print(f"\nOverall test accuracy: {overall_acc * 100:.2f}%")
    print("\nPer-class accuracy (class_id: accuracy%):")
    for i, a in enumerate(per_class_acc.tolist()):
        print(f"{i}: {a * 100:.2f}%")


if __name__ == "__main__":
    main()

