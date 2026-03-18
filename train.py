from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.dataset import get_modelnet40_datasets
from models.pointnet import PointNetClassifier
from utils.checkpoint import save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    tnet_reg_weight: float,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        y = batch.y.view(-1).long()

        logits, reg_loss = model(batch.pos, batch.batch)
        ce_loss = criterion(logits, y)
        loss = ce_loss + tnet_reg_weight * reg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += bs

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tnet_reg_weight: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        batch = batch.to(device)
        y = batch.y.view(-1).long()

        logits, reg_loss = model(batch.pos, batch.batch)
        ce_loss = criterion(logits, y)
        loss = ce_loss + tnet_reg_weight * reg_loss

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += bs

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="PointNet classification on ModelNet40")
    parser.add_argument("--data-root", type=str, default="data/modelnet", help="Dataset root directory")
    parser.add_argument("--num-points", type=int, default=1024, help="Number of points sampled per shape")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--tnet-reg-weight", type=float, default=1e-3, help="Orthogonality regularization weight")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-reload", action="store_true", help="Re-process dataset even if cached")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    parser.add_argument("--output-dir", type=str, default="checkpoints/pointnet-modelnet40")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = get_modelnet40_datasets(
        root=args.data_root,
        num_points=args.num_points,
        force_reload=args.force_reload,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = PointNetClassifier(num_classes=args.num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_best_path = os.path.join(args.output_dir, "best_model.pth")
    log_path = os.path.join(args.output_dir, "training_log.csv")

    # Initialize CSV log.
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"])

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            tnet_reg_weight=args.tnet_reg_weight,
        )
        val_loss, val_acc = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            tnet_reg_weight=args.tnet_reg_weight,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                ckpt_best_path,
            )

        print(
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
            f"Best val acc: {best_val_acc:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, best_val_acc])

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

