import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Input/Feature T-Net from PointNet.

    Predicts a k x k transformation matrix A(x) for transforming features.
    """

    def __init__(self, k: int, conv_channels=(64, 128, 1024), fc_channels=(512, 256), dropout=0.3):
        super().__init__()
        self.k = k

        c1, c2, c3 = conv_channels
        self.conv1 = nn.Conv1d(k, c1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(c3)

        f1, f2 = fc_channels
        self.fc1 = nn.Linear(c3, f1, bias=False)
        self.bn4 = nn.BatchNorm1d(f1)
        self.fc2 = nn.Linear(f1, f2, bias=False)
        self.bn5 = nn.BatchNorm1d(f2)
        self.fc3 = nn.Linear(f2, k * k)

        self.dropout = dropout

        # Initialize fc3 to predict identity at start (helps stability).
        nn.init.zeros_(self.fc3.weight)
        self.fc3.bias.data.copy_(torch.eye(k).reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, k, N]
        Returns:
            Transformation matrix [B, k, k]
        """
        b, _, _ = x.shape

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Symmetric function: global max pooling over points.
        x = torch.max(x, dim=2)[0]  # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)  # [B, k*k]
        trans = x.view(b, self.k, self.k)
        return trans


def tnet_orthogonality_loss(trans: torch.Tensor) -> torch.Tensor:
    """
    Encourage predicted transform matrix A to be orthogonal.

    Loss = ||A A^T - I||_F^2 averaged over batch.
    """
    b, k, _ = trans.shape
    device = trans.device
    iden = torch.eye(k, device=device, dtype=trans.dtype).unsqueeze(0).expand(b, k, k)
    diff = torch.bmm(trans, trans.transpose(1, 2)) - iden
    return (diff ** 2).mean()


class PointNetClassifier(nn.Module):
    """
    PointNet for 3D point cloud classification (ModelNet40-style).
    """

    def __init__(self, num_classes: int = 40, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Input T-Net for 3D coordinates.
        self.input_tnet = TNet(k=3, dropout=dropout)

        # Shared MLP (implemented as Conv1d(1x1)) producing 64, then 128, then 1024 channels.
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        # Feature T-Net for 64D features.
        self.feature_tnet = TNet(k=64, dropout=dropout)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)

        # Classification head.
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = dropout

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            pos: [total_points, 3]
            batch: [total_points] with values in [0, B-1]
        Returns:
            logits: [B, num_classes]
            reg_loss: scalar regularization loss from T-Nets
        """
        # Recover batch size and fixed point count N (ModelNet40 preprocessing uses fixed num_points).
        b = int(batch.max().item()) + 1
        total_points = pos.size(0)
        n = total_points // b
        if total_points % b != 0:
            raise ValueError(f"pos.size(0)={total_points} is not divisible by batch size B={b}.")

        # [B, N, 3] -> [B, 3, N]
        x = pos.view(b, n, 3).transpose(1, 2).contiguous()

        # Input transform.
        trans_in = self.input_tnet(x)  # [B, 3, 3]
        x = torch.bmm(trans_in, x)  # [B, 3, N]

        # Shared MLP level 1 -> 64D features.
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]

        # Feature transform.
        trans_feat = self.feature_tnet(x)  # [B, 64, 64]
        x = torch.bmm(trans_feat, x)  # [B, 64, N]

        # Shared MLP level 2/3 -> 128D, 1024D.
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 1024, N]

        # Global max pooling over points -> [B, 1024]
        x = torch.max(x, dim=2)[0]

        # MLP head.
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc3(x)

        reg_loss = tnet_orthogonality_loss(trans_in) + tnet_orthogonality_loss(trans_feat)
        return logits, reg_loss

