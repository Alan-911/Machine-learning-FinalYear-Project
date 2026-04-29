"""
CNN-based synthetic speech detector using ResNet18 and LCNN architectures.
Input: log-mel spectrogram treated as a single-channel 2D image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional


# ── LCNN (Light CNN) ────────────────────────────────────────────────────────

class MaxFeatureMap(nn.Module):
    """Max-feature-map activation: splits channels in half, takes element-wise max."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) % 2 == 0, "Channel dim must be even for MFM."
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, stride=stride, padding=pad)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.mfm = MaxFeatureMap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.bn(self.conv(x)))


class LCNN(nn.Module):
    """
    Light CNN for anti-spoofing, following Wu et al. (2018).
    Designed for spectrogram inputs; uses MFM activations to suppress noise.
    """

    def __init__(self, n_mels: int = 80, time_frames: int = 300, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            LCNNBlock(1, 32, kernel=5, stride=1, pad=2),
            nn.MaxPool2d(2, 2),                          # /2
            LCNNBlock(32, 48),
            LCNNBlock(48, 48),
            nn.MaxPool2d(2, 2),                          # /4
            LCNNBlock(48, 64),
            LCNNBlock(64, 64),
            nn.MaxPool2d(2, 2),                          # /8
            LCNNBlock(64, 32),
        )
        self.dropout = nn.Dropout(0.5)

        # Compute flattened size
        dummy = torch.zeros(1, 1, n_mels, time_frames)
        flat_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 160),
            MaxFeatureMap(),           # 160 → 80
            nn.Linear(80, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, time_frames)
        x = self.features(x)
        x = self.dropout(x.view(x.size(0), -1))
        return self.classifier(x)  # (B, 1) — raw logit


# ── ResNet18 (adapted for spectrogram input) ─────────────────────────────────

class ResNetDetector(nn.Module):
    """
    ResNet18 adapted for single-channel spectrogram anti-spoofing.
    Replaces the final FC layer with a binary classification head.
    """

    def __init__(self, pretrained: bool = False, num_classes: int = 1):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace first conv to accept 1-channel input (spectrogram)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace final FC
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 1)


# ── Training utilities ────────────────────────────────────────────────────────

class CNNTrainer:
    """Handles training loop, early stopping, and checkpointing for CNN models."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y in loader:
            X = X.unsqueeze(1).to(self.device)   # add channel dim
            y = y.float().unsqueeze(1).to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader) -> tuple:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        for X, y in loader:
            X = X.unsqueeze(1).to(self.device)
            y = y.float().unsqueeze(1).to(self.device)
            logits = self.model(X)
            total_loss += self.criterion(logits, y).item() * len(y)
            all_logits.append(logits.sigmoid().cpu())
            all_labels.append(y.cpu())
        probs = torch.cat(all_logits).numpy().flatten()
        labels = torch.cat(all_labels).numpy().flatten()
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, probs, labels

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        patience: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.evaluate(val_loader)
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  -> Checkpoint saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if save_path:
            self.model.load_state_dict(torch.load(save_path))
