from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


AUDIO_DIM = 53
TAGS_DIM = 512
FEATURE_DIM = AUDIO_DIM + TAGS_DIM

TEMPO_IDX = 48
LOUDNESS_IDX = 49
KEY_IDX = 50
MODE_IDX = 51
TS_IDX = 52

_TEMPO_RANGE = (0.0, 250.0)
_LOUDNESS_RANGE = (-60.0, 0.0)
_KEY_RANGE = (0.0, 11.0)
_MODE_RANGE = (0.0, 1.0)
_TS_RANGE = (0.0, 7.0)


def _to_1d_float_tensor(vec: Any) -> torch.Tensor:
    tensor = torch.as_tensor(vec, dtype=torch.float32).reshape(-1)
    if tensor.numel() != FEATURE_DIM:
        raise ValueError(f"Expected vector with {FEATURE_DIM} values, got {tensor.numel()}")
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _split_msd_vec(vec: Any) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = _to_1d_float_tensor(vec)
    return tensor[:AUDIO_DIM], tensor[AUDIO_DIM:]


def _denorm_scalar(normalized: torch.Tensor, low: float, high: float) -> torch.Tensor:
    scaled = normalized * (high - low) + low
    clipped = torch.clamp(scaled, min=low, max=high)
    return torch.round(clipped)


def _binary_match(
    left_audio: torch.Tensor,
    right_audio: torch.Tensor,
    index: int,
    value_range: tuple[float, float],
) -> torch.Tensor:
    left = _denorm_scalar(left_audio[index], *value_range)
    right = _denorm_scalar(right_audio[index], *value_range)
    return torch.tensor(1.0 if bool(left == right) else 0.0, dtype=torch.float32)


def featureize_pair(track_vec: Any, pl_vec: Any) -> torch.Tensor:
    track_audio, track_tags = _split_msd_vec(track_vec)
    pl_audio, pl_tags = _split_msd_vec(pl_vec)

    audio_diff = track_audio - pl_audio
    cos_audio = F.cosine_similarity(track_audio, pl_audio, dim=0, eps=1e-8)
    cos_tags = F.cosine_similarity(track_tags, pl_tags, dim=0, eps=1e-8)

    l1_audio = torch.linalg.vector_norm(audio_diff, ord=1)
    l2_audio = torch.linalg.vector_norm(audio_diff, ord=2)

    tempo_delta = torch.abs(track_audio[TEMPO_IDX] - pl_audio[TEMPO_IDX])
    loudness_delta = torch.abs(track_audio[LOUDNESS_IDX] - pl_audio[LOUDNESS_IDX])

    key_match = _binary_match(track_audio, pl_audio, KEY_IDX, _KEY_RANGE)
    mode_match = _binary_match(track_audio, pl_audio, MODE_IDX, _MODE_RANGE)
    ts_match = _binary_match(track_audio, pl_audio, TS_IDX, _TS_RANGE)

    features = torch.stack(
        [
            cos_audio,
            cos_tags,
            l1_audio,
            l2_audio,
            tempo_delta,
            loudness_delta,
            key_match,
            mode_match,
            ts_match,
        ]
    )
    return features.to(dtype=torch.float32)


class CompatibilityMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
        logits = self.network(x_tensor).squeeze(-1)
        return logits


def binary_acc(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    yhat_flat = torch.as_tensor(yhat, dtype=torch.float32).reshape(-1)
    y_flat = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    if yhat_flat.numel() != y_flat.numel():
        raise ValueError(
            f"Expected predictions and labels to have the same number of elements, got "
            f"{yhat_flat.numel()} and {y_flat.numel()}"
        )
    preds = yhat_flat >= 0.5
    labels = y_flat >= 0.5
    return (preds == labels).to(dtype=torch.float32).mean()


def binary_auroc(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    try:
        from torchmetrics.classification import BinaryAUROC
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "binary_auroc requires torchmetrics. Install it with: pip install torchmetrics"
        ) from exc

    yhat_flat = torch.as_tensor(yhat, dtype=torch.float32).reshape(-1)
    y_flat = torch.as_tensor(y, dtype=torch.int64).reshape(-1)
    if yhat_flat.numel() != y_flat.numel():
        raise ValueError(
            f"Expected predictions and labels to have the same number of elements, got "
            f"{yhat_flat.numel()} and {y_flat.numel()}"
        )
    metric = BinaryAUROC()
    return metric(yhat_flat, y_flat)


__all__ = [
    "AUDIO_DIM",
    "TAGS_DIM",
    "FEATURE_DIM",
    "TEMPO_IDX",
    "LOUDNESS_IDX",
    "KEY_IDX",
    "MODE_IDX",
    "TS_IDX",
    "CompatibilityMLP",
    "featureize_pair",
    "binary_acc",
    "binary_auroc",
]
