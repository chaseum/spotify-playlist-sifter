from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.sifter_msd.model import (
    AUDIO_DIM,
    FEATURE_DIM,
    KEY_IDX,
    LOUDNESS_IDX,
    MODE_IDX,
    TEMPO_IDX,
    TS_IDX,
    CompatibilityMLP,
)


LOGGER = logging.getLogger(__name__)

_KEY_RANGE = (0.0, 11.0)
_MODE_RANGE = (0.0, 1.0)
_TS_RANGE = (0.0, 7.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sifter MSD compatibility model.")
    parser.add_argument("--vecs", required=True, help="Path to msd_vecs.npy with shape (N, 565).")
    parser.add_argument("--out", required=True, help="Path to save checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size.")
    parser.add_argument("--k", type=int, default=200, help="Number of audio KMeans clusters.")
    parser.add_argument("--neg_ratio", type=int, default=5, help="Number of negatives per positive.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation split fraction.")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got ndim={arr.ndim}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalized = arr / safe_norms
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def build_split_indices(
    n_tracks: int,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if n_tracks <= 1:
        raise ValueError("Need at least 2 tracks for train/val split.")
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")

    val_count = max(1, int(n_tracks * val_frac))
    if val_count >= n_tracks:
        val_count = n_tracks - 1
    perm = rng.permutation(n_tracks).astype(np.int64)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return train_idx, val_idx


def sample_negative_clusters(
    positive_clusters: np.ndarray,
    k: int,
    neg_ratio: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if k < 2:
        raise ValueError("k must be at least 2 for negative sampling.")
    if neg_ratio < 1:
        raise ValueError("neg_ratio must be >= 1.")

    pos = np.asarray(positive_clusters, dtype=np.int64).reshape(-1)
    random_ids = rng.integers(0, k - 1, size=(pos.size, neg_ratio), dtype=np.int64)
    negatives = random_ids + (random_ids >= pos[:, None])
    return negatives


def build_pair_index_arrays(
    track_indices: np.ndarray,
    labels: np.ndarray,
    k: int,
    neg_ratio: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tracks = np.asarray(track_indices, dtype=np.int64).reshape(-1)
    if tracks.size == 0:
        raise ValueError("track_indices must be non-empty.")

    cluster_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    positive_clusters = cluster_labels[tracks]

    pos_track_idx = tracks
    pos_cluster_idx = positive_clusters
    pos_y = np.ones(pos_track_idx.shape[0], dtype=np.float32)

    neg_clusters = sample_negative_clusters(positive_clusters, k=k, neg_ratio=neg_ratio, rng=rng)
    neg_track_idx = np.repeat(tracks, neg_ratio)
    neg_cluster_idx = neg_clusters.reshape(-1)
    neg_y = np.zeros(neg_track_idx.shape[0], dtype=np.float32)

    track_idx = np.concatenate([pos_track_idx, neg_track_idx], axis=0)
    cluster_idx = np.concatenate([pos_cluster_idx, neg_cluster_idx], axis=0)
    labels_out = np.concatenate([pos_y, neg_y], axis=0)
    return track_idx, cluster_idx, labels_out


def _denorm_and_round(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    scaled = values * (high - low) + low
    clipped = torch.clamp(scaled, min=low, max=high)
    return torch.round(clipped)


def featureize_pair_batch(
    track_batch_raw: torch.Tensor,
    proto_batch_raw: torch.Tensor,
    track_batch_norm: torch.Tensor,
    proto_batch_norm: torch.Tensor,
) -> torch.Tensor:
    if track_batch_raw.ndim != 2 or proto_batch_raw.ndim != 2:
        raise ValueError("track_batch_raw and proto_batch_raw must be 2D tensors.")
    if track_batch_norm.ndim != 2 or proto_batch_norm.ndim != 2:
        raise ValueError("track_batch_norm and proto_batch_norm must be 2D tensors.")
    if track_batch_raw.shape != proto_batch_raw.shape:
        raise ValueError("track_batch_raw and proto_batch_raw must have matching shapes.")
    if track_batch_norm.shape != proto_batch_norm.shape:
        raise ValueError("track_batch_norm and proto_batch_norm must have matching shapes.")
    if track_batch_raw.shape != track_batch_norm.shape:
        raise ValueError("raw and norm batches must have the same shape.")
    if track_batch_raw.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected vectors with {FEATURE_DIM} features.")

    track_audio_raw = track_batch_raw[:, :AUDIO_DIM]
    proto_audio_raw = proto_batch_raw[:, :AUDIO_DIM]

    track_audio = track_batch_norm[:, :AUDIO_DIM]
    track_tags = track_batch_norm[:, AUDIO_DIM:]
    proto_audio = proto_batch_norm[:, :AUDIO_DIM]
    proto_tags = proto_batch_norm[:, AUDIO_DIM:]

    cos_audio = torch.nn.functional.cosine_similarity(track_audio, proto_audio, dim=1, eps=1e-8)
    cos_tags = torch.nn.functional.cosine_similarity(track_tags, proto_tags, dim=1, eps=1e-8)

    audio_diff = track_audio - proto_audio
    l1_audio = torch.linalg.vector_norm(audio_diff, ord=1, dim=1)
    l2_audio = torch.linalg.vector_norm(audio_diff, ord=2, dim=1)

    tempo_delta = torch.abs(track_audio_raw[:, TEMPO_IDX] - proto_audio_raw[:, TEMPO_IDX])
    loudness_delta = torch.abs(track_audio_raw[:, LOUDNESS_IDX] - proto_audio_raw[:, LOUDNESS_IDX])

    key_match = (
        _denorm_and_round(track_audio_raw[:, KEY_IDX], *_KEY_RANGE)
        == _denorm_and_round(proto_audio_raw[:, KEY_IDX], *_KEY_RANGE)
    ).to(dtype=torch.float32)
    mode_match = (
        _denorm_and_round(track_audio_raw[:, MODE_IDX], *_MODE_RANGE)
        == _denorm_and_round(proto_audio_raw[:, MODE_IDX], *_MODE_RANGE)
    ).to(dtype=torch.float32)
    ts_match = (
        _denorm_and_round(track_audio_raw[:, TS_IDX], *_TS_RANGE)
        == _denorm_and_round(proto_audio_raw[:, TS_IDX], *_TS_RANGE)
    ).to(dtype=torch.float32)

    features = torch.stack(
        [cos_audio, cos_tags, l1_audio, l2_audio, tempo_delta, loudness_delta, key_match, mode_match, ts_match],
        dim=1,
    )
    return features.to(dtype=torch.float32)


def _build_loader(
    track_idx: np.ndarray,
    cluster_idx: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(track_idx, dtype=np.int64)),
        torch.from_numpy(np.asarray(cluster_idx, dtype=np.int64)),
        torch.from_numpy(np.asarray(labels, dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: CompatibilityMLP,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    vectors_raw: torch.Tensor,
    vectors_norm: torch.Tensor,
    prototypes_raw: torch.Tensor,
    prototypes_norm: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for track_idx, cluster_idx, labels in loader:
        track_batch_raw = vectors_raw[track_idx].to(device=device, dtype=torch.float32)
        track_batch_norm = vectors_norm[track_idx].to(device=device, dtype=torch.float32)
        proto_batch_raw = prototypes_raw[cluster_idx].to(device=device, dtype=torch.float32)
        proto_batch_norm = prototypes_norm[cluster_idx].to(device=device, dtype=torch.float32)
        labels_batch = labels.to(device=device, dtype=torch.float32)

        features = featureize_pair_batch(track_batch_raw, proto_batch_raw, track_batch_norm, proto_batch_norm)
        logits = model(features)
        loss = criterion(logits, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = int(labels_batch.numel())
        probs = torch.sigmoid(logits)
        preds = probs >= 0.5
        gold = labels_batch >= 0.5
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == gold).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples


def evaluate(
    model: CompatibilityMLP,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    vectors_raw: torch.Tensor,
    vectors_norm: torch.Tensor,
    prototypes_raw: torch.Tensor,
    prototypes_norm: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for track_idx, cluster_idx, labels in loader:
            track_batch_raw = vectors_raw[track_idx].to(device=device, dtype=torch.float32)
            track_batch_norm = vectors_norm[track_idx].to(device=device, dtype=torch.float32)
            proto_batch_raw = prototypes_raw[cluster_idx].to(device=device, dtype=torch.float32)
            proto_batch_norm = prototypes_norm[cluster_idx].to(device=device, dtype=torch.float32)
            labels_batch = labels.to(device=device, dtype=torch.float32)

            features = featureize_pair_batch(track_batch_raw, proto_batch_raw, track_batch_norm, proto_batch_norm)
            logits = model(features)
            loss = criterion(logits, labels_batch)

            batch_size = int(labels_batch.numel())
            probs = torch.sigmoid(logits)
            preds = probs >= 0.5
            gold = labels_batch >= 0.5
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == gold).sum().item())
            total_examples += batch_size

            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels_batch.detach().cpu().numpy())

    if total_examples == 0:
        return 0.0, 0.0, np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    probs_flat = np.concatenate(all_probs).astype(np.float32, copy=False)
    labels_flat = np.concatenate(all_labels).astype(np.float32, copy=False)
    return total_loss / total_examples, total_correct / total_examples, probs_flat, labels_flat


def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(labels, dtype=np.float32).reshape(-1)
    y_score = np.asarray(probs, dtype=np.float32).reshape(-1)
    if y_true.size == 0 or y_score.size == 0:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _validate_args(args: argparse.Namespace, n_tracks: int) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch <= 0:
        raise ValueError("--batch must be > 0")
    if args.neg_ratio < 1:
        raise ValueError("--neg_ratio must be >= 1")
    if args.k < 2:
        raise ValueError("--k must be >= 2")
    if args.k > n_tracks:
        raise ValueError(f"--k must be <= number of tracks ({n_tracks})")
    if args.lr <= 0.0:
        raise ValueError("--lr must be > 0")
    if not 0.0 < args.val_frac < 1.0:
        raise ValueError("--val_frac must be between 0 and 1")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    try:
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device.type}")

        vecs_path = Path(args.vecs).expanduser().resolve()
        out_path = Path(args.out).expanduser().resolve()

        if not vecs_path.exists():
            raise FileNotFoundError(f"Vector file does not exist: {vecs_path}")

        matrix = np.asarray(np.load(vecs_path), dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected vectors with shape (N, {FEATURE_DIM}), got ndim={matrix.ndim}")
        if matrix.shape[0] <= 1:
            raise ValueError("Need at least 2 rows in vector matrix.")
        if matrix.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected vectors with {FEATURE_DIM} columns, got {matrix.shape[1]}")

        _validate_args(args, n_tracks=int(matrix.shape[0]))

        vectors_raw = np.asarray(matrix, dtype=np.float32)
        vectors_norm = l2_normalize_rows(vectors_raw)
        audio_vectors = vectors_norm[:, :AUDIO_DIM]

        kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(audio_vectors)
        centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

        rng = np.random.default_rng(args.seed)
        prototypes_raw = np.zeros((args.k, FEATURE_DIM), dtype=np.float32)
        for c in range(args.k):
            members = vectors_raw[cluster_labels == c]
            if members.shape[0] == 0:
                prototypes_raw[c, :AUDIO_DIM] = centroids[c]
                continue
            prototypes_raw[c, :AUDIO_DIM] = members[:, :AUDIO_DIM].mean(axis=0)
            prototypes_raw[c, AUDIO_DIM:] = members[:, AUDIO_DIM:].mean(axis=0)
        prototypes_norm = l2_normalize_rows(prototypes_raw)

        train_tracks, val_tracks = build_split_indices(vectors_raw.shape[0], args.val_frac, rng)
        train_track_idx, train_cluster_idx, train_y = build_pair_index_arrays(
            train_tracks, cluster_labels, args.k, args.neg_ratio, rng
        )
        val_track_idx, val_cluster_idx, val_y = build_pair_index_arrays(
            val_tracks, cluster_labels, args.k, args.neg_ratio, rng
        )

        train_loader = _build_loader(train_track_idx, train_cluster_idx, train_y, batch_size=args.batch, shuffle=True)
        val_loader = _build_loader(val_track_idx, val_cluster_idx, val_y, batch_size=args.batch, shuffle=False)

        vectors_raw_tensor = torch.from_numpy(vectors_raw)
        vectors_norm_tensor = torch.from_numpy(vectors_norm)
        prototypes_raw_tensor = torch.from_numpy(prototypes_raw)
        prototypes_norm_tensor = torch.from_numpy(prototypes_norm)

        model = CompatibilityMLP(in_dim=9).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                vectors_raw=vectors_raw_tensor,
                vectors_norm=vectors_norm_tensor,
                prototypes_raw=prototypes_raw_tensor,
                prototypes_norm=prototypes_norm_tensor,
                device=device,
            )
            val_loss, val_acc, val_probs, val_labels = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                vectors_raw=vectors_raw_tensor,
                vectors_norm=vectors_norm_tensor,
                prototypes_raw=prototypes_raw_tensor,
                prototypes_norm=prototypes_norm_tensor,
                device=device,
            )
            val_auc = compute_auc(val_labels, val_probs)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss {train_loss:.6f} | "
                f"train_acc {train_acc:.6f} | "
                f"val_loss {val_loss:.6f} | "
                f"val_acc {val_acc:.6f} | "
                f"val_auc {val_auc:.6f}"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "vecs": str(vecs_path),
            "out": str(out_path),
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "k": int(args.k),
            "neg_ratio": int(args.neg_ratio),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "val_frac": float(args.val_frac),
            "feature_dim": int(FEATURE_DIM),
            "audio_dim": int(AUDIO_DIM),
            "num_tracks": int(vectors_raw.shape[0]),
            "train_tracks": int(train_tracks.shape[0]),
            "val_tracks": int(val_tracks.shape[0]),
            "train_pairs": int(train_y.shape[0]),
            "val_pairs": int(val_y.shape[0]),
            "device": device.type,
        }
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "kmeans_centroids": torch.from_numpy(centroids.astype(np.float32, copy=False)),
            "config_json": json.dumps(config_payload, sort_keys=True),
        }
        torch.save(checkpoint, out_path)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
