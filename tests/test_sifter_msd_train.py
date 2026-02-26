from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from ml.sifter_msd.model import AUDIO_DIM, FEATURE_DIM
from ml.sifter_msd.train import main


def _make_synthetic_vectors(num_tracks: int = 120, num_clusters: int = 4) -> np.ndarray:
    rng = np.random.default_rng(123)
    vectors = np.zeros((num_tracks, FEATURE_DIM), dtype=np.float32)
    centers = rng.normal(loc=0.0, scale=1.0, size=(num_clusters, AUDIO_DIM)).astype(np.float32)
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.where(center_norms > 0.0, center_norms, 1.0)

    for i in range(num_tracks):
        cluster_id = i % num_clusters
        audio = centers[cluster_id] + rng.normal(loc=0.0, scale=0.05, size=AUDIO_DIM).astype(np.float32)
        tags = rng.normal(loc=0.0, scale=0.01, size=(FEATURE_DIM - AUDIO_DIM)).astype(np.float32)
        vectors[i, :AUDIO_DIM] = audio
        vectors[i, AUDIO_DIM:] = tags

    return vectors


def _is_formatted_metric(value: str) -> bool:
    if value == "nan":
        return True
    return bool(re.fullmatch(r"-?\d+\.\d{6}", value))


def test_train_cli_happy_path_writes_checkpoint_and_epoch_lines(
    tmp_path: Path,
    capsys,
) -> None:
    vecs_path = tmp_path / "msd_vecs.npy"
    out_path = tmp_path / "sifter_msd.pt"
    np.save(vecs_path, _make_synthetic_vectors())

    code = main(
        [
            "--vecs",
            str(vecs_path),
            "--out",
            str(out_path),
            "--epochs",
            "2",
            "--batch",
            "64",
            "--k",
            "4",
            "--neg_ratio",
            "2",
            "--seed",
            "7",
        ]
    )

    assert code == 0
    assert out_path.exists()

    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert output_lines
    assert output_lines[0].startswith("Device: ")

    epoch_lines = [line for line in output_lines if line.startswith("Epoch ")]
    assert len(epoch_lines) == 2

    for i, line in enumerate(epoch_lines, start=1):
        parts = line.split(" | ")
        assert len(parts) == 6
        assert parts[0] == f"Epoch {i}/2"
        expected_keys = ("train_loss", "train_acc", "val_loss", "val_acc", "val_auc")
        for part, key in zip(parts[1:], expected_keys):
            assert part.startswith(f"{key} ")
            metric_text = part.split(" ", maxsplit=1)[1]
            assert _is_formatted_metric(metric_text)
            metric_value = float(metric_text)
            if key != "val_auc" or not math.isnan(metric_value):
                assert math.isfinite(metric_value)

    checkpoint = torch.load(out_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "kmeans_centroids" in checkpoint
    assert "config_json" in checkpoint

    centroids = checkpoint["kmeans_centroids"]
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.detach().cpu().numpy()
    assert tuple(centroids.shape) == (4, AUDIO_DIM)
    assert centroids.dtype == np.float32

    config = json.loads(checkpoint["config_json"])
    assert config["epochs"] == 2
    assert config["batch"] == 64
    assert config["k"] == 4
    assert config["neg_ratio"] == 2


def test_train_cli_rejects_invalid_vector_shape(tmp_path: Path) -> None:
    vecs_path = tmp_path / "bad_vecs.npy"
    out_path = tmp_path / "should_not_exist.pt"
    np.save(vecs_path, np.zeros((10,), dtype=np.float32))

    code = main(["--vecs", str(vecs_path), "--out", str(out_path)])

    assert code == 1
    assert not out_path.exists()
