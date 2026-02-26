from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from ml.sifter_msd.model import (
    AUDIO_DIM,
    FEATURE_DIM,
    KEY_IDX,
    LOUDNESS_IDX,
    MODE_IDX,
    TAGS_DIM,
    TEMPO_IDX,
    TS_IDX,
    CompatibilityMLP,
    binary_acc,
    binary_auroc,
    featureize_pair,
)


def _make_vec(audio_overrides: dict[int, float] | None = None, tag_overrides: dict[int, float] | None = None) -> torch.Tensor:
    vec = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    if audio_overrides:
        for idx, value in audio_overrides.items():
            vec[idx] = float(value)
    if tag_overrides:
        for idx, value in tag_overrides.items():
            vec[AUDIO_DIM + idx] = float(value)
    return vec


def test_featureize_pair_returns_float32_len_9_and_finite() -> None:
    track = _make_vec(audio_overrides={0: 1.0, TEMPO_IDX: 0.5}, tag_overrides={0: 1.0, 1: 2.0})
    playlist = _make_vec(audio_overrides={0: 1.0, TEMPO_IDX: 0.2}, tag_overrides={0: 1.0, 2: 1.0})

    features = featureize_pair(track, playlist)

    assert features.shape == (9,)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_featureize_pair_computes_expected_semantics() -> None:
    track = _make_vec(
        audio_overrides={
            0: 1.0,
            TEMPO_IDX: 120.0 / 250.0,
            LOUDNESS_IDX: (-20.0 + 60.0) / 60.0,
            KEY_IDX: 5.0 / 11.0,
            MODE_IDX: 1.0,
            TS_IDX: 4.0 / 7.0,
        },
        tag_overrides={0: 1.0, 1: 0.0},
    )
    playlist = _make_vec(
        audio_overrides={
            0: 1.0,
            TEMPO_IDX: 130.0 / 250.0,
            LOUDNESS_IDX: (-10.0 + 60.0) / 60.0,
            KEY_IDX: 5.0 / 11.0,
            MODE_IDX: 0.0,
            TS_IDX: 4.0 / 7.0,
        },
        tag_overrides={0: 0.0, 1: 1.0},
    )

    features = featureize_pair(track, playlist)

    track_audio = track[:AUDIO_DIM]
    playlist_audio = playlist[:AUDIO_DIM]
    track_tags = track[AUDIO_DIM:]
    playlist_tags = playlist[AUDIO_DIM:]
    audio_diff = track_audio - playlist_audio

    expected_cos_audio = F.cosine_similarity(track_audio, playlist_audio, dim=0, eps=1e-8)
    expected_cos_tags = F.cosine_similarity(track_tags, playlist_tags, dim=0, eps=1e-8)
    expected_l1 = torch.linalg.vector_norm(audio_diff, ord=1)
    expected_l2 = torch.linalg.vector_norm(audio_diff, ord=2)

    assert torch.isclose(features[0], expected_cos_audio, atol=1e-6)
    assert torch.isclose(features[1], expected_cos_tags, atol=1e-6)
    assert torch.isclose(features[2], expected_l1, atol=1e-6)
    assert torch.isclose(features[3], expected_l2, atol=1e-6)
    assert features[4] == pytest.approx(abs((120.0 / 250.0) - (130.0 / 250.0)), abs=1e-8)
    assert features[5] == pytest.approx(abs((40.0 / 60.0) - (50.0 / 60.0)), abs=1e-6)
    assert features[6] == 1.0
    assert features[7] == 0.0
    assert features[8] == 1.0


def test_featureize_pair_rejects_wrong_vector_length() -> None:
    with pytest.raises(ValueError, match=f"Expected vector with {FEATURE_DIM} values"):
        featureize_pair(torch.zeros(FEATURE_DIM - 1), torch.zeros(FEATURE_DIM))


def test_model_forward_single_and_batch() -> None:
    model = CompatibilityMLP(in_dim=9)

    single = model(torch.zeros(9, dtype=torch.float32))
    batch = model(torch.zeros((4, 9), dtype=torch.float32))

    assert single.numel() == 1
    assert batch.shape == (4,)
    assert torch.isfinite(single).all()
    assert torch.isfinite(batch).all()


def test_binary_acc_returns_expected_mean() -> None:
    yhat = torch.tensor([0.2, 0.6, 0.8, 0.49], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    score = binary_acc(yhat, y)

    assert score.dtype == torch.float32
    assert score.item() == pytest.approx(0.75, abs=1e-8)


def test_binary_auroc_optional_dependency_behavior() -> None:
    yhat = torch.tensor([0.1, 0.4, 0.35, 0.8], dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="requires torchmetrics"):
            binary_auroc(yhat, y)
    else:
        score = binary_auroc(yhat, y)
        assert isinstance(score, torch.Tensor)
        assert score.ndim == 0
        assert math.isfinite(float(score))
        assert 0.0 <= float(score) <= 1.0


def test_constants_match_msd_contract() -> None:
    assert AUDIO_DIM == 53
    assert TAGS_DIM == 512
    assert FEATURE_DIM == 565
    assert TEMPO_IDX == 48
    assert LOUDNESS_IDX == 49
    assert KEY_IDX == 50
    assert MODE_IDX == 51
    assert TS_IDX == 52
