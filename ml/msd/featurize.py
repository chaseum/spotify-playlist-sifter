from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


HASH_DIM = 512
FEATURE_DIM = 24 + 24 + 5 + HASH_DIM

SCALAR_RANGES: dict[str, tuple[float, float]] = {
    "tempo": (0.0, 250.0),
    "loudness": (-60.0, 0.0),
    "key": (0.0, 11.0),
    "mode": (0.0, 1.0),
    "time_signature": (0.0, 7.0),
}


def _matrix_stats(values: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 12 or arr.shape[0] == 0:
        zeros = np.zeros(12, dtype=np.float32)
        return zeros, zeros

    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = clean.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = clean.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, std


def _normalize_scalar(value: Any, low: float, high: float) -> np.float32:
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = low

    if not np.isfinite(val):
        val = low

    clipped = min(max(val, low), high)
    return np.float32((clipped - low) / (high - low))


def _stable_bucket(term: str) -> int:
    digest = hashlib.blake2b(term.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % HASH_DIM


def _terms_hash_vector(track: dict[str, Any]) -> np.ndarray:
    vector = np.zeros(HASH_DIM, dtype=np.float32)
    terms = track.get("artist_terms") or []
    weights = track.get("artist_terms_weight") or []

    for term, weight in zip(terms, weights):
        term_text = str(term).strip()
        if not term_text:
            continue
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_weight):
            continue
        vector[_stable_bucket(term_text)] += np.float32(numeric_weight)

    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def featurize(track: dict[str, Any]) -> np.ndarray:
    timbre_mean, timbre_std = _matrix_stats(track.get("segments_timbre", np.zeros((0, 12), dtype=np.float32)))
    pitch_mean, pitch_std = _matrix_stats(track.get("segments_pitches", np.zeros((0, 12), dtype=np.float32)))

    scalars = np.asarray(
        [
            _normalize_scalar(track.get("tempo"), *SCALAR_RANGES["tempo"]),
            _normalize_scalar(track.get("loudness"), *SCALAR_RANGES["loudness"]),
            _normalize_scalar(track.get("key"), *SCALAR_RANGES["key"]),
            _normalize_scalar(track.get("mode"), *SCALAR_RANGES["mode"]),
            _normalize_scalar(track.get("time_signature"), *SCALAR_RANGES["time_signature"]),
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            timbre_mean,
            timbre_std,
            pitch_mean,
            pitch_std,
            scalars,
            _terms_hash_vector(track),
        ]
    ).astype(np.float32, copy=False)

    if features.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected feature vector of shape ({FEATURE_DIM},), got {features.shape}")

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
