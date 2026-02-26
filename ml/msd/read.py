from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode_text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _slice_by_song_index(
    songs_dataset: h5py.Dataset,
    data_dataset: h5py.Dataset,
    idx_field: str,
    song_index: int,
) -> np.ndarray:
    if len(songs_dataset) == 0:
        return np.asarray(data_dataset[:0])

    start = int(songs_dataset[song_index][idx_field])
    if song_index + 1 < len(songs_dataset):
        end = int(songs_dataset[song_index + 1][idx_field])
    else:
        end = int(len(data_dataset))

    start = max(0, min(start, len(data_dataset)))
    end = max(start, min(end, len(data_dataset)))
    return np.asarray(data_dataset[start:end])


def _to_matrix_12(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 12), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 12:
        return arr

    flat = arr.reshape(-1)
    rows = flat.size // 12
    if rows == 0:
        return np.zeros((0, 12), dtype=np.float32)
    return flat[: rows * 12].reshape(rows, 12)


def _to_float_list(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return [float(x) for x in arr.tolist()]


def _to_text_list(values: np.ndarray) -> list[str]:
    return [_decode_text(item) for item in np.asarray(values).reshape(-1)]


def read_track(h5_path: str | Path) -> dict[str, Any]:
    path = Path(h5_path)
    with h5py.File(path, "r") as h5:
        analysis_songs = h5["analysis"]["songs"]
        metadata_songs = h5["metadata"]["songs"]
        musicbrainz_songs = h5["musicbrainz"]["songs"]

        if len(analysis_songs) == 0 or len(metadata_songs) == 0 or len(musicbrainz_songs) == 0:
            raise ValueError(f"HDF5 file has no song rows: {path}")

        song_index = 0

        analysis_song = analysis_songs[song_index]
        metadata_song = metadata_songs[song_index]
        musicbrainz_song = musicbrainz_songs[song_index]

        segments_timbre = _slice_by_song_index(
            analysis_songs,
            h5["analysis"]["segments_timbre"],
            "idx_segments_timbre",
            song_index,
        )
        segments_pitches = _slice_by_song_index(
            analysis_songs,
            h5["analysis"]["segments_pitches"],
            "idx_segments_pitches",
            song_index,
        )
        artist_terms = _slice_by_song_index(
            metadata_songs,
            h5["metadata"]["artist_terms"],
            "idx_artist_terms",
            song_index,
        )
        artist_terms_weight = _slice_by_song_index(
            metadata_songs,
            h5["metadata"]["artist_terms_weight"],
            "idx_artist_terms",
            song_index,
        )

        return {
            "track_id": _decode_text(analysis_song["track_id"]),
            "title": _decode_text(metadata_song["title"]),
            "artist_name": _decode_text(metadata_song["artist_name"]),
            "duration": float(analysis_song["duration"]),
            "year": int(musicbrainz_song["year"]),
            "tempo": float(analysis_song["tempo"]),
            "key": int(analysis_song["key"]),
            "mode": int(analysis_song["mode"]),
            "time_signature": int(analysis_song["time_signature"]),
            "loudness": float(analysis_song["loudness"]),
            "segments_timbre": _to_matrix_12(segments_timbre),
            "segments_pitches": _to_matrix_12(segments_pitches),
            "artist_terms": _to_text_list(artist_terms),
            "artist_terms_weight": _to_float_list(artist_terms_weight),
        }
