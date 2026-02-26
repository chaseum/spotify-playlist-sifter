from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ml.msd.featurize import featurize
from ml.msd.read import read_track


def _write_tiny_msd_fixture(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        analysis = h5.create_group("analysis")
        metadata = h5.create_group("metadata")
        musicbrainz = h5.create_group("musicbrainz")

        analysis_dtype = np.dtype(
            [
                ("track_id", "S32"),
                ("duration", "<f4"),
                ("tempo", "<f4"),
                ("key", "<i4"),
                ("mode", "<i4"),
                ("time_signature", "<i4"),
                ("loudness", "<f4"),
                ("idx_segments_timbre", "<i4"),
                ("idx_segments_pitches", "<i4"),
            ]
        )
        analysis_songs = np.zeros(2, dtype=analysis_dtype)
        analysis_songs[0]["track_id"] = b"TRTEST000000000001"
        analysis_songs[0]["duration"] = 210.5
        analysis_songs[0]["tempo"] = 120.0
        analysis_songs[0]["key"] = 5
        analysis_songs[0]["mode"] = 1
        analysis_songs[0]["time_signature"] = 4
        analysis_songs[0]["loudness"] = -8.0
        analysis_songs[0]["idx_segments_timbre"] = 0
        analysis_songs[0]["idx_segments_pitches"] = 0
        analysis_songs[1]["idx_segments_timbre"] = 2
        analysis_songs[1]["idx_segments_pitches"] = 3
        analysis.create_dataset("songs", data=analysis_songs)

        segments_timbre = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        analysis.create_dataset("segments_timbre", data=segments_timbre)

        segments_pitches = np.asarray(
            [
                [0.1] * 12,
                [0.4] * 12,
                [0.9] * 12,
                [0.2] * 12,
            ],
            dtype=np.float32,
        )
        analysis.create_dataset("segments_pitches", data=segments_pitches)

        metadata_dtype = np.dtype(
            [
                ("title", "S128"),
                ("artist_name", "S128"),
                ("idx_artist_terms", "<i4"),
            ]
        )
        metadata_songs = np.zeros(2, dtype=metadata_dtype)
        metadata_songs[0]["title"] = b"Tiny Track"
        metadata_songs[0]["artist_name"] = b"Tiny Artist"
        metadata_songs[0]["idx_artist_terms"] = 0
        metadata_songs[1]["idx_artist_terms"] = 2
        metadata.create_dataset("songs", data=metadata_songs)
        metadata.create_dataset("artist_terms", data=np.asarray([b"rock", b"indie", b"jazz", b"blues"]))
        metadata.create_dataset("artist_terms_weight", data=np.asarray([1.2, 0.8, 0.1, 0.2], dtype=np.float32))

        musicbrainz_dtype = np.dtype([("year", "<i4")])
        musicbrainz_songs = np.zeros(2, dtype=musicbrainz_dtype)
        musicbrainz_songs[0]["year"] = 1999
        musicbrainz_songs[1]["year"] = 2001
        musicbrainz.create_dataset("songs", data=musicbrainz_songs)


def test_read_track_reads_expected_fields(tmp_path: Path) -> None:
    fixture_path = tmp_path / "tiny_msd.h5"
    _write_tiny_msd_fixture(fixture_path)

    track = read_track(fixture_path)

    assert set(track.keys()) == {
        "track_id",
        "title",
        "artist_name",
        "duration",
        "year",
        "tempo",
        "key",
        "mode",
        "time_signature",
        "loudness",
        "segments_timbre",
        "segments_pitches",
        "artist_terms",
        "artist_terms_weight",
    }
    assert track["track_id"] == "TRTEST000000000001"
    assert track["title"] == "Tiny Track"
    assert track["artist_name"] == "Tiny Artist"
    assert track["segments_timbre"].shape == (2, 12)
    assert track["segments_pitches"].shape == (3, 12)
    assert track["artist_terms"] == ["rock", "indie"]
    assert track["artist_terms_weight"] == [1.2000000476837158, 0.800000011920929]


def test_featurize_returns_float32_565_and_finite(tmp_path: Path) -> None:
    fixture_path = tmp_path / "tiny_msd.h5"
    _write_tiny_msd_fixture(fixture_path)

    vector = featurize(read_track(fixture_path))

    assert vector.shape == (565,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_featurize_is_deterministic_for_hashed_terms(tmp_path: Path) -> None:
    fixture_path = tmp_path / "tiny_msd.h5"
    _write_tiny_msd_fixture(fixture_path)
    track = read_track(fixture_path)

    vector_a = featurize(track)
    vector_b = featurize(track)

    np.testing.assert_array_equal(vector_a, vector_b)


def test_featurize_handles_empty_and_nonfinite_values() -> None:
    track = {
        "tempo": float("nan"),
        "loudness": float("-inf"),
        "key": 99,
        "mode": -3,
        "time_signature": float("inf"),
        "segments_timbre": np.zeros((0, 12), dtype=np.float32),
        "segments_pitches": np.zeros((0, 12), dtype=np.float32),
        "artist_terms": ["rock", "indie", "electronic"],
        "artist_terms_weight": [1.0, float("nan")],
    }

    vector = featurize(track)

    assert vector.shape == (565,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
