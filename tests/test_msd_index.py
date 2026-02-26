from __future__ import annotations

from pathlib import Path

import faiss
import h5py
import numpy as np
import pytest

from ml.msd import index as msd_index
from ml.msd.featurize import featurize
from ml.msd.read import read_track


def _write_tiny_track_h5(
    path: Path,
    *,
    track_id: str = "TRTEST000000000001",
    title: str = "Tiny Track",
    artist: str = "Tiny Artist",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        analysis_songs = np.zeros(1, dtype=analysis_dtype)
        analysis_songs[0]["track_id"] = track_id.encode("utf-8")
        analysis_songs[0]["duration"] = 210.5
        analysis_songs[0]["tempo"] = 120.0
        analysis_songs[0]["key"] = 5
        analysis_songs[0]["mode"] = 1
        analysis_songs[0]["time_signature"] = 4
        analysis_songs[0]["loudness"] = -8.0
        analysis_songs[0]["idx_segments_timbre"] = 0
        analysis_songs[0]["idx_segments_pitches"] = 0
        analysis.create_dataset("songs", data=analysis_songs)
        analysis.create_dataset(
            "segments_timbre",
            data=np.asarray(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ],
                dtype=np.float32,
            ),
        )
        analysis.create_dataset(
            "segments_pitches",
            data=np.asarray([[0.1] * 12, [0.4] * 12, [0.9] * 12], dtype=np.float32),
        )

        metadata_dtype = np.dtype([("title", "S128"), ("artist_name", "S128"), ("idx_artist_terms", "<i4")])
        metadata_songs = np.zeros(1, dtype=metadata_dtype)
        metadata_songs[0]["title"] = title.encode("utf-8")
        metadata_songs[0]["artist_name"] = artist.encode("utf-8")
        metadata_songs[0]["idx_artist_terms"] = 0
        metadata.create_dataset("songs", data=metadata_songs)
        metadata.create_dataset("artist_terms", data=np.asarray([b"rock"]))
        metadata.create_dataset("artist_terms_weight", data=np.asarray([1.0], dtype=np.float32))

        musicbrainz_dtype = np.dtype([("year", "<i4")])
        musicbrainz_songs = np.zeros(1, dtype=musicbrainz_dtype)
        musicbrainz_songs[0]["year"] = 1999
        musicbrainz.create_dataset("songs", data=musicbrainz_songs)


def test_l2_normalize_handles_nonzero_and_zero_rows() -> None:
    matrix = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = msd_index.l2_normalize(matrix)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalized[0]), 1.0, atol=1e-6)
    np.testing.assert_array_equal(normalized[1], np.zeros(2, dtype=np.float32))
    assert np.isfinite(normalized).all()


def test_build_happy_path_writes_faiss(tmp_path: Path) -> None:
    vecs = np.asarray([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]], dtype=np.float32)
    vecs_path = tmp_path / "vecs.npy"
    out_index = tmp_path / "msd.faiss"
    np.save(vecs_path, vecs)

    code = msd_index.main(["build", "--vecs", str(vecs_path), "--out", str(out_index)])

    assert code == 0
    assert out_index.exists()
    index = faiss.read_index(str(out_index))
    assert index.ntotal == 2
    assert index.d == 3


def test_query_vec_self_is_top1_and_near_one() -> None:
    matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(msd_index.l2_normalize(matrix))

    row_ids, scores = msd_index.query_vec(matrix[2], index, 3)

    assert int(row_ids[0]) == 2
    assert float(scores[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(scores[0]) == pytest.approx(float(np.max(scores)), abs=1e-6)


def test_query_cli_happy_path_prints_rank_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h5_path = tmp_path / "tiny.h5"
    _write_tiny_track_h5(h5_path)

    track_vec = featurize(read_track(h5_path))
    vecs_path = tmp_path / "vecs.npy"
    out_index = tmp_path / "msd.faiss"
    np.save(vecs_path, np.asarray([track_vec], dtype=np.float32))

    build_code = msd_index.main(["build", "--vecs", str(vecs_path), "--out", str(out_index)])
    assert build_code == 0

    query_code = msd_index.main(["query", "--h5", str(h5_path), "--k", "5", "--index", str(out_index)])
    assert query_code == 0
    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(output_lines) == 1
    assert output_lines[0].startswith("rank=1 row_id=0 score=")
    score_text = output_lines[0].split("score=", maxsplit=1)[1]
    assert float(score_text) == pytest.approx(1.0, abs=1e-6)


def test_build_returns_error_for_missing_or_invalid_vectors(tmp_path: Path) -> None:
    out_index = tmp_path / "msd.faiss"

    missing_code = msd_index.main(["build", "--vecs", str(tmp_path / "missing.npy"), "--out", str(out_index)])
    assert missing_code == 1

    bad_vecs_path = tmp_path / "bad.npy"
    np.save(bad_vecs_path, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    invalid_code = msd_index.main(["build", "--vecs", str(bad_vecs_path), "--out", str(out_index)])
    assert invalid_code == 1


def test_query_returns_error_when_index_missing(tmp_path: Path) -> None:
    h5_path = tmp_path / "tiny.h5"
    _write_tiny_track_h5(h5_path)

    code = msd_index.main(["query", "--h5", str(h5_path), "--k", "5", "--index", str(tmp_path / "missing.faiss")])

    assert code == 1
