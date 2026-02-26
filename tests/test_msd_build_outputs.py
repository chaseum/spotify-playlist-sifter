from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import h5py
import numpy as np

from ml.msd import build_db, build_vectors


def _write_tiny_track_h5(
    path: Path,
    *,
    track_id: str,
    title: str,
    artist: str,
    duration: float = 210.5,
    year: int = 1999,
    tempo: float = 120.0,
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
        analysis_songs[0]["duration"] = duration
        analysis_songs[0]["tempo"] = tempo
        analysis_songs[0]["key"] = 5
        analysis_songs[0]["mode"] = 1
        analysis_songs[0]["time_signature"] = 4
        analysis_songs[0]["loudness"] = -8.0
        analysis_songs[0]["idx_segments_timbre"] = 0
        analysis_songs[0]["idx_segments_pitches"] = 0
        analysis.create_dataset("songs", data=analysis_songs)

        segments_timbre = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        )
        segments_pitches = np.asarray(
            [
                [0.1] * 12,
                [0.4] * 12,
                [0.9] * 12,
            ],
            dtype=np.float32,
        )
        analysis.create_dataset("segments_timbre", data=segments_timbre)
        analysis.create_dataset("segments_pitches", data=segments_pitches)

        metadata_dtype = np.dtype(
            [
                ("title", "S128"),
                ("artist_name", "S128"),
                ("idx_artist_terms", "<i4"),
            ]
        )
        metadata_songs = np.zeros(1, dtype=metadata_dtype)
        metadata_songs[0]["title"] = title.encode("utf-8")
        metadata_songs[0]["artist_name"] = artist.encode("utf-8")
        metadata_songs[0]["idx_artist_terms"] = 0
        metadata.create_dataset("songs", data=metadata_songs)
        metadata.create_dataset("artist_terms", data=np.asarray([b"rock"]))
        metadata.create_dataset("artist_terms_weight", data=np.asarray([1.0], dtype=np.float32))

        musicbrainz_dtype = np.dtype([("year", "<i4")])
        musicbrainz_songs = np.zeros(1, dtype=musicbrainz_dtype)
        musicbrainz_songs[0]["year"] = year
        musicbrainz.create_dataset("songs", data=musicbrainz_songs)


def _index_columns(conn: sqlite3.Connection, index_name: str) -> tuple[str, ...]:
    rows = conn.execute(f"PRAGMA index_info({index_name!r})").fetchall()
    ordered = sorted(rows, key=lambda row: int(row[0]))
    return tuple(str(row[2]) for row in ordered)


def test_build_db_happy_path_creates_table_and_indexes(tmp_path: Path) -> None:
    msd_root = tmp_path / "msd"
    _write_tiny_track_h5(
        msd_root / "a" / "track.h5",
        track_id="TRTEST000000000001",
        title="Tiny Track",
        artist="Tiny Artist",
        year=2001,
    )
    out_db = tmp_path / "data" / "msd_meta.sqlite"

    code = build_db.main(["--msd_root", str(msd_root), "--out", str(out_db)])

    assert code == 0
    assert out_db.exists()
    with sqlite3.connect(out_db) as conn:
        rows = conn.execute("SELECT track_id, title, artist, duration, year FROM msd_meta").fetchall()
        assert rows == [("TRTEST000000000001", "Tiny Track", "Tiny Artist", 210.5, 2001)]

        index_names = [row[1] for row in conn.execute("PRAGMA index_list('msd_meta')").fetchall()]
        index_columns = {_index_columns(conn, name) for name in index_names}
        assert ("title", "artist") in index_columns
        assert ("artist",) in index_columns


def test_build_db_failure_when_all_files_bad(tmp_path: Path) -> None:
    msd_root = tmp_path / "msd"
    msd_root.mkdir()
    (msd_root / "bad.h5").write_text("not an hdf5", encoding="utf-8")
    out_db = tmp_path / "meta.sqlite"

    code = build_db.main(["--msd_root", str(msd_root), "--out", str(out_db)])

    assert code == 1
    assert out_db.exists()
    with sqlite3.connect(out_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM msd_meta").fetchone()[0]
    assert count == 0


def test_build_db_duplicates_keep_first_record(tmp_path: Path) -> None:
    msd_root = tmp_path / "msd"
    track_id = "TRTEST000000000099"
    _write_tiny_track_h5(
        msd_root / "a_first.h5",
        track_id=track_id,
        title="First Title",
        artist="Artist A",
    )
    _write_tiny_track_h5(
        msd_root / "b_second.h5",
        track_id=track_id,
        title="Second Title",
        artist="Artist B",
    )
    out_db = tmp_path / "meta.sqlite"

    code = build_db.main(["--msd_root", str(msd_root), "--out", str(out_db)])

    assert code == 0
    with sqlite3.connect(out_db) as conn:
        row = conn.execute("SELECT title, artist FROM msd_meta WHERE track_id = ?", (track_id,)).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM msd_meta").fetchone()[0]
    assert row == ("First Title", "Artist A")
    assert count == 1


def test_build_db_overwrites_existing_output(tmp_path: Path) -> None:
    msd_root = tmp_path / "msd"
    _write_tiny_track_h5(
        msd_root / "track.h5",
        track_id="TRTEST000000000010",
        title="New Title",
        artist="New Artist",
    )
    out_db = tmp_path / "meta.sqlite"
    out_db.write_text("stale", encoding="utf-8")

    code = build_db.main(["--msd_root", str(msd_root), "--out", str(out_db)])

    assert code == 0
    with sqlite3.connect(out_db) as conn:
        row = conn.execute("SELECT title, artist FROM msd_meta").fetchone()
    assert row == ("New Title", "New Artist")


def test_build_vectors_happy_path_writes_ids_and_vectors(tmp_path: Path) -> None:
    msd_root = tmp_path / "msd"
    _write_tiny_track_h5(
        msd_root / "a.h5",
        track_id="TRTEST000000000011",
        title="Track One",
        artist="Artist One",
    )
    _write_tiny_track_h5(
        msd_root / "b.h5",
        track_id="TRTEST000000000012",
        title="Track Two",
        artist="Artist Two",
    )
    out_ids = tmp_path / "ids.json"
    out_vecs = tmp_path / "vecs.npy"

    code = build_vectors.main(
        ["--msd_root", str(msd_root), "--out_ids", str(out_ids), "--out_vecs", str(out_vecs)]
    )

    assert code == 0
    ids = json.loads(out_ids.read_text(encoding="utf-8"))
    matrix = np.load(out_vecs)
    assert len(ids) == matrix.shape[0]
    assert matrix.shape == (2, 565)
    assert matrix.dtype == np.float32
    assert ids == ["TRTEST000000000011", "TRTEST000000000012"]


def test_build_vectors_failure_when_all_files_bad(tmp_path: Path) -> None:
    msd_root = tmp_path / "msd"
    msd_root.mkdir()
    (msd_root / "bad.h5").write_text("not an hdf5", encoding="utf-8")
    out_ids = tmp_path / "ids.json"
    out_vecs = tmp_path / "vecs.npy"

    code = build_vectors.main(
        ["--msd_root", str(msd_root), "--out_ids", str(out_ids), "--out_vecs", str(out_vecs)]
    )

    assert code == 1
    assert not out_ids.exists()
    assert not out_vecs.exists()


def test_build_vectors_duplicates_keep_first_record(tmp_path: Path) -> None:
    msd_root = tmp_path / "msd"
    track_id = "TRTEST000000000013"
    _write_tiny_track_h5(
        msd_root / "a_first.h5",
        track_id=track_id,
        title="First",
        artist="Artist",
        tempo=90.0,
    )
    _write_tiny_track_h5(
        msd_root / "b_second.h5",
        track_id=track_id,
        title="Second",
        artist="Artist",
        tempo=180.0,
    )
    out_ids = tmp_path / "ids.json"
    out_vecs = tmp_path / "vecs.npy"

    code = build_vectors.main(
        ["--msd_root", str(msd_root), "--out_ids", str(out_ids), "--out_vecs", str(out_vecs)]
    )

    assert code == 0
    ids = json.loads(out_ids.read_text(encoding="utf-8"))
    matrix = np.load(out_vecs)
    assert ids == [track_id]
    assert matrix.shape == (1, 565)


def test_build_vectors_overwrites_existing_outputs(tmp_path: Path) -> None:
    msd_root = tmp_path / "msd"
    _write_tiny_track_h5(
        msd_root / "track.h5",
        track_id="TRTEST000000000014",
        title="Fresh",
        artist="Artist",
    )
    out_ids = tmp_path / "ids.json"
    out_vecs = tmp_path / "vecs.npy"
    out_ids.write_text('["stale"]', encoding="utf-8")
    np.save(out_vecs, np.zeros((1, 3), dtype=np.float32))

    code = build_vectors.main(
        ["--msd_root", str(msd_root), "--out_ids", str(out_ids), "--out_vecs", str(out_vecs)]
    )

    assert code == 0
    ids = json.loads(out_ids.read_text(encoding="utf-8"))
    matrix = np.load(out_vecs)
    assert ids == ["TRTEST000000000014"]
    assert matrix.shape == (1, 565)
