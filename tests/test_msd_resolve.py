from __future__ import annotations

import sqlite3
from pathlib import Path

from ml.msd.resolve import normalize_text, resolve_spotify_track


def _seed_test_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE msd_meta (
                track_id TEXT PRIMARY KEY,
                title TEXT,
                artist TEXT,
                duration REAL,
                year INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO msd_meta(track_id, title, artist, duration, year)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("TR0001", "Echoes of You", "Nova", 200.0, 2019),
                ("TR0002", "Echoes of You Live", "Nova", 201.0, 2019),
                ("TR0003", "Neon Hearts", "Lumen", 180.0, 2021),
                ("TR0004", "Satellite Dreams", "Aurora Sky", 240.0, 2020),
                ("TR0005", "Paper Trails", "Gray Fox", 210.0, 2018),
            ],
        )


def test_normalize_text_strips_noise_tokens_and_collapses_spaces() -> None:
    raw = "  Echoes of You (Remastered 2012) - LIVE stereo mix ft. Alice  "

    normalized = normalize_text(raw)

    assert normalized == "echoes of you mix"


def test_resolve_spotify_track_happy_path(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.sqlite"
    _seed_test_db(db_path)

    match = resolve_spotify_track(
        title="Echoes of You (Remastered 2012) ft. Alice",
        artist="Nova feat. Bob",
        duration_s=200.2,
        sqlite_path=db_path,
    )

    assert match is not None
    track_id, confidence = match
    assert track_id == "TR0001"
    assert 0.88 <= confidence <= 1.0


def test_resolve_spotify_track_duration_filter_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.sqlite"
    _seed_test_db(db_path)

    match = resolve_spotify_track(
        title="Echoes of You",
        artist="Nova",
        duration_s=230.0,
        sqlite_path=db_path,
    )

    assert match is None


def test_resolve_spotify_track_score_threshold_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.sqlite"
    _seed_test_db(db_path)

    match = resolve_spotify_track(
        title="Satellite",
        artist="Completely Different",
        duration_s=240.0,
        sqlite_path=db_path,
    )

    assert match is None


def test_resolve_spotify_track_is_deterministic_for_ties(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.sqlite"
    _seed_test_db(db_path)

    match = resolve_spotify_track(
        title="Echoes of You",
        artist="Nova",
        duration_s=200.5,
        sqlite_path=db_path,
    )

    assert match is not None
    assert match[0] == "TR0001"
