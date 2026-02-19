import json
import sqlite3
from urllib.error import URLError

import app.services.feature_store as feature_store


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _configure_env(monkeypatch, tmp_path) -> str:
    db_path = tmp_path / "feature_store.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("MUSICBRAINZ_USER_AGENT", "spotify-project-tests/1.0 (test@example.com)")
    monkeypatch.setattr(feature_store, "MB_MIN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(feature_store, "_LAST_MUSICBRAINZ_REQUEST_MONO", 0.0)
    return str(db_path)


def test_mbid_from_isrc_cache_miss_then_hit(monkeypatch, tmp_path) -> None:
    _configure_env(monkeypatch, tmp_path)
    state = {"calls": 0}

    def fake_urlopen(request, timeout=15):
        state["calls"] += 1
        return _FakeResponse(
            {
                "recordings": [
                    {"id": "00000000-0000-0000-0000-000000000001", "score": 80},
                    {"id": "00000000-0000-0000-0000-000000000002", "score": 100},
                ]
            }
        )

    monkeypatch.setattr(feature_store, "urlopen", fake_urlopen)

    first = feature_store.mbid_from_isrc("usabc1234567")
    second = feature_store.mbid_from_isrc("USABC1234567")

    assert first == "00000000-0000-0000-0000-000000000002"
    assert second == "00000000-0000-0000-0000-000000000002"
    assert state["calls"] == 1


def test_get_isrc_from_spotify_track_for_session_missing_isrc_uses_negative_cache(monkeypatch, tmp_path) -> None:
    _configure_env(monkeypatch, tmp_path)
    state = {"calls": 0}

    def fake_get_track_for_session(session_id: str, track_id: str) -> dict:
        state["calls"] += 1
        assert session_id == "session-123"
        assert track_id == "track-123"
        return {"id": "track-123", "external_ids": {}}

    monkeypatch.setattr(feature_store, "get_track_for_session", fake_get_track_for_session)

    first = feature_store.get_isrc_from_spotify_track_for_session("session-123", "track-123")
    second = feature_store.get_isrc_from_spotify_track_for_session("session-123", "track-123")

    assert first is None
    assert second is None
    assert state["calls"] == 1


def test_mbid_from_isrc_returns_none_when_musicbrainz_lookup_fails(monkeypatch, tmp_path) -> None:
    _configure_env(monkeypatch, tmp_path)

    def failing_urlopen(request, timeout=15):
        raise URLError("downstream-unavailable")

    monkeypatch.setattr(feature_store, "urlopen", failing_urlopen)

    assert feature_store.mbid_from_isrc("USABC1234567") is None


def test_get_track_features_returns_stale_cache_when_refresh_fails(monkeypatch, tmp_path) -> None:
    db_path = _configure_env(monkeypatch, tmp_path)
    mbid = "123e4567-e89b-12d3-a456-426614174000"
    state = {"calls": 0}

    def fake_urlopen(request, timeout=15):
        state["calls"] += 1
        if state["calls"] == 1:
            return _FakeResponse(
                {
                    "id": mbid,
                    "title": "Song A",
                    "length": 201000,
                    "disambiguation": "studio",
                    "artist-credit": [{"name": "Artist A"}],
                    "tags": [{"name": "indie", "count": 5}],
                    "genres": [{"name": "rock", "count": 2}],
                    "releases": [{"id": "release-1", "title": "Album A", "date": "2020-01-01"}],
                }
            )
        raise URLError("rate-limited")

    monkeypatch.setattr(feature_store, "urlopen", fake_urlopen)

    initial = feature_store.get_track_features(mbid)
    assert initial is not None

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE track_features SET expires_at = ?, backoff_until = 0 WHERE mbid = ?",
            (feature_store._epoch_seconds() - 1, mbid),
        )
        conn.commit()

    stale = feature_store.get_track_features(mbid)
    assert stale == initial

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT backoff_until FROM track_features WHERE mbid = ?", (mbid,)).fetchone()

    assert row is not None
    assert int(row[0]) > feature_store._epoch_seconds()
