import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, unquote
from urllib.request import Request, urlopen

from app.services.spotify_client import SpotifyClientError, get_track, get_track_for_session

DATABASE_URL_DEFAULT = "sqlite:///./feature_store.db"
MUSICBRAINZ_BASE_URL = "https://musicbrainz.org"

MAPPING_TTL_SECONDS = 30 * 24 * 60 * 60
TRACK_FEATURES_TTL_SECONDS = 7 * 24 * 60 * 60
NEGATIVE_TTL_SECONDS = 6 * 60 * 60
ERROR_BACKOFF_SECONDS = 15 * 60
MB_MIN_INTERVAL_SECONDS = 1.1

_MB_THROTTLE_LOCK = threading.Lock()
_LAST_MUSICBRAINZ_REQUEST_MONO = 0.0


class _MusicBrainzLookupError(Exception):
    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _epoch_seconds() -> int:
    return int(time.time())


def _sqlite_path_from_database_url(database_url: str) -> str:
    raw_url = database_url.strip() or DATABASE_URL_DEFAULT
    if not raw_url.startswith("sqlite:///"):
        raise ValueError("DATABASE_URL must use sqlite:/// for feature_store")

    raw_path = unquote(raw_url[len("sqlite:///") :]).strip()
    if not raw_path:
        raw_path = "./feature_store.db"

    # sqlite:///C:/path/db.sqlite can be parsed as /C:/path/db.sqlite.
    if raw_path.startswith("/") and len(raw_path) >= 3 and raw_path[2] == ":" and raw_path[1].isalpha():
        raw_path = raw_path[1:]

    return raw_path


@contextmanager
def _db_connection() -> Any:
    database_url = os.getenv("DATABASE_URL", DATABASE_URL_DEFAULT)
    db_path = _sqlite_path_from_database_url(database_url)
    if db_path != ":memory:":
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spotify_to_isrc (
            spotify_track_id TEXT PRIMARY KEY,
            isrc TEXT NULL,
            updated_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            backoff_until INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS isrc_to_mbid (
            isrc TEXT PRIMARY KEY,
            mbid TEXT NULL,
            updated_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            backoff_until INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS track_features (
            mbid TEXT PRIMARY KEY,
            tags_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            backoff_until INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def _musicbrainz_user_agent() -> str:
    user_agent = os.getenv("MUSICBRAINZ_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError("MUSICBRAINZ_USER_AGENT is required")
    return user_agent


def _throttle_musicbrainz_requests() -> None:
    global _LAST_MUSICBRAINZ_REQUEST_MONO

    interval = max(0.0, float(MB_MIN_INTERVAL_SECONDS))
    if interval <= 0:
        return

    with _MB_THROTTLE_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_MUSICBRAINZ_REQUEST_MONO
        if elapsed < interval:
            time.sleep(interval - elapsed)
            now = time.monotonic()
        _LAST_MUSICBRAINZ_REQUEST_MONO = now


def _musicbrainz_request_json(path: str) -> dict[str, Any]:
    _throttle_musicbrainz_requests()
    request = Request(
        f"{MUSICBRAINZ_BASE_URL}{path}",
        headers={
            "User-Agent": _musicbrainz_user_agent(),
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise _MusicBrainzLookupError(status_code=exc.code, message="MusicBrainz request failed") from exc
    except URLError as exc:
        raise _MusicBrainzLookupError(status_code=None, message="MusicBrainz unavailable") from exc

    if not body:
        return {}

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise _MusicBrainzLookupError(status_code=None, message="MusicBrainz returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise _MusicBrainzLookupError(status_code=None, message="MusicBrainz returned invalid payload")

    return payload


def _is_cache_usable(row: sqlite3.Row | None, now: int) -> bool:
    if not row:
        return False
    expires_at = int(row["expires_at"])
    backoff_until = int(row["backoff_until"])
    return now <= expires_at or now <= backoff_until


def _normalize_isrc(isrc: str) -> str:
    return isrc.strip().upper()


def _normalize_mbid(mbid: str) -> str:
    return mbid.strip()


def _get_spotify_to_isrc_row(conn: sqlite3.Connection, spotify_track_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT spotify_track_id, isrc, updated_at, expires_at, backoff_until
        FROM spotify_to_isrc
        WHERE spotify_track_id = ?
        """,
        (spotify_track_id,),
    ).fetchone()


def _upsert_spotify_to_isrc(
    conn: sqlite3.Connection,
    spotify_track_id: str,
    isrc: str | None,
    now: int,
    ttl_seconds: int,
) -> None:
    conn.execute(
        """
        INSERT INTO spotify_to_isrc (spotify_track_id, isrc, updated_at, expires_at, backoff_until)
        VALUES (?, ?, ?, ?, 0)
        ON CONFLICT(spotify_track_id) DO UPDATE SET
            isrc = excluded.isrc,
            updated_at = excluded.updated_at,
            expires_at = excluded.expires_at,
            backoff_until = 0
        """,
        (spotify_track_id, isrc, now, now + ttl_seconds),
    )


def _set_spotify_to_isrc_backoff(conn: sqlite3.Connection, spotify_track_id: str, now: int) -> None:
    conn.execute(
        "UPDATE spotify_to_isrc SET backoff_until = ? WHERE spotify_track_id = ?",
        (now + ERROR_BACKOFF_SECONDS, spotify_track_id),
    )


def _get_isrc_to_mbid_row(conn: sqlite3.Connection, isrc: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT isrc, mbid, updated_at, expires_at, backoff_until
        FROM isrc_to_mbid
        WHERE isrc = ?
        """,
        (isrc,),
    ).fetchone()


def _upsert_isrc_to_mbid(
    conn: sqlite3.Connection,
    isrc: str,
    mbid: str | None,
    now: int,
    ttl_seconds: int,
) -> None:
    conn.execute(
        """
        INSERT INTO isrc_to_mbid (isrc, mbid, updated_at, expires_at, backoff_until)
        VALUES (?, ?, ?, ?, 0)
        ON CONFLICT(isrc) DO UPDATE SET
            mbid = excluded.mbid,
            updated_at = excluded.updated_at,
            expires_at = excluded.expires_at,
            backoff_until = 0
        """,
        (isrc, mbid, now, now + ttl_seconds),
    )


def _set_isrc_to_mbid_backoff(conn: sqlite3.Connection, isrc: str, now: int) -> None:
    conn.execute(
        "UPDATE isrc_to_mbid SET backoff_until = ? WHERE isrc = ?",
        (now + ERROR_BACKOFF_SECONDS, isrc),
    )


def _get_track_features_row(conn: sqlite3.Connection, mbid: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT mbid, tags_json, metadata_json, updated_at, expires_at, backoff_until
        FROM track_features
        WHERE mbid = ?
        """,
        (mbid,),
    ).fetchone()


def _upsert_track_features(
    conn: sqlite3.Connection,
    mbid: str,
    tags: list[dict[str, Any]],
    metadata: dict[str, Any],
    now: int,
    ttl_seconds: int,
) -> None:
    conn.execute(
        """
        INSERT INTO track_features (mbid, tags_json, metadata_json, updated_at, expires_at, backoff_until)
        VALUES (?, ?, ?, ?, ?, 0)
        ON CONFLICT(mbid) DO UPDATE SET
            tags_json = excluded.tags_json,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at,
            expires_at = excluded.expires_at,
            backoff_until = 0
        """,
        (
            mbid,
            json.dumps(tags, separators=(",", ":")),
            json.dumps(metadata, separators=(",", ":")),
            now,
            now + ttl_seconds,
        ),
    )


def _set_track_features_backoff(conn: sqlite3.Connection, mbid: str, now: int) -> None:
    conn.execute(
        "UPDATE track_features SET backoff_until = ? WHERE mbid = ?",
        (now + ERROR_BACKOFF_SECONDS, mbid),
    )


def _extract_isrc_from_track(track_payload: dict[str, Any]) -> str | None:
    external_ids = track_payload.get("external_ids")
    if not isinstance(external_ids, dict):
        return None

    raw_isrc = external_ids.get("isrc")
    if not isinstance(raw_isrc, str):
        return None

    normalized_isrc = _normalize_isrc(raw_isrc)
    if not normalized_isrc:
        return None

    return normalized_isrc


def get_isrc_from_spotify_track(spotify_track_id: str, access_token: str) -> str | None:
    safe_track_id = spotify_track_id.strip()
    if not safe_track_id:
        return None

    now = _epoch_seconds()
    with _db_connection() as conn:
        cached_row = _get_spotify_to_isrc_row(conn, safe_track_id)
        if _is_cache_usable(cached_row, now):
            return cached_row["isrc"]

    fetched_isrc: str | None = None
    fetch_failed = False
    try:
        track_payload = get_track(access_token=access_token, track_id=safe_track_id)
        fetched_isrc = _extract_isrc_from_track(track_payload)
    except SpotifyClientError:
        fetch_failed = True

    now = _epoch_seconds()
    with _db_connection() as conn:
        cached_row = _get_spotify_to_isrc_row(conn, safe_track_id)
        if fetch_failed:
            if cached_row:
                _set_spotify_to_isrc_backoff(conn, safe_track_id, now)
                return cached_row["isrc"]
            return None

        ttl_seconds = MAPPING_TTL_SECONDS if fetched_isrc else NEGATIVE_TTL_SECONDS
        _upsert_spotify_to_isrc(conn, safe_track_id, fetched_isrc, now, ttl_seconds)
        return fetched_isrc


def get_isrc_from_spotify_track_for_session(session_id: str, spotify_track_id: str) -> str | None:
    safe_track_id = spotify_track_id.strip()
    if not safe_track_id:
        return None

    now = _epoch_seconds()
    with _db_connection() as conn:
        cached_row = _get_spotify_to_isrc_row(conn, safe_track_id)
        if _is_cache_usable(cached_row, now):
            return cached_row["isrc"]

    fetched_isrc: str | None = None
    fetch_failed = False
    try:
        track_payload = get_track_for_session(session_id=session_id, track_id=safe_track_id)
        fetched_isrc = _extract_isrc_from_track(track_payload)
    except SpotifyClientError:
        fetch_failed = True

    now = _epoch_seconds()
    with _db_connection() as conn:
        cached_row = _get_spotify_to_isrc_row(conn, safe_track_id)
        if fetch_failed:
            if cached_row:
                _set_spotify_to_isrc_backoff(conn, safe_track_id, now)
                return cached_row["isrc"]
            return None

        ttl_seconds = MAPPING_TTL_SECONDS if fetched_isrc else NEGATIVE_TTL_SECONDS
        _upsert_spotify_to_isrc(conn, safe_track_id, fetched_isrc, now, ttl_seconds)
        return fetched_isrc


def _recording_score(recording: dict[str, Any]) -> int:
    raw_score = recording.get("score", 0)
    try:
        return int(raw_score)
    except (TypeError, ValueError):
        return 0


def _count_value(item: dict[str, Any]) -> int:
    raw_count = item.get("count", 0)
    try:
        return int(raw_count)
    except (TypeError, ValueError):
        return 0


def _pick_best_recording(recordings: list[Any], expected_mbid: str | None = None) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = [item for item in recordings if isinstance(item, dict)]
    if not candidates:
        return None

    if expected_mbid:
        for recording in candidates:
            if recording.get("id") == expected_mbid:
                return recording

    best = candidates[0]
    for candidate in candidates[1:]:
        best_key = (_recording_score(best), str(best.get("id", "")))
        candidate_key = (_recording_score(candidate), str(candidate.get("id", "")))
        if candidate_key > best_key:
            best = candidate
    return best


def mbid_from_isrc(isrc: str) -> str | None:
    normalized_isrc = _normalize_isrc(isrc)
    if not normalized_isrc:
        return None

    now = _epoch_seconds()
    with _db_connection() as conn:
        cached_row = _get_isrc_to_mbid_row(conn, normalized_isrc)
        if _is_cache_usable(cached_row, now):
            return cached_row["mbid"]

    fetched_mbid: str | None = None
    fetch_failed = False
    try:
        payload = _musicbrainz_request_json(f"/ws/2/isrc/{quote(normalized_isrc)}?fmt=json")
        recordings = payload.get("recordings")
        if isinstance(recordings, list):
            best_recording = _pick_best_recording(recordings)
            if best_recording:
                raw_mbid = best_recording.get("id")
                if isinstance(raw_mbid, str) and raw_mbid.strip():
                    fetched_mbid = _normalize_mbid(raw_mbid)
    except (_MusicBrainzLookupError, RuntimeError):
        fetch_failed = True

    now = _epoch_seconds()
    with _db_connection() as conn:
        cached_row = _get_isrc_to_mbid_row(conn, normalized_isrc)
        if fetch_failed:
            if cached_row:
                _set_isrc_to_mbid_backoff(conn, normalized_isrc, now)
                return cached_row["mbid"]
            return None

        ttl_seconds = MAPPING_TTL_SECONDS if fetched_mbid else NEGATIVE_TTL_SECONDS
        _upsert_isrc_to_mbid(conn, normalized_isrc, fetched_mbid, now, ttl_seconds)
        return fetched_mbid


def _extract_track_features(recording: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_tags = recording.get("tags")
    raw_genres = recording.get("genres")

    tags: list[dict[str, Any]] = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tags.append(
                {
                    "name": name.strip(),
                    "count": _count_value(item),
                    "source": "tag",
                }
            )

    if isinstance(raw_genres, list):
        for item in raw_genres:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tags.append(
                {
                    "name": name.strip(),
                    "count": _count_value(item),
                    "source": "genre",
                }
            )

    artist_names: list[str] = []
    artist_credit = recording.get("artist-credit")
    if isinstance(artist_credit, list):
        for item in artist_credit:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                artist_names.append(name.strip())

            artist = item.get("artist")
            if isinstance(artist, dict):
                artist_name = artist.get("name")
                if isinstance(artist_name, str) and artist_name.strip():
                    artist_names.append(artist_name.strip())

    seen_artists: set[str] = set()
    unique_artists: list[str] = []
    for artist_name in artist_names:
        folded = artist_name.casefold()
        if folded in seen_artists:
            continue
        seen_artists.add(folded)
        unique_artists.append(artist_name)

    releases_summary: list[dict[str, Any]] = []
    releases = recording.get("releases")
    if isinstance(releases, list):
        for item in releases:
            if not isinstance(item, dict):
                continue
            release_id = item.get("id")
            release_title = item.get("title")
            release_date = item.get("date")
            releases_summary.append(
                {
                    "id": release_id if isinstance(release_id, str) else None,
                    "title": release_title if isinstance(release_title, str) else None,
                    "date": release_date if isinstance(release_date, str) else None,
                }
            )

    metadata = {
        "mbid": recording.get("id") if isinstance(recording.get("id"), str) else None,
        "title": recording.get("title") if isinstance(recording.get("title"), str) else None,
        "length_ms": recording.get("length") if isinstance(recording.get("length"), int) else None,
        "disambiguation": (
            recording.get("disambiguation") if isinstance(recording.get("disambiguation"), str) else None
        ),
        "artists": unique_artists,
        "releases": releases_summary,
    }
    return tags, metadata


def _lookup_recording_by_mbid(mbid: str) -> dict[str, Any] | None:
    primary_path = f"/ws/2/recording/{quote(mbid)}?fmt=json&inc=tags+genres+artist-credits+releases"

    primary_error: _MusicBrainzLookupError | RuntimeError | None = None
    try:
        primary_payload = _musicbrainz_request_json(primary_path)
        if isinstance(primary_payload, dict) and isinstance(primary_payload.get("id"), str):
            return primary_payload
    except (_MusicBrainzLookupError, RuntimeError) as exc:
        primary_error = exc

    fallback_query = urlencode({"query": f"rid:{mbid}", "fmt": "json", "limit": 1})
    fallback_path = f"/ws/2/recording?{fallback_query}"
    try:
        fallback_payload = _musicbrainz_request_json(fallback_path)
    except (_MusicBrainzLookupError, RuntimeError):
        if primary_error:
            raise primary_error
        raise

    recordings = fallback_payload.get("recordings")
    if not isinstance(recordings, list):
        if primary_error and isinstance(primary_error, _MusicBrainzLookupError):
            if primary_error.status_code not in (None, 404):
                raise primary_error
        return None

    return _pick_best_recording(recordings, expected_mbid=mbid)


def _decode_track_features_row(row: sqlite3.Row) -> dict[str, Any] | None:
    try:
        tags = json.loads(row["tags_json"])
        metadata = json.loads(row["metadata_json"])
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(tags, list):
        tags = []
    if not isinstance(metadata, dict):
        metadata = {}

    if metadata.get("__missing__") is True:
        return None

    return {"tags": tags, "metadata": metadata}


def get_track_features(mbid: str) -> dict[str, Any] | None:
    normalized_mbid = _normalize_mbid(mbid)
    if not normalized_mbid:
        return None

    now = _epoch_seconds()
    with _db_connection() as conn:
        cached_row = _get_track_features_row(conn, normalized_mbid)
        if _is_cache_usable(cached_row, now):
            return _decode_track_features_row(cached_row)

    fetched_tags: list[dict[str, Any]] = []
    fetched_metadata: dict[str, Any] | None = None
    fetch_failed = False
    try:
        recording = _lookup_recording_by_mbid(normalized_mbid)
        if recording:
            fetched_tags, fetched_metadata = _extract_track_features(recording)
    except (_MusicBrainzLookupError, RuntimeError):
        fetch_failed = True

    now = _epoch_seconds()
    with _db_connection() as conn:
        cached_row = _get_track_features_row(conn, normalized_mbid)
        if fetch_failed:
            if cached_row:
                _set_track_features_backoff(conn, normalized_mbid, now)
                return _decode_track_features_row(cached_row)
            return None

        if fetched_metadata is None:
            _upsert_track_features(
                conn=conn,
                mbid=normalized_mbid,
                tags=[],
                metadata={"__missing__": True},
                now=now,
                ttl_seconds=NEGATIVE_TTL_SECONDS,
            )
            return None

        _upsert_track_features(
            conn=conn,
            mbid=normalized_mbid,
            tags=fetched_tags,
            metadata=fetched_metadata,
            now=now,
            ttl_seconds=TRACK_FEATURES_TTL_SECONDS,
        )
        return {"tags": fetched_tags, "metadata": fetched_metadata}
