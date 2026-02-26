from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from rapidfuzz import fuzz

MAX_CANDIDATES = 200
MAX_DURATION_DELTA_SECONDS = 2.5
MIN_SCORE = 88.0

_FEATURE_SEGMENT_PATTERN = re.compile(r"\b(?:feat|ft|featuring)\.?\b.*$", flags=re.IGNORECASE)
_PUNCTUATION_PATTERN = re.compile(r"[^a-z0-9\s]+")
_NOISE_TOKEN_PATTERN = re.compile(
    r"\b(?:remaster(?:ed)?(?:\s+\d{2,4})?|live|mono|stereo)\b",
    flags=re.IGNORECASE,
)
_SPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    without_feature_segment = _FEATURE_SEGMENT_PATTERN.sub(" ", lowered)
    without_punct = _PUNCTUATION_PATTERN.sub(" ", without_feature_segment)
    without_noise_tokens = _NOISE_TOKEN_PATTERN.sub(" ", without_punct)
    return _SPACE_PATTERN.sub(" ", without_noise_tokens).strip()


def _candidate_rows(sqlite_path: str | Path, query_token: str) -> list[sqlite3.Row]:
    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        return list(
            conn.execute(
                """
                SELECT track_id, title, artist, duration
                FROM msd_meta
                WHERE lower(title) LIKE ?
                LIMIT ?
                """,
                (f"%{query_token}%", MAX_CANDIDATES),
            ).fetchall()
        )


def resolve_spotify_track(
    title: str,
    artist: str,
    duration_s: float,
    sqlite_path: str | Path,
) -> tuple[str, float] | None:
    normalized_title = normalize_text(title)
    normalized_artist = normalize_text(artist)
    normalized_query = f"{normalized_title} {normalized_artist}".strip()
    if not normalized_title or not normalized_query:
        return None

    title_tokens = normalized_title.split()
    if not title_tokens:
        return None
    query_token = max(title_tokens, key=len)

    best_track_id: str | None = None
    best_score = -1.0
    best_duration_delta = float("inf")

    for row in _candidate_rows(sqlite_path, query_token):
        track_id = str(row["track_id"])
        row_duration = float(row["duration"])
        duration_delta = abs(row_duration - float(duration_s))
        if duration_delta > MAX_DURATION_DELTA_SECONDS:
            continue

        row_title = normalize_text(str(row["title"] or ""))
        row_artist = normalize_text(str(row["artist"] or ""))
        row_text = f"{row_title} {row_artist}".strip()
        if not row_text:
            continue

        score = float(fuzz.token_set_ratio(normalized_query, row_text))
        is_better = False
        if score > best_score:
            is_better = True
        elif score == best_score and duration_delta < best_duration_delta:
            is_better = True
        elif (
            score == best_score
            and duration_delta == best_duration_delta
            and best_track_id is not None
            and track_id < best_track_id
        ):
            is_better = True

        if is_better:
            best_track_id = track_id
            best_score = score
            best_duration_delta = duration_delta

    if best_track_id is None or best_score < MIN_SCORE:
        return None
    return best_track_id, best_score / 100.0
