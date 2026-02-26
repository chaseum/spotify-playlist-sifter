from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SongFeatures:
    title_text: str
    artist_text: str
    genre_text: str | None
    lyrics_text: str | None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _first_text(mapping: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _clean_text(mapping.get(key))
        if value:
            return value
    return None


def _normalize_genres(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return cleaned

    if not isinstance(value, (list, tuple, set)):
        return None

    genres: list[str] = []
    for item in value:
        cleaned = _clean_text(item)
        if cleaned:
            genres.append(cleaned)

    if not genres:
        return None
    return ", ".join(genres)


def build_song_features_from_dataset_row(row: dict[str, Any]) -> SongFeatures:
    title_text = _first_text(row, ("title_text", "title", "song"))
    if not title_text:
        raise ValueError("Missing required field: title_text")

    artist_text = _first_text(row, ("artist_text", "artist", "artist_name"))
    if not artist_text:
        raise ValueError("Missing required field: artist_text")

    genre_text = _normalize_genres(row.get("genre_text"))
    if genre_text is None:
        genre_text = _normalize_genres(row.get("genre"))
    if genre_text is None:
        genre_text = _normalize_genres(row.get("genres"))

    lyrics_text = _first_text(row, ("lyrics_text", "lyrics", "text"))

    return SongFeatures(
        title_text=title_text,
        artist_text=artist_text,
        genre_text=genre_text,
        lyrics_text=lyrics_text,
    )


def build_song_features_from_spotify_track(
    track: dict[str, Any],
    artist_genres: list[str] | None,
    matched_lyrics: str | None,
) -> SongFeatures:
    title_text = _clean_text(track.get("name"))
    if not title_text:
        raise ValueError("Missing required field: title_text")

    artists_payload = track.get("artists")
    artist_names: list[str] = []
    if isinstance(artists_payload, list):
        for artist_item in artists_payload:
            if not isinstance(artist_item, dict):
                continue
            cleaned_name = _clean_text(artist_item.get("name"))
            if cleaned_name:
                artist_names.append(cleaned_name)

    artist_text = ", ".join(artist_names).strip()
    if not artist_text:
        raise ValueError("Missing required field: artist_text")

    return SongFeatures(
        title_text=title_text,
        artist_text=artist_text,
        genre_text=_normalize_genres(artist_genres),
        lyrics_text=_clean_text(matched_lyrics),
    )
