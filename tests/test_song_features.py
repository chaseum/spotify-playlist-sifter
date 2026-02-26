import pytest

from ml.sifter.features import (
    SongFeatures,
    build_song_features_from_dataset_row,
    build_song_features_from_spotify_track,
)


def test_build_song_features_from_dataset_row_missing_genre_and_lyrics() -> None:
    row = {
        "title": "My Song",
        "artist": "My Artist",
    }

    features = build_song_features_from_dataset_row(row)

    assert features == SongFeatures(
        title_text="My Song",
        artist_text="My Artist",
        genre_text=None,
        lyrics_text=None,
    )


def test_build_song_features_from_dataset_row_uses_aliases() -> None:
    row = {
        "song": "Alias Song",
        "artist_name": "Alias Artist",
        "genres": [" indie ", "", "pop"],
        "text": "  some lyrics  ",
    }

    features = build_song_features_from_dataset_row(row)

    assert features == SongFeatures(
        title_text="Alias Song",
        artist_text="Alias Artist",
        genre_text="indie, pop",
        lyrics_text="some lyrics",
    )


def test_build_song_features_from_spotify_track_missing_genres_and_lyrics() -> None:
    track = {
        "name": "Track A",
        "artists": [{"name": "Artist A"}],
    }

    features = build_song_features_from_spotify_track(
        track=track,
        artist_genres=[],
        matched_lyrics=None,
    )

    assert features == SongFeatures(
        title_text="Track A",
        artist_text="Artist A",
        genre_text=None,
        lyrics_text=None,
    )


def test_build_song_features_from_spotify_track_with_values() -> None:
    track = {
        "name": "Track A",
        "artists": [{"name": "Artist A"}, {"name": "Artist B"}],
    }

    features = build_song_features_from_spotify_track(
        track=track,
        artist_genres=[" rock ", "", "alt pop"],
        matched_lyrics="  la la la  ",
    )

    assert features == SongFeatures(
        title_text="Track A",
        artist_text="Artist A, Artist B",
        genre_text="rock, alt pop",
        lyrics_text="la la la",
    )


def test_build_song_features_from_dataset_row_missing_required_raises() -> None:
    with pytest.raises(ValueError, match="Missing required field: title_text"):
        build_song_features_from_dataset_row({"artist": "Only Artist"})

    with pytest.raises(ValueError, match="Missing required field: artist_text"):
        build_song_features_from_dataset_row({"title": "Only Title"})
