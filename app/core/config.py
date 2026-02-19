import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    spotify_client_id: str
    spotify_redirect_uri: str
    spotify_scopes: str
    spotify_authorize_url: str
    spotify_token_url: str


def _read_config_value(key: str) -> str:
    env_value = os.getenv(key, "").strip()
    if env_value:
        return env_value

    env_file = Path(".env")
    if not env_file.exists():
        return ""

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, raw_value = line.split("=", 1)
        if name.strip() != key:
            continue

        value = raw_value.split("#", 1)[0].strip().strip("\"'")
        return value

    return ""


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


settings = Settings(
    spotify_client_id=_first_non_empty(
        _read_config_value("SPOTIFY_CLIENT_ID"),
        _read_config_value("CLIENT_ID"),
    ),
    spotify_redirect_uri=_first_non_empty(
        _read_config_value("SPOTIFY_REDIRECT_URI"),
        "http://127.0.0.1:8000/",
    ),
    spotify_scopes=_first_non_empty(
        _read_config_value("SPOTIFY_SCOPES"),
        (
            "user-read-private user-read-email playlist-modify-private "
            "playlist-modify-public user-library-read user-library-modify"
        ),
    ),
    spotify_authorize_url=_first_non_empty(
        _read_config_value("SPOTIFY_AUTHORIZE_URL"),
        "https://accounts.spotify.com/authorize",
    ),
    spotify_token_url=_first_non_empty(
        _read_config_value("SPOTIFY_TOKEN_URL"),
        "https://accounts.spotify.com/api/token",
    ),
)
