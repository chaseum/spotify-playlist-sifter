from fastapi.testclient import TestClient

import app.api.routes.config as config_route
from app.core.config import Settings
from app.main import app

client = TestClient(app)


def test_api_config_returns_public_spotify_config(monkeypatch) -> None:
    monkeypatch.setattr(
        config_route,
        "settings",
        Settings(
            spotify_client_id="client-123",
            spotify_redirect_uri="http://127.0.0.1:8000/",
            spotify_scopes="user-read-private",
            spotify_authorize_url="https://accounts.spotify.com/authorize",
            spotify_token_url="https://accounts.spotify.com/api/token",
        ),
    )

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json() == {
        "spotify_client_id": "client-123",
        "spotify_redirect_uri": "http://127.0.0.1:8000/",
        "spotify_scopes": "user-read-private",
        "spotify_authorize_url": "https://accounts.spotify.com/authorize",
        "spotify_token_url": "https://accounts.spotify.com/api/token",
    }


def test_api_config_returns_500_when_client_id_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        config_route,
        "settings",
        Settings(
            spotify_client_id="",
            spotify_redirect_uri="http://127.0.0.1:8000/",
            spotify_scopes="user-read-private",
            spotify_authorize_url="https://accounts.spotify.com/authorize",
            spotify_token_url="https://accounts.spotify.com/api/token",
        ),
    )

    response = client.get("/api/config")

    assert response.status_code == 500
    assert response.json() == {"detail": "SPOTIFY_CLIENT_ID is not configured"}


def test_api_config_uses_default_scopes_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        config_route,
        "settings",
        Settings(
            spotify_client_id="client-123",
            spotify_redirect_uri="http://127.0.0.1:8000/",
            spotify_scopes="",
            spotify_authorize_url="https://accounts.spotify.com/authorize",
            spotify_token_url="https://accounts.spotify.com/api/token",
        ),
    )

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["spotify_scopes"] == (
        "user-read-private user-read-email playlist-modify-private "
        "playlist-modify-public user-library-read user-library-modify"
    )
