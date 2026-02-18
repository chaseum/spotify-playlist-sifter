from fastapi.testclient import TestClient

import app.api.routes.auth_spotify as auth_route
from app.core.config import Settings
from app.main import app

client = TestClient(app)


def test_callback_redirects_to_frontend_when_cookie_state_missing() -> None:
    response = client.get(
        "/auth/spotify/callback?code=abc123&state=frontend-state",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["location"] == "/?code=abc123&state=frontend-state"


def test_callback_returns_400_when_missing_callback_params() -> None:
    response = client.get("/auth/spotify/callback", follow_redirects=False)

    assert response.status_code == 400
    assert response.json() == {"detail": "Missing authorization code"}


def test_api_login_alias_redirects(monkeypatch) -> None:
    monkeypatch.setattr(
        auth_route,
        "settings",
        Settings(
            spotify_client_id="client-123",
            spotify_redirect_uri="http://127.0.0.1:8000/auth/spotify/callback",
            spotify_scopes="user-read-private user-read-email",
            spotify_authorize_url="https://accounts.spotify.com/authorize",
            spotify_token_url="https://accounts.spotify.com/api/token",
        ),
    )

    response = client.get("/api/auth/spotify/login", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"].startswith("https://accounts.spotify.com/authorize?")


def test_api_logout_alias_clears_session_cookie() -> None:
    response = client.get(
        "/api/auth/logout",
        cookies={auth_route.SESSION_COOKIE_NAME: "session-123"},
    )

    assert response.status_code == 204
    set_cookie_header = response.headers.get("set-cookie", "")
    assert "spotify_session_id=" in set_cookie_header
