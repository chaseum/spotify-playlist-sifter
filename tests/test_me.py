import pytest
from fastapi.testclient import TestClient

import app.api.routes.me as me_route
import app.services.spotify_client as spotify_client
from app.main import app

client = TestClient(app)


def test_api_me_returns_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        me_route,
        "get_current_user_for_session",
        lambda session_id: {"display_name": "Test User"},
    )

    response = client.get("/api/me", cookies={me_route.SESSION_COOKIE_NAME: "session-123"})

    assert response.status_code == 200
    assert response.json() == {"display_name": "Test User"}


def test_get_current_user_for_session_refreshes_and_retries(monkeypatch) -> None:
    session_id = "session-123"
    initial_tokens = {"access_token": "expired-access", "refresh_token": "refresh-123"}
    state: dict[str, object] = {"calls": 0, "stored_tokens": None}

    monkeypatch.setattr(spotify_client, "get_tokens", lambda value: initial_tokens if value == session_id else None)
    monkeypatch.setattr(spotify_client, "clear_tokens", lambda _: None)

    def fake_get_current_user(access_token: str) -> dict:
        state["calls"] = int(state["calls"]) + 1
        if state["calls"] == 1:
            raise spotify_client.SpotifyClientError(
                status_code=401,
                message="Expired token",
                auth_error=True,
            )
        assert access_token == "new-access"
        return {"display_name": "Refreshed User"}

    monkeypatch.setattr(spotify_client, "get_current_user", fake_get_current_user)
    monkeypatch.setattr(
        spotify_client,
        "_refresh_access_token",
        lambda refresh_token: {"access_token": "new-access", "expires_in": 3600},
    )

    def fake_store_tokens(stored_session_id: str, token_data: dict) -> None:
        state["stored_tokens"] = (stored_session_id, token_data)

    monkeypatch.setattr(spotify_client, "store_tokens", fake_store_tokens)

    profile = spotify_client.get_current_user_for_session(session_id)

    assert profile == {"display_name": "Refreshed User"}
    assert state["calls"] == 2
    assert state["stored_tokens"] == (
        session_id,
        {"access_token": "new-access", "refresh_token": "refresh-123", "expires_in": 3600},
    )


def test_get_current_user_for_session_fails_when_refresh_token_missing(monkeypatch) -> None:
    monkeypatch.setattr(spotify_client, "get_tokens", lambda _: {"access_token": "expired-access"})
    monkeypatch.setattr(
        spotify_client,
        "get_current_user",
        lambda _: (_ for _ in ()).throw(
            spotify_client.SpotifyClientError(
                status_code=401,
                message="Expired token",
                auth_error=True,
            )
        ),
    )
    clear_state = {"called": False}
    monkeypatch.setattr(
        spotify_client,
        "clear_tokens",
        lambda _: clear_state.__setitem__("called", True),
    )

    with pytest.raises(spotify_client.SpotifyClientError) as exc_info:
        spotify_client.get_current_user_for_session("session-123")

    assert exc_info.value.status_code == 401
    assert exc_info.value.auth_error is True
    assert clear_state["called"] is True
