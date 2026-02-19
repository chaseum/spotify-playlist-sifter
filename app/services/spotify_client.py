import json
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from app.core.config import settings
from app.services.spotify_oauth import clear_tokens, get_tokens, store_tokens

SPOTIFY_API_BASE_URL = "https://api.spotify.com"


class SpotifyClientError(Exception):
    def __init__(self, status_code: int, message: str, auth_error: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.auth_error = auth_error


def _extract_error_message(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        nested_error = payload.get("error")
        if isinstance(nested_error, dict):
            nested_message = nested_error.get("message")
            if isinstance(nested_message, str) and nested_message:
                return nested_message

        description = payload.get("error_description")
        if isinstance(description, str) and description:
            return description

        if isinstance(nested_error, str) and nested_error:
            return nested_error

        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail

        message = payload.get("message")
        if isinstance(message, str) and message:
            return message

    return fallback


def _decode_json(response_body: str) -> Any:
    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid JSON") from exc


def _spotify_request_json(
    path: str,
    access_token: str,
    method: str = "GET",
    json_payload: dict[str, Any] | None = None,
) -> Any:
    headers = {"Authorization": f"Bearer {access_token}"}
    data: bytes | None = None
    if json_payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_payload).encode("utf-8")

    request = Request(
        f"{SPOTIFY_API_BASE_URL}{path}",
        headers=headers,
        data=data,
        method=method,
    )

    try:
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        payload = None
        if error_body:
            try:
                payload = json.loads(error_body)
            except json.JSONDecodeError:
                payload = None

        if exc.code == 401:
            message = _extract_error_message(payload, "Unauthorized Spotify token")
            raise SpotifyClientError(status_code=exc.code, message=message, auth_error=True) from exc

        message = _extract_error_message(payload, "Spotify API request failed")
        raise SpotifyClientError(status_code=exc.code, message=message) from exc
    except URLError as exc:
        raise SpotifyClientError(status_code=502, message="Spotify API unavailable") from exc

    if not body:
        return {}

    return _decode_json(body)


def _refresh_access_token(refresh_token: str) -> dict[str, Any]:
    payload = urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": settings.spotify_client_id,
        }
    ).encode("utf-8")
    request = Request(
        settings.spotify_token_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        payload = None
        if error_body:
            try:
                payload = json.loads(error_body)
            except json.JSONDecodeError:
                payload = None
        message = _extract_error_message(payload, "Not authorized")
        raise SpotifyClientError(status_code=401, message=message, auth_error=True) from exc
    except URLError as exc:
        raise SpotifyClientError(status_code=502, message="Spotify token endpoint unavailable") from exc

    token_data = _decode_json(body)
    if not isinstance(token_data, dict):
        raise SpotifyClientError(status_code=502, message="Spotify token endpoint returned invalid JSON")

    access_token = token_data.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise SpotifyClientError(status_code=401, message="Not authorized", auth_error=True)

    return token_data


def get_current_user(access_token: str) -> dict[str, Any]:
    profile = _spotify_request_json("/v1/me", access_token)
    if not isinstance(profile, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid profile data")
    return profile


def get_track(access_token: str, track_id: str) -> dict[str, Any]:
    safe_track_id = track_id.strip()
    if not safe_track_id:
        raise SpotifyClientError(status_code=400, message="Track ID is required")

    payload = _spotify_request_json(f"/v1/tracks/{quote(safe_track_id)}", access_token)
    if not isinstance(payload, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid track data")
    return payload


def get_my_playlists(access_token: str, limit: int = 10, offset: int = 0) -> dict[str, Any]:
    safe_limit = max(1, min(10, int(limit)))
    safe_offset = max(0, int(offset))
    query = urlencode({"limit": safe_limit, "offset": safe_offset})
    payload = _spotify_request_json(f"/v1/me/playlists?{query}", access_token)
    if not isinstance(payload, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid playlists data")
    return payload


def get_playlist_items(
    access_token: str,
    playlist_id: str,
    limit: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    safe_playlist_id = playlist_id.strip()
    if not safe_playlist_id:
        raise SpotifyClientError(status_code=400, message="Playlist ID is required")

    safe_limit = max(1, min(50, int(limit)))
    safe_offset = max(0, int(offset))
    query = urlencode({"limit": safe_limit, "offset": safe_offset})
    payload = _spotify_request_json(f"/v1/playlists/{safe_playlist_id}/items?{query}", access_token)
    if not isinstance(payload, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid playlist items data")
    return payload


def create_my_playlist(
    access_token: str,
    name: str,
    description: str | None = None,
    public: bool = False,
) -> dict[str, Any]:
    safe_name = name.strip()
    if not safe_name:
        raise SpotifyClientError(status_code=400, message="Playlist name is required")

    payload: dict[str, Any] = {"name": safe_name, "public": bool(public)}
    if description is not None:
        payload["description"] = description

    response_payload = _spotify_request_json(
        "/v1/me/playlists",
        access_token,
        method="POST",
        json_payload=payload,
    )
    if not isinstance(response_payload, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid playlist data")
    return response_payload


def search_tracks(
    access_token: str,
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> dict[str, Any]:
    safe_query = query.strip()
    if not safe_query:
        raise SpotifyClientError(status_code=400, message="Search query is required")

    safe_limit = max(1, min(10, int(limit)))
    safe_offset = max(0, int(offset))
    search_query = urlencode(
        {
            "q": safe_query,
            "type": "track",
            "limit": safe_limit,
            "offset": safe_offset,
        }
    )
    payload = _spotify_request_json(f"/v1/search?{search_query}", access_token)
    if not isinstance(payload, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid search data")
    return payload


def add_items_to_playlist(
    access_token: str,
    playlist_id: str,
    uris: list[str],
) -> dict[str, Any]:
    safe_playlist_id = playlist_id.strip()
    if not safe_playlist_id:
        raise SpotifyClientError(status_code=400, message="Playlist ID is required")

    safe_uris = [uri.strip() for uri in uris if isinstance(uri, str) and uri.strip()]
    if not safe_uris:
        raise SpotifyClientError(status_code=400, message="At least one track URI is required")

    payload = _spotify_request_json(
        f"/v1/playlists/{safe_playlist_id}/items",
        access_token,
        method="POST",
        json_payload={"uris": safe_uris},
    )
    if not isinstance(payload, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid add-items data")
    return payload


def _spotify_uri_to_url(uri: str) -> str | None:
    parts = uri.split(":")
    if len(parts) != 3 or parts[0] != "spotify":
        return None

    item_type = parts[1].strip()
    item_id = parts[2].strip()
    if not item_type or not item_id:
        return None

    if item_type == "user":
        return f"https://open.spotify.com/user/{item_id}"

    return f"https://open.spotify.com/{item_type}/{item_id}"


def _library_query_from_uris(uris: list[str]) -> str:
    safe_uris = [uri.strip() for uri in uris if isinstance(uri, str) and uri.strip()]
    if not safe_uris:
        raise SpotifyClientError(status_code=400, message="At least one Spotify URI is required")

    # Keep URI-based contract and include URL form for compatibility with /me/library validation variants.
    query_payload: dict[str, str] = {"uris": ",".join(safe_uris)}
    urls = [_spotify_uri_to_url(uri) for uri in safe_uris]
    safe_urls = [url for url in urls if isinstance(url, str) and url]
    if safe_urls:
        query_payload["urls"] = ",".join(safe_urls)

    return urlencode(query_payload)


def save_to_my_library(
    access_token: str,
    uris: list[str],
) -> dict[str, Any]:
    query = _library_query_from_uris(uris)

    payload = _spotify_request_json(
        f"/v1/me/library?{query}",
        access_token,
        method="PUT",
    )
    if not isinstance(payload, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid library save data")
    return payload


def remove_from_my_library(
    access_token: str,
    uris: list[str],
) -> dict[str, Any]:
    query = _library_query_from_uris(uris)

    payload = _spotify_request_json(
        f"/v1/me/library?{query}",
        access_token,
        method="DELETE",
    )
    if not isinstance(payload, dict):
        raise SpotifyClientError(status_code=502, message="Spotify API returned invalid library remove data")
    return payload


def _request_for_session(session_id: str, request_fn: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    token_data = get_tokens(session_id)
    if not token_data:
        raise SpotifyClientError(status_code=401, message="Not authorized", auth_error=True)

    access_token = token_data.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        clear_tokens(session_id)
        raise SpotifyClientError(status_code=401, message="Not authorized", auth_error=True)

    try:
        return request_fn(access_token)
    except SpotifyClientError as exc:
        if exc.status_code != 401:
            raise

    refresh_token = token_data.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        clear_tokens(session_id)
        raise SpotifyClientError(status_code=401, message="Not authorized", auth_error=True)

    refreshed_tokens = _refresh_access_token(refresh_token)
    merged_tokens = {**token_data, **refreshed_tokens}
    if not isinstance(merged_tokens.get("refresh_token"), str) or not merged_tokens.get("refresh_token"):
        merged_tokens["refresh_token"] = refresh_token
    store_tokens(session_id=session_id, token_data=merged_tokens)

    refreshed_access_token = merged_tokens.get("access_token")
    if not isinstance(refreshed_access_token, str) or not refreshed_access_token:
        clear_tokens(session_id)
        raise SpotifyClientError(status_code=401, message="Not authorized", auth_error=True)

    try:
        return request_fn(refreshed_access_token)
    except SpotifyClientError as exc:
        if exc.status_code == 401:
            clear_tokens(session_id)
            raise SpotifyClientError(status_code=401, message="Not authorized", auth_error=True) from exc
        raise


def get_current_user_for_session(session_id: str) -> dict[str, Any]:
    return _request_for_session(session_id, get_current_user)


def get_track_for_session(session_id: str, track_id: str) -> dict[str, Any]:
    return _request_for_session(
        session_id,
        lambda access_token: get_track(access_token=access_token, track_id=track_id),
    )


def get_my_playlists_for_session(session_id: str, limit: int = 10, offset: int = 0) -> dict[str, Any]:
    return _request_for_session(
        session_id,
        lambda access_token: get_my_playlists(access_token=access_token, limit=limit, offset=offset),
    )


def get_playlist_items_for_session(
    session_id: str,
    playlist_id: str,
    limit: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    return _request_for_session(
        session_id,
        lambda access_token: get_playlist_items(
            access_token=access_token,
            playlist_id=playlist_id,
            limit=limit,
            offset=offset,
        ),
    )


def create_my_playlist_for_session(
    session_id: str,
    name: str,
    description: str | None = None,
    public: bool = False,
) -> dict[str, Any]:
    return _request_for_session(
        session_id,
        lambda access_token: create_my_playlist(
            access_token=access_token,
            name=name,
            description=description,
            public=public,
        ),
    )


def search_tracks_for_session(
    session_id: str,
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> dict[str, Any]:
    return _request_for_session(
        session_id,
        lambda access_token: search_tracks(
            access_token=access_token,
            query=query,
            limit=limit,
            offset=offset,
        ),
    )


def add_items_to_playlist_for_session(
    session_id: str,
    playlist_id: str,
    uris: list[str],
) -> dict[str, Any]:
    return _request_for_session(
        session_id,
        lambda access_token: add_items_to_playlist(
            access_token=access_token,
            playlist_id=playlist_id,
            uris=uris,
        ),
    )


def save_to_my_library_for_session(
    session_id: str,
    uris: list[str],
) -> dict[str, Any]:
    return _request_for_session(
        session_id,
        lambda access_token: save_to_my_library(access_token=access_token, uris=uris),
    )


def remove_from_my_library_for_session(
    session_id: str,
    uris: list[str],
) -> dict[str, Any]:
    return _request_for_session(
        session_id,
        lambda access_token: remove_from_my_library(access_token=access_token, uris=uris),
    )
