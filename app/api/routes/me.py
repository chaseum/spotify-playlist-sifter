from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.services.spotify_client import (
    SpotifyClientError,
    add_items_to_playlist_for_session,
    create_my_playlist_for_session,
    get_current_user_for_session,
    get_playlist_items_for_session,
    get_my_playlists_for_session,
    remove_from_my_library_for_session,
    save_to_my_library_for_session,
    search_tracks_for_session,
)

SESSION_COOKIE_NAME = "spotify_session_id"

router = APIRouter(tags=["spotify-me"])


class CreatePlaylistRequest(BaseModel):
    name: str
    description: str | None = None
    public: bool = False


class AddPlaylistItemsRequest(BaseModel):
    uris: list[str]


class LibraryItemsRequest(BaseModel):
    uris: list[str]


@router.get("/api/me")
async def get_me(request: Request) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authorized")

    try:
        return get_current_user_for_session(session_id)
    except SpotifyClientError as exc:
        status_code = 401 if exc.auth_error else exc.status_code
        raise HTTPException(status_code=status_code, detail=exc.message) from exc


@router.get("/api/me/playlists")
async def get_my_playlists(
    request: Request,
    limit: int = Query(default=10, ge=1, le=10),
    offset: int = Query(default=0, ge=0),
) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authorized")

    try:
        return get_my_playlists_for_session(session_id=session_id, limit=limit, offset=offset)
    except SpotifyClientError as exc:
        status_code = 401 if exc.auth_error else exc.status_code
        raise HTTPException(status_code=status_code, detail=exc.message) from exc


@router.post("/api/me/playlists")
async def create_my_playlist(request: Request, payload: CreatePlaylistRequest) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authorized")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Playlist name is required")

    description = payload.description
    if isinstance(description, str):
        description = description.strip()
        if not description:
            description = None

    try:
        return create_my_playlist_for_session(
            session_id=session_id,
            name=name,
            description=description,
            public=payload.public,
        )
    except SpotifyClientError as exc:
        status_code = 401 if exc.auth_error else exc.status_code
        raise HTTPException(status_code=status_code, detail=exc.message) from exc


@router.get("/api/me/playlists/{playlist_id}/items")
async def get_playlist_items(
    playlist_id: str,
    request: Request,
    limit: int = Query(default=25, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authorized")

    try:
        return get_playlist_items_for_session(
            session_id=session_id,
            playlist_id=playlist_id,
            limit=limit,
            offset=offset,
        )
    except SpotifyClientError as exc:
        status_code = 401 if exc.auth_error else exc.status_code
        raise HTTPException(status_code=status_code, detail=exc.message) from exc


@router.get("/api/search")
async def search_tracks(
    request: Request,
    q: str = Query(min_length=1),
    _search_type: str = Query(default="track", alias="type", pattern="^track$"),
    limit: int = Query(default=10, ge=1, le=10),
    offset: int = Query(default=0, ge=0),
) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authorized")

    try:
        return search_tracks_for_session(
            session_id=session_id,
            query=q,
            limit=limit,
            offset=offset,
        )
    except SpotifyClientError as exc:
        status_code = 401 if exc.auth_error else exc.status_code
        raise HTTPException(status_code=status_code, detail=exc.message) from exc


@router.post("/api/playlists/{playlist_id}/items")
async def add_playlist_items(
    playlist_id: str,
    request: Request,
    payload: AddPlaylistItemsRequest,
) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authorized")

    uris = [uri.strip() for uri in payload.uris if isinstance(uri, str) and uri.strip()]
    if not uris:
        raise HTTPException(status_code=422, detail="At least one track URI is required")

    try:
        return add_items_to_playlist_for_session(
            session_id=session_id,
            playlist_id=playlist_id,
            uris=uris,
        )
    except SpotifyClientError as exc:
        status_code = 401 if exc.auth_error else exc.status_code
        raise HTTPException(status_code=status_code, detail=exc.message) from exc


@router.put("/api/library")
async def save_to_my_library(request: Request, payload: LibraryItemsRequest) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authorized")

    uris = [uri.strip() for uri in payload.uris if isinstance(uri, str) and uri.strip()]
    if not uris:
        raise HTTPException(status_code=422, detail="At least one Spotify URI is required")

    try:
        return save_to_my_library_for_session(session_id=session_id, uris=uris)
    except SpotifyClientError as exc:
        status_code = 401 if exc.auth_error else exc.status_code
        raise HTTPException(status_code=status_code, detail=exc.message) from exc


@router.delete("/api/library")
async def remove_from_my_library(request: Request, payload: LibraryItemsRequest) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authorized")

    uris = [uri.strip() for uri in payload.uris if isinstance(uri, str) and uri.strip()]
    if not uris:
        raise HTTPException(status_code=422, detail="At least one Spotify URI is required")

    try:
        return remove_from_my_library_for_session(session_id=session_id, uris=uris)
    except SpotifyClientError as exc:
        status_code = 401 if exc.auth_error else exc.status_code
        raise HTTPException(status_code=status_code, detail=exc.message) from exc
