from fastapi import APIRouter, HTTPException, Request

from app.services.spotify_client import SpotifyClientError, get_current_user_for_session

SESSION_COOKIE_NAME = "spotify_session_id"

router = APIRouter(tags=["spotify-me"])


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
