from fastapi import APIRouter, HTTPException

from app.core.config import settings

DEFAULT_REDIRECT_URI = "http://127.0.0.1:8000/"
DEFAULT_SCOPES = (
    "user-read-private user-read-email playlist-modify-private "
    "playlist-modify-public user-library-read user-library-modify"
)
DEFAULT_AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
DEFAULT_TOKEN_URL = "https://accounts.spotify.com/api/token"

router = APIRouter(tags=["config"])


@router.get("/api/config")
async def get_public_config() -> dict[str, str]:
    client_id = settings.spotify_client_id.strip()
    if not client_id:
        raise HTTPException(status_code=500, detail="SPOTIFY_CLIENT_ID is not configured")

    redirect_uri = settings.spotify_redirect_uri.strip() or DEFAULT_REDIRECT_URI
    scopes = settings.spotify_scopes.strip() or DEFAULT_SCOPES
    authorize_url = settings.spotify_authorize_url.strip() or DEFAULT_AUTHORIZE_URL
    token_url = settings.spotify_token_url.strip() or DEFAULT_TOKEN_URL
    return {
        "spotify_client_id": client_id,
        "spotify_redirect_uri": redirect_uri,
        "spotify_scopes": scopes,
        "spotify_authorize_url": authorize_url,
        "spotify_token_url": token_url,
    }
