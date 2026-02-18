from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode

from app.core.config import settings
from app.services.spotify_oauth import (
    build_authorize_url,
    clear_tokens,
    exchange_code_for_tokens,
    generate_code_challenge,
    generate_code_verifier,
    generate_session_id,
    generate_state,
    is_pending_auth_expired,
    pop_pending_auth,
    store_pending_auth,
    store_tokens,
)

SESSION_COOKIE_NAME = "spotify_session_id"
STATE_COOKIE_NAME = "spotify_oauth_state"

router = APIRouter(tags=["auth-spotify"])


def _redirect_to_frontend_callback(
    code: str | None,
    state: str | None,
    error: str | None,
) -> RedirectResponse:
    query_params: dict[str, str] = {}
    if code:
        query_params["code"] = code
    if state:
        query_params["state"] = state
    if error:
        query_params["error"] = error
    if not query_params:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    return RedirectResponse(url=f"/?{urlencode(query_params)}", status_code=302)


@router.get("/auth/spotify/login")
@router.get("/api/auth/spotify/login")
async def spotify_login(request: Request) -> RedirectResponse:
    if not settings.spotify_client_id:
        raise HTTPException(status_code=500, detail="CLIENT_ID is not configured")

    session_id = request.cookies.get(SESSION_COOKIE_NAME) or generate_session_id()
    state = generate_state()
    verifier = generate_code_verifier()
    challenge = generate_code_challenge(verifier)
    store_pending_auth(state=state, session_id=session_id, verifier=verifier)

    authorize_url = build_authorize_url(
        authorize_url=settings.spotify_authorize_url,
        client_id=settings.spotify_client_id,
        redirect_uri=settings.spotify_redirect_uri,
        scopes=settings.spotify_scopes,
        state=state,
        code_challenge=challenge,
    )

    response = RedirectResponse(url=authorize_url, status_code=302)
    response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True, samesite="lax")
    response.set_cookie(
        STATE_COOKIE_NAME,
        state,
        httponly=True,
        max_age=600,
        samesite="lax",
    )
    return response


@router.get("/auth/spotify/callback")
async def spotify_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> RedirectResponse:
    cookie_state = request.cookies.get(STATE_COOKIE_NAME)
    if not cookie_state:
        return _redirect_to_frontend_callback(code=code, state=state, error=error)

    if error:
        raise HTTPException(status_code=400, detail=f"Spotify authorization failed: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
    if not state:
        raise HTTPException(status_code=400, detail="Missing OAuth state")

    if cookie_state != state:
        return _redirect_to_frontend_callback(code=code, state=state, error=error)

    pending_auth = pop_pending_auth(state)
    if not pending_auth:
        return _redirect_to_frontend_callback(code=code, state=state, error=error)
    if is_pending_auth_expired(pending_auth):
        raise HTTPException(status_code=400, detail="OAuth state expired")

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id or pending_auth["session_id"] != session_id:
        raise HTTPException(status_code=400, detail="Invalid OAuth session")

    verifier = str(pending_auth["verifier"])
    try:
        token_data = exchange_code_for_tokens(
            token_url=settings.spotify_token_url,
            client_id=settings.spotify_client_id,
            code=code,
            redirect_uri=settings.spotify_redirect_uri,
            code_verifier=verifier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store_tokens(session_id=session_id, token_data=token_data)

    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(STATE_COOKIE_NAME)
    return response


@router.get("/auth/logout")
@router.get("/api/auth/logout")
async def auth_logout(request: Request) -> Response:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        clear_tokens(session_id)

    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE_NAME)
    response.delete_cookie(STATE_COOKIE_NAME)
    return response
