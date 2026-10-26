"""Dashboard entrypoint overlay (Ruche and Tribute): one account system and real inference only, points only.

Wraps whatever `app` the dashboard image's own server builds (`/app/server_orig.py`), so it holds whichever version of
that code the image carries:
- every local `/v1/*` route is removed and `/v1/*` is forwarded unchanged (headers, body, streaming) to the HiveMind
  OpenRouter, where keys live in the one ledger and completions come from live producers. A dashboard build that answers
  `/v1/chat/completions` itself, without a key and from a template, is therefore never reachable;
- `/signup`, `/login`, `/openrouter`, `/playground` redirect to the router's own pages;
- pages and account JSON go through `to_points_wording` (HiveMind Reflexion Points only); completions are never rewritten.
"""

from __future__ import annotations

import os
import runpy
from collections.abc import Awaitable, Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from hivemind.router.points_wording import to_points_wording

ROUTER: str = os.environ.get("HIVEMIND_ROUTER_URL", "https://hivemind-openrouter-1003063861791.us-central1.run.app").rstrip("/")
ROUTER_PUBLIC: str = os.environ.get("HIVEMIND_ROUTER_PUBLIC_URL", "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app").rstrip("/")
REDIRECTS: dict[str, str] = {"/signup": "/signup", "/login": "/login", "/openrouter": "/", "/playground": "/"}
HOP_BY_HOP: frozenset[str] = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
                                        "transfer-encoding", "upgrade", "host", "content-length", "content-encoding"})
REWRITTEN_TYPES: tuple[str, ...] = ("text/html", "application/json", "application/javascript", "text/javascript")
NEVER_REWRITTEN: tuple[str, ...] = ("/v1/",)

app: FastAPI = runpy.run_path(os.environ.get("HIVEMIND_ORIG_SERVER", "/app/server_orig.py"), run_name="hivemind_dashboard_server")["app"]
app.router.routes = [r for r in app.router.routes
                     if not (getattr(r, "path", "").startswith("/v1/") or getattr(r, "path", "") in REDIRECTS)]
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def forward_to_router(path: str, request: Request) -> Response:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    upstream = _client.build_request(request.method, f"{ROUTER}/v1/{path}", params=request.query_params, headers=headers,
                                     content=await request.body())
    try:
        resp = await _client.send(upstream, stream=True)
    except httpx.HTTPError:
        return JSONResponse(status_code=502, content={"error": {"message": "The HiveMind router is not reachable.", "type": "api_error",
                                                                "code": "router_unreachable"}})
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
    return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=out_headers, background=BackgroundTask(resp.aclose))


def _redirect_to(target: str) -> Callable[[], RedirectResponse]:
    def handler() -> RedirectResponse:
        return RedirectResponse(ROUTER_PUBLIC + target, status_code=307)
    return handler


for _local, _target in REDIRECTS.items():
    app.add_api_route(_local, _redirect_to(_target), methods=["GET"], include_in_schema=False)


@app.middleware("http")
async def points_only(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    response = await call_next(request)
    if request.url.path.startswith(NEVER_REWRITTEN) or not response.headers.get("content-type", "").startswith(REWRITTEN_TYPES):
        return response
    body = b"".join([chunk async for chunk in response.body_iterator])  # type: ignore[attr-defined]
    try:
        new_body = to_points_wording(body.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError:
        new_body = body
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return Response(content=new_body, status_code=response.status_code, headers=headers)
