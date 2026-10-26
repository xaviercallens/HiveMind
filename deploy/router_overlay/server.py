"""Router entrypoint overlay: price among recently active consumers, and show points only.

1. `funded_consumer_bids` counts every funded account as a bidder in the Kelly game. With a starter grant on every sign-up,
   idle accounts that once raised their max price keep the clearing price above the default max price of new accounts,
   so a new user's first request is refused (402). This keeps only accounts with a request or a hold in the last
   HIVEMIND_BID_WINDOW_S seconds (default 900; 0 restores the original rule), as PR #182 does in the source.
2. Pages and account JSON are passed through `to_points_wording`: no euro or dollar amount is shown, only HiveMind
   Reflexion Points (HMT). Completions (`/v1/chat/...`, `/v1/completions`) are never rewritten.
"""

from __future__ import annotations

import os
import runpy
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response

from hivemind.router.accounts import RouterStore
from hivemind.router.points_wording import to_points_wording

BID_WINDOW_S: float = float(os.environ.get("HIVEMIND_BID_WINDOW_S", "900") or "900")
REWRITTEN_TYPES: tuple[str, ...] = ("text/html", "application/json", "application/javascript", "text/javascript")
NEVER_REWRITTEN: tuple[str, ...] = ("/v1/chat/", "/v1/completions")
_funded_consumer_bids = RouterStore.funded_consumer_bids


def active_funded_consumer_bids(self: RouterStore, expected_tokens: int) -> dict[str, float]:
    bids = _funded_consumer_bids(self, expected_tokens)
    if BID_WINDOW_S <= 0:
        return bids
    since = time.time() - BID_WINDOW_S
    with self._lock:
        rows = self._conn.execute(
            "SELECT consumer_id AS a FROM usage WHERE utc >= ? UNION SELECT account_id AS a FROM holds WHERE utc >= ?", (since, since)
        ).fetchall()
    active = {str(r["a"]) for r in rows}
    return {a: p for a, p in bids.items() if a in active}


RouterStore.funded_consumer_bids = active_funded_consumer_bids
app: FastAPI = runpy.run_path(os.environ.get("HIVEMIND_ORIG_SERVER", "/app/server_orig.py"), run_name="hivemind_router_server")["app"]


@app.middleware("http")
async def points_only(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    response = await call_next(request)
    if request.url.path.startswith(NEVER_REWRITTEN) or not response.headers.get("content-type", "").startswith(REWRITTEN_TYPES):
        return response
    body = b"".join([chunk async for chunk in response.body_iterator])  # type: ignore[attr-defined]
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        new_body = body
    else:
        new_body = to_points_wording(text).encode("utf-8")
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return Response(content=new_body, status_code=response.status_code, headers=headers)
