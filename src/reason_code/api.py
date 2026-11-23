"""FastAPI entrypoint for the decision-support demo."""

import math
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from reason_code.decision import DecisionService
from reason_code.schemas import Application, DecisionResponse
from reason_code.scoring import DemoHeuristicScorer

app = FastAPI(
    title="Reason Code",
    description="Decision-support demo only; not production underwriting or lending advice.",
    version="0.1.0",
)
_UI_PATH = Path(__file__).with_name("static") / "index.html"


@app.get("/", include_in_schema=False)
def local_demo() -> FileResponse:
    """Serve the self-contained, local-only walkthrough interface."""
    if not _UI_PATH.is_file():
        raise HTTPException(status_code=503, detail="Local demo UI asset is unavailable.")
    return FileResponse(_UI_PATH, media_type="text/html")


def _json_safe(value: Any) -> Any:
    """Replace non-finite validation inputs so the 422 response remains valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return JSON-safe 422 details when a client supplies NaN or infinity."""
    return JSONResponse(
        status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
    )


def get_decision_service() -> DecisionService:
    """Build the default demo dependency; tests and hosts may override this."""
    return DecisionService(scorer=DemoHeuristicScorer())


@app.post("/decisions", response_model=DecisionResponse)
def create_decision(
    application: Application,
    service: Annotated[DecisionService, Depends(get_decision_service)],
) -> DecisionResponse:
    """Apply the deterministic demo policy to a validated application record."""
    return service.decide(application)
