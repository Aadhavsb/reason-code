"""FastAPI entrypoint for the decision-support demo."""

from typing import Annotated

from fastapi import Depends, FastAPI

from reason_code.decision import DecisionService
from reason_code.schemas import Application, DecisionResponse
from reason_code.scoring import DemoHeuristicScorer

app = FastAPI(
    title="Reason Code",
    description="Decision-support demo only; not production underwriting or lending advice.",
    version="0.1.0",
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
