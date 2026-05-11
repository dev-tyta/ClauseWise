import uuid

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    ClauseItem,
    FollowUpRequest,
    FollowUpResponse,
    SimplifyRequest,
    SimplifyResponse,
    StudyLogEvent,
    UploadResponse,
)

app = FastAPI(title="ClauseWise API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/simplify", response_model=SimplifyResponse)
async def simplify(request: SimplifyRequest) -> SimplifyResponse:
    """
    Full pipeline: clause → retrieval (Config N) → generation (Variant N)
    → risk classification → fidelity verification → response.
    """
    raise NotImplementedError


@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    """Parse uploaded contract PDF/DOCX → extract clause list."""
    if file.content_type not in ("application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        raise HTTPException(status_code=415, detail="Only PDF and DOCX are supported")
    raise NotImplementedError


@app.post("/api/v1/followup", response_model=FollowUpResponse)
async def followup(request: FollowUpRequest) -> FollowUpResponse:
    """Answer a follow-up question about a previously processed clause."""
    raise NotImplementedError


@app.get("/api/v1/clause/{clause_id}")
async def get_clause(clause_id: str) -> dict:
    """Retrieve clause details and all generated explanations by variant."""
    raise NotImplementedError


@app.post("/api/v1/study/log")
async def log_study_event(event: StudyLogEvent) -> dict:
    """Persist interaction event from the user study frontend."""
    raise NotImplementedError
