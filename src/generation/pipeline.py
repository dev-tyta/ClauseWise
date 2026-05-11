"""
GenerationPipeline — 5 variants from extractive to full ClauseWise RAG.

Variant 1: Extractive (TextRank via sumy, no LLM)
Variant 2: Vanilla LLM (GPT-4o-mini, clause text only)
Variant 3: Prompted LLM (structured prompt + JSON output, no evidence)
Variant 4: Standard RAG (Variant 3 + hybrid retrieval evidence)
Variant 5: ClauseWise full (Variant 3 + Config-5 evidence + risk ontology context)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum

from src.retrieval.engine import RetrievedEvidence


class GenerationVariant(IntEnum):
    EXTRACTIVE = 1
    VANILLA_LLM = 2
    PROMPTED_LLM = 3
    STANDARD_RAG = 4
    CLAUSEWISE = 5


@dataclass
class GeneratedExplanation:
    clause_id: str
    variant: GenerationVariant
    plain_english: str
    risk_flags: list[str]
    severity: str  # critical | high | medium | low
    recommended_action: str
    confidence: float
    evidence_used: list[RetrievedEvidence] = field(default_factory=list)
    seek_legal_advice: bool = False
    raw_response: str = ""


class GenerationPipeline:
    def __init__(
        self,
        variant: GenerationVariant = GenerationVariant.CLAUSEWISE,
        openai_api_key: str = "",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.variant = variant
        self.model = model
        self._client = None
        self._openai_api_key = openai_api_key

    def load(self) -> None:
        """Initialise OpenAI client (and sumy resources for Variant 1)."""
        raise NotImplementedError

    def generate(
        self,
        clause_id: str,
        clause_text: str,
        clause_type: str,
        evidence: list[RetrievedEvidence] | None = None,
        risk_context: dict | None = None,
    ) -> GeneratedExplanation:
        match self.variant:
            case GenerationVariant.EXTRACTIVE:
                return self._extractive(clause_id, clause_text)
            case GenerationVariant.VANILLA_LLM:
                return self._vanilla_llm(clause_id, clause_text)
            case GenerationVariant.PROMPTED_LLM:
                return self._prompted_llm(clause_id, clause_text, clause_type)
            case GenerationVariant.STANDARD_RAG:
                return self._rag(clause_id, clause_text, clause_type, evidence or [])
            case GenerationVariant.CLAUSEWISE:
                return self._clausewise(
                    clause_id, clause_text, clause_type, evidence or [], risk_context or {}
                )
            case _:
                raise ValueError(f"Unknown variant: {self.variant}")

    # ------------------------------------------------------------------
    # Variant implementations
    # ------------------------------------------------------------------

    def _extractive(self, clause_id: str, clause_text: str) -> GeneratedExplanation:
        """TextRank summary via sumy. No LLM call."""
        raise NotImplementedError

    def _vanilla_llm(self, clause_id: str, clause_text: str) -> GeneratedExplanation:
        raise NotImplementedError

    def _prompted_llm(
        self, clause_id: str, clause_text: str, clause_type: str
    ) -> GeneratedExplanation:
        raise NotImplementedError

    def _rag(
        self,
        clause_id: str,
        clause_text: str,
        clause_type: str,
        evidence: list[RetrievedEvidence],
    ) -> GeneratedExplanation:
        raise NotImplementedError

    def _clausewise(
        self,
        clause_id: str,
        clause_text: str,
        clause_type: str,
        evidence: list[RetrievedEvidence],
        risk_context: dict,
    ) -> GeneratedExplanation:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict]) -> dict:
        """Call GPT-4o-mini and parse JSON response."""
        raise NotImplementedError

    def _parse_llm_response(
        self,
        clause_id: str,
        raw: str,
        variant: GenerationVariant,
        evidence: list[RetrievedEvidence],
    ) -> GeneratedExplanation:
        raise NotImplementedError
