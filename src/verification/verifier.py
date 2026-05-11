"""
FidelityVerifier — dual verification pipeline.

Step 1: NLI entailment score via cross-encoder/nli-deberta-v3-base (local, CPU).
Step 2: LLM-as-judge via GPT-4o-mini structured rubric.
Final:  fidelity_score = nli_weight * nli_score + llm_weight * llm_score

Pass/fail threshold configurable (default 0.7).
Error types: hallucination | distortion | omission | overstatement
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.generation.pipeline import GeneratedExplanation
from src.retrieval.engine import RetrievedEvidence

ERROR_TYPES = ["hallucination", "distortion", "omission", "overstatement"]


@dataclass
class VerificationResult:
    explanation_id: str
    nli_entailment_score: float
    llm_judge_score: float
    fidelity_score: float  # weighted combination
    passed: bool
    error_types: list[str] = field(default_factory=list)
    llm_judge_rationale: str = ""


class FidelityVerifier:
    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        model: str = "gpt-4o-mini",
        openai_api_key: str = "",
        threshold: float = 0.7,
        nli_weight: float = 0.4,
        llm_weight: float = 0.6,
    ) -> None:
        self.nli_model_name = nli_model
        self.model = model
        self._openai_api_key = openai_api_key
        self.threshold = threshold
        self.nli_weight = nli_weight
        self.llm_weight = llm_weight
        self._nli = None
        self._client = None

    def load(self) -> None:
        """Load NLI cross-encoder locally and initialise OpenAI client."""
        raise NotImplementedError

    def verify(
        self,
        explanation: GeneratedExplanation,
        original_clause: str,
        evidence: list[RetrievedEvidence],
    ) -> VerificationResult:
        nli_score = self._nli_score(
            premise=original_clause + " " + " ".join(e.text for e in evidence),
            hypothesis=explanation.plain_english,
        )
        llm_score, error_types, rationale = self._llm_judge(
            explanation, original_clause, evidence
        )

        fidelity = self.nli_weight * nli_score + self.llm_weight * llm_score

        return VerificationResult(
            explanation_id=explanation.clause_id,
            nli_entailment_score=nli_score,
            llm_judge_score=llm_score,
            fidelity_score=round(fidelity, 4),
            passed=fidelity >= self.threshold,
            error_types=error_types,
            llm_judge_rationale=rationale,
        )

    def _nli_score(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability from deberta-v3-base NLI cross-encoder."""
        raise NotImplementedError

    def _llm_judge(
        self,
        explanation: GeneratedExplanation,
        original_clause: str,
        evidence: list[RetrievedEvidence],
    ) -> tuple[float, list[str], str]:
        """Return (score 0-1, error_type list, rationale string)."""
        raise NotImplementedError

    def _build_judge_prompt(
        self,
        explanation: GeneratedExplanation,
        original_clause: str,
        evidence: list[RetrievedEvidence],
    ) -> list[dict]:
        raise NotImplementedError
