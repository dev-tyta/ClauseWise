"""
RiskClassifier — LLM-as-classifier using the risk ontology + 3 few-shot examples.

Performs multi-label classification across 8 risk categories with severity levels.
One GPT-4o-mini call per clause (~$0.003).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

RISK_CATEGORIES = [
    "automatic_renewal",
    "broad_liability",
    "vague_cancellation",
    "one_sided_indemnity",
    "hidden_penalties",
    "excessive_data_sharing",
    "unclear_dispute_resolution",
    "missing_refund_terms",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

# 3 few-shot examples (clause_text → expected predictions) loaded at runtime
FEW_SHOT_EXAMPLES: list[dict] = []


@dataclass
class RiskPrediction:
    category: str
    severity: str
    confidence: float
    rationale: str
    recommended_action: str


class RiskClassifier:
    def __init__(
        self,
        ontology_path: str = "data/ontology/risk_ontology.yaml",
        model: str = "gpt-4o-mini",
        openai_api_key: str = "",
    ) -> None:
        self.ontology_path = Path(ontology_path)
        self.model = model
        self._openai_api_key = openai_api_key
        self._ontology: dict = {}
        self._client = None

    def load(self) -> None:
        """Load ontology YAML and initialise OpenAI client."""
        with self.ontology_path.open() as f:
            self._ontology = yaml.safe_load(f)
        raise NotImplementedError  # also init openai client

    def classify(self, clause_text: str, clause_type: str) -> list[RiskPrediction]:
        """Multi-label risk classification. Returns list of detected risks (may be empty)."""
        raise NotImplementedError

    def _build_prompt(self, clause_text: str, clause_type: str) -> list[dict]:
        raise NotImplementedError

    def _parse_response(self, raw: str) -> list[RiskPrediction]:
        raise NotImplementedError
