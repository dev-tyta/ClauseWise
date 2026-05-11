"""
RetrievalEngine — 5 configs switchable via the `config` parameter.

Config 1: BM25 only
Config 2: Dense only (MiniLM-L6-v2 + FAISS)
Config 3: Hybrid (reciprocal rank fusion of BM25 + dense)
Config 4: Hybrid + cross-encoder reranker
Config 5: Hybrid + reranker + clause-type metadata filter  ← production
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class RetrievalConfig(IntEnum):
    BM25_ONLY = 1
    DENSE_ONLY = 2
    HYBRID = 3
    HYBRID_RERANKER = 4
    HYBRID_RERANKER_FILTER = 5


@dataclass
class RetrievedEvidence:
    evidence_id: str
    text: str
    source_type: str
    legal_concept: str
    clause_type: str
    citation: str
    score: float


class RetrievalEngine:
    def __init__(
        self,
        config: RetrievalConfig = RetrievalConfig.HYBRID_RERANKER_FILTER,
        bm25_index_path: str = "data/indexes/bm25_index.pkl",
        chroma_path: str = "data/indexes/chroma_db",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.config = config
        self.bm25_index_path = Path(bm25_index_path)
        self.chroma_path = chroma_path
        self.reranker_model_name = reranker_model
        self.embedding_model_name = embedding_model

        self._bm25 = None
        self._corpus_docs: list[dict] = []
        self._chroma_collection = None
        self._embedder = None
        self._reranker = None

    def load(self) -> None:
        """Load all indexes and models required by the active config."""
        if self.config in (
            RetrievalConfig.BM25_ONLY,
            RetrievalConfig.HYBRID,
            RetrievalConfig.HYBRID_RERANKER,
            RetrievalConfig.HYBRID_RERANKER_FILTER,
        ):
            self._load_bm25()

        if self.config in (
            RetrievalConfig.DENSE_ONLY,
            RetrievalConfig.HYBRID,
            RetrievalConfig.HYBRID_RERANKER,
            RetrievalConfig.HYBRID_RERANKER_FILTER,
        ):
            self._load_dense()

        if self.config in (
            RetrievalConfig.HYBRID_RERANKER,
            RetrievalConfig.HYBRID_RERANKER_FILTER,
        ):
            self._load_reranker()

    def retrieve(
        self,
        query: str,
        clause_type: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        """Return top-k evidence items for a clause query."""
        match self.config:
            case RetrievalConfig.BM25_ONLY:
                return self._bm25_search(query, top_k)
            case RetrievalConfig.DENSE_ONLY:
                return self._dense_search(query, top_k)
            case RetrievalConfig.HYBRID:
                return self._hybrid_search(query, top_k)
            case RetrievalConfig.HYBRID_RERANKER:
                candidates = self._hybrid_search(query, top_k * 3)
                return self._rerank(query, candidates)[:top_k]
            case RetrievalConfig.HYBRID_RERANKER_FILTER:
                candidates = self._hybrid_search(query, top_k * 3, clause_type_filter=clause_type)
                return self._rerank(query, candidates)[:top_k]
            case _:
                raise ValueError(f"Unknown config: {self.config}")

    # ------------------------------------------------------------------
    # Internal search methods
    # ------------------------------------------------------------------

    def _load_bm25(self) -> None:
        raise NotImplementedError

    def _load_dense(self) -> None:
        raise NotImplementedError

    def _load_reranker(self) -> None:
        raise NotImplementedError

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedEvidence]:
        raise NotImplementedError

    def _dense_search(self, query: str, top_k: int) -> list[RetrievedEvidence]:
        raise NotImplementedError

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        clause_type_filter: str | None = None,
    ) -> list[RetrievedEvidence]:
        """Reciprocal rank fusion of BM25 + dense, with optional clause-type pre-filter."""
        raise NotImplementedError

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[RetrievedEvidence],
        dense_results: list[RetrievedEvidence],
        k: int = 60,
    ) -> list[RetrievedEvidence]:
        raise NotImplementedError

    def _rerank(self, query: str, candidates: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
        raise NotImplementedError
