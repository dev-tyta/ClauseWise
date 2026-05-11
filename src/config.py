from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # Local models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # Paths
    chroma_path: str = "data/indexes/chroma_db"
    bm25_index_path: str = "data/indexes/bm25_index.pkl"
    evidence_corpus_path: str = "data/evidence_corpus/evidence.jsonl"
    benchmark_path: str = "data/benchmark/benchmark.jsonl"
    risk_ontology_path: str = "data/ontology/risk_ontology.yaml"

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_config: int = 5  # production default: hybrid + reranker + clause-type filter

    # Verification
    fidelity_threshold: float = 0.7
    nli_weight: float = 0.4
    llm_judge_weight: float = 0.6


settings = Settings()
