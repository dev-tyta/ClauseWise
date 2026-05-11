"""
Extract clause text and type from the CUAD dataset.

Usage:
    python -m src.ingestion.extract_cuad --input data/raw/cuad/ --output data/processed/

Output: data/processed/cuad_clauses.jsonl
Each line: {"clause_id", "text", "clause_type", "contract_name", "source": "cuad"}
"""

import argparse
import json
from pathlib import Path

CLAUSE_TYPES = {
    "indemnity",
    "termination",
    "confidentiality",
    "auto_renewal",
    "liability_limitation",
    "payment_terms",
    "dispute_resolution",
    "data_sharing",
    "non_compete",
    "refund_policy",
}


def extract_cuad(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clauses = _parse_annotations(input_path)
    out_file = output_path / "cuad_clauses.jsonl"
    with out_file.open("w") as f:
        for clause in clauses:
            f.write(json.dumps(clause) + "\n")

    print(f"Extracted {len(clauses)} clauses → {out_file}")


def _parse_annotations(input_path: Path) -> list[dict]:
    """Parse CUAD master_clauses.csv and return filtered clause dicts."""
    raise NotImplementedError


def _normalize_clause_type(raw_type: str) -> str | None:
    """Map CUAD column name to internal clause_type slug, or None if not in scope."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CUAD raw directory")
    parser.add_argument("--output", required=True, help="Path to processed output directory")
    args = parser.parse_args()
    extract_cuad(args.input, args.output)
