#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    root_dir = Path(__file__).resolve().parents[1]
    dataset_path = root_dir / "data" / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    documents = load_movies()
    hybrid = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden_dataset.get("test_cases", []):
        query = test_case.get("query", "")
        relevant_titles = test_case.get("relevant_docs", [])

        results = hybrid.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [result.get("title", "") for result in results]

        relevant_set = set(relevant_titles)
        relevant_retrieved = sum(1 for title in retrieved_titles if title in relevant_set)
        precision_at_k = relevant_retrieved / len(retrieved_titles) if retrieved_titles else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision_at_k:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}\n")


if __name__ == "__main__":
    main()
