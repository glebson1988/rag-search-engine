#!/usr/bin/env python3

import argparse
import contextlib
import io

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")
    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Run weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for BM25 score"
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                return

            min_score = min(scores)
            max_score = max(scores)

            if min_score == max_score:
                normalized_scores = [1.0 for _ in scores]
            else:
                normalized_scores = [
                    (score - min_score) / (max_score - min_score) for score in scores
                ]

            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                documents = load_movies()
                hybrid = HybridSearch(documents)
                results = hybrid.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"   Hybrid Score: {result['hybrid']:.3f}")
                print(f"   BM25: {result['bm25']:.3f}, Semantic: {result['semantic']:.3f}")
                print(f"   {result['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
