#!/usr/bin/env python3

import argparse
import contextlib
import io
import os

from dotenv import load_dotenv
from lib.hybrid_search import HybridSearch
from openai import OpenAI
from openai import OpenAIError
from lib.search_utils import load_movies


def enhance_query_spell(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return query

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        corrected = (response.choices[0].message.content or "").strip().strip('"')
        return corrected or query
    except OpenAIError:
        return query


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
    rrf_parser = subparsers.add_parser("rrf-search", help="Run RRF hybrid search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell"],
        help="Query enhancement method",
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
        case "rrf-search":
            search_query = args.query
            if args.enhance == "spell":
                enhanced_query = enhance_query_spell(args.query)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
                search_query = enhanced_query
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                documents = load_movies()
                hybrid = HybridSearch(documents)
                results = hybrid.rrf_search(search_query, k=args.k, limit=args.limit)
            for i, result in enumerate(results, start=1):
                bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "-"
                semantic_rank = (
                    result["semantic_rank"] if result["semantic_rank"] is not None else "-"
                )
                print(f"{i}. {result['title']}")
                print(f"   RRF Score: {result['rrf']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {result['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
