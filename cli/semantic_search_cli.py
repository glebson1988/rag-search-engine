#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text
from lib.semantic_search import embed_query_text
from lib.semantic_search import SemanticSearch
from lib.semantic_search import verify_embeddings
from lib.semantic_search import verify_model
from lib.search_utils import load_movies


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify semantic model setup")
    subparsers.add_parser("verify_embeddings", help="Verify and cache document embeddings")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for input text")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for search query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    search_parser = subparsers.add_parser("search", help="Search movies using semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embedding = embed_text(args.text)
            print(embedding)
            print(f"Dimensions: {len(embedding)}")
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            documents = load_movies()
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                description = result["description"]
                preview = description[:120] + "..." if len(description) > 120 else description
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {preview}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
