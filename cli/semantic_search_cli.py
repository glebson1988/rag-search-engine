#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text
from lib.semantic_search import embed_query_text
from lib.semantic_search import verify_embeddings
from lib.semantic_search import verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify semantic model setup")
    subparsers.add_parser("verify_embeddings", help="Verify and cache document embeddings")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for input text")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for search query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
