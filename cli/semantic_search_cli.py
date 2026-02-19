#!/usr/bin/env python3

import argparse
import re

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
    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size word chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Chunk size in words"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of overlapping words between chunks"
    )
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into sentence-based semantic chunks"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Maximum number of sentences per chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of overlapping sentences between chunks"
    )
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
        case "chunk":
            words = args.text.split()
            if args.chunk_size <= 0:
                raise ValueError("chunk size must be greater than 0")
            if args.overlap < 0:
                raise ValueError("overlap must be greater than or equal to 0")
            if args.overlap >= args.chunk_size:
                raise ValueError("overlap must be less than chunk size")

            chunks = []
            start = 0
            step = args.chunk_size - args.overlap
            while start < len(words):
                chunk_words = words[start : start + args.chunk_size]
                chunks.append(" ".join(chunk_words))
                start += step

            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            if args.max_chunk_size <= 0:
                raise ValueError("max chunk size must be greater than 0")
            if args.overlap < 0:
                raise ValueError("overlap must be greater than or equal to 0")
            if args.overlap >= args.max_chunk_size:
                raise ValueError("overlap must be less than max chunk size")

            sentences = [s for s in re.split(r"(?<=[.!?])\s+", args.text.strip()) if s]
            chunks = []
            start = 0
            step = args.max_chunk_size - args.overlap
            while start < len(sentences):
                chunk_sentences = sentences[start : start + args.max_chunk_size]
                chunks.append(" ".join(chunk_sentences))
                start += step

            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
