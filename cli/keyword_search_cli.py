#!/usr/bin/env python3

import argparse

from lib.keyword_search import InvertedIndex
from lib.search_utils import tokenize_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using inverted index")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build and cache the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            try:
                index = InvertedIndex()
                index.load()
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return

            query_tokens = tokenize_text(args.query)
            results: list[dict] = []
            seen_doc_ids: set[int] = set()
            for token in query_tokens:
                for doc_id in index.get_documents(token):
                    if doc_id in seen_doc_ids:
                        continue
                    movie = index.docmap.get(doc_id)
                    if movie is None:
                        continue
                    results.append(movie)
                    seen_doc_ids.add(doc_id)
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break

            print(f"Searching for: '{args.query}'")
            if not results:
                print("No movies found.")
            else:
                for movie in results:
                    print(f"{movie['id']}: {movie['title']}")

        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            print("Inverted index built and saved.")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
