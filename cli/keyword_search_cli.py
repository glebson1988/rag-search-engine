#!/usr/bin/env python3

import argparse
import math

from lib.keyword_search import InvertedIndex
from lib.keyword_search import bm25_idf_command
from lib.search_utils import tokenize_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using inverted index")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build and cache the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to inspect")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to inspect")
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to inspect")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

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

        case "tf":
            try:
                index = InvertedIndex()
                index.load()
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            print(index.get_tf(args.doc_id, args.term))

        case "idf":
            try:
                index = InvertedIndex()
                index.load()
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(args.term))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            try:
                index = InvertedIndex()
                index.load()
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            tf = index.get_tf(args.doc_id, args.term)
            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(args.term))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
