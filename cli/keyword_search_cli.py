#!/usr/bin/env python3

import json
import argparse
import string

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            try:
                with open("data/movies.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print("Error: movies.json file not found.")
                return

            translator = str.maketrans("", "", string.punctuation)

            query_raw = args.query.lower().translate(translator)
            query_tokens = query_raw.split()

            results = []

            for movie in data.get("movies", []):
                title_raw = movie.get("title", "").lower().translate(translator)

                title_tokens = title_raw.split()

                match_found = False
                for q_token in query_tokens:
                    for t_token in title_tokens:
                        if q_token in t_token:
                            match_found = True
                            break
                    if match_found:
                        break

                if match_found:
                    results.append(movie)

            results.sort(key=lambda x: x.get("id", 0))
            final_results = results[:5]

            print(f"Searching for: '{args.query}'")
            if not final_results:
                print("No movies found.")
            else:
                for i, movie in enumerate(final_results, 1):
                    print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
