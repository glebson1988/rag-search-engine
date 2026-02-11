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
            # 1. Load Movies Data
            try:
                with open("data/movies.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print("Error: data/movies.json file not found.")
                return

            # 2. Load Stop Words
            try:
                with open("data/stopwords.txt", "r", encoding="utf-8") as f:
                    # .splitlines() creates a list of words from the file
                    stop_words = set(f.read().splitlines())
            except FileNotFoundError:
                print("Warning: data/stopwords.txt not found. Proceeding without stop words.")
                stop_words = set()

            # 3. Setup Punctuation Removal
            translator = str.maketrans("", "", string.punctuation)

            # 4. Tokenize and Filter Query
            # lowercase -> remove punctuation -> split -> remove stop words
            query_raw = args.query.lower().translate(translator)
            query_tokens = [word for word in query_raw.split() if word not in stop_words]

            results = []

            # 5. Search Loop
            for movie in data.get("movies", []):
                title = movie.get("title", "")

                # Tokenize and Filter Title
                title_raw = title.lower().translate(translator)
                title_tokens = [word for word in title_raw.split() if word not in stop_words]

                # Matching logic: Check if any query token is a substring of any title token
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

            # 6. Sort and Truncate
            results.sort(key=lambda x: x.get("id", 0))
            final_results = results[:5]

            # 7. Print Results
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
