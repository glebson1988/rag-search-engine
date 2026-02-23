#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
import os
import re
import time

from dotenv import load_dotenv
from lib.hybrid_search import HybridSearch
from openai import OpenAI
from openai import OpenAIError
from lib.search_utils import load_movies
from sentence_transformers import CrossEncoder


def _clean_enhanced_query(text: str, fallback: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return fallback
    # Remove common assistant prefixes and wrapping quotes.
    prefixes = ("Corrected:", "Rewritten query:", "Rewritten:", "Query:")
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.strip().strip('"').strip("'")
    return cleaned or fallback


def _enhance_query_with_groq(query: str, method: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return query

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    if method == "spell":
        prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    elif method == "rewrite":
        prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    elif method == "expand":
        prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    else:
        return query

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        enhanced = response.choices[0].message.content or ""
        return _clean_enhanced_query(enhanced, query)
    except OpenAIError:
        return query


def _extract_score_0_10(text: str) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text or "")
    if not match:
        return 0.0
    value = float(match.group(1))
    return max(0.0, min(10.0, value))


def rerank_individual_with_groq(query: str, docs: list[dict], sleep_seconds: int = 3) -> list[dict]:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        for doc in docs:
            doc["rerank_score"] = 0.0
        return docs

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    reranked = []
    for i, doc in enumerate(docs):
        document_text = doc.get("document", doc.get("description", ""))
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {document_text}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.choices[0].message.content or ""
            score = _extract_score_0_10(raw_text)
        except OpenAIError:
            score = 0.0

        enriched = dict(doc)
        enriched["rerank_score"] = score
        reranked.append(enriched)

        if i < len(docs) - 1:
            time.sleep(sleep_seconds)

    reranked.sort(key=lambda item: (item["rerank_score"], item["rrf"]), reverse=True)
    return reranked


def rerank_batch_with_groq(query: str, docs: list[dict]) -> list[dict]:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        reranked = []
        for i, doc in enumerate(docs, start=1):
            enriched = dict(doc)
            enriched["rerank_rank"] = i
            reranked.append(enriched)
        return reranked

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    doc_list_str = "\n".join(
        f"- id={doc.get('id')} | title={doc.get('title', '')} | document={doc.get('document', doc.get('description', ''))}"
        for doc in docs
    )
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            ranked_ids = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", raw)
            ranked_ids = json.loads(match.group(0)) if match else []
    except OpenAIError:
        ranked_ids = []

    rank_map: dict[int, int] = {}
    for idx, doc_id in enumerate(ranked_ids, start=1):
        try:
            rank_map[int(doc_id)] = idx
        except (TypeError, ValueError):
            continue

    reranked = []
    fallback_rank = len(docs) + 1
    for i, doc in enumerate(docs, start=1):
        enriched = dict(doc)
        enriched["rerank_rank"] = rank_map.get(doc["id"], fallback_rank + i)
        reranked.append(enriched)

    reranked.sort(key=lambda item: item["rerank_rank"])
    return reranked


def rerank_cross_encoder(query: str, docs: list[dict]) -> list[dict]:
    pairs = []
    for doc in docs:
        document_text = doc.get("document", doc.get("description", ""))
        pairs.append([query, f"{doc.get('title', '')} - {document_text}"])

    # Suppress model loading logs so rerank output remains readable.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        scores = cross_encoder.predict(pairs)

    reranked = []
    for doc, score in zip(docs, scores):
        enriched = dict(doc)
        enriched["cross_encoder_score"] = float(score)
        reranked.append(enriched)

    reranked.sort(key=lambda item: item["cross_encoder_score"], reverse=True)
    return reranked


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
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM reranking method",
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
            if args.enhance in ("spell", "rewrite", "expand"):
                enhanced_query = _enhance_query_with_groq(args.query, args.enhance)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
                search_query = enhanced_query
            rrf_limit = (
                args.limit * 5
                if args.rerank_method in ("individual", "batch", "cross_encoder")
                else args.limit
            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                documents = load_movies()
                hybrid = HybridSearch(documents)
                results = hybrid.rrf_search(search_query, k=args.k, limit=rrf_limit)

            if args.rerank_method == "individual":
                print(f"Reranking top {args.limit} results using individual method...")
                results = rerank_individual_with_groq(search_query, results)
            elif args.rerank_method == "batch":
                print(f"Reranking top {args.limit} results using batch method...")
                results = rerank_batch_with_groq(search_query, results)
            elif args.rerank_method == "cross_encoder":
                print(f"Reranking top {rrf_limit} results using cross_encoder method...\n")
                results = rerank_cross_encoder(search_query, results)

            print(f"Reciprocal Rank Fusion Results for '{search_query}' (k={args.k}):\n")
            final_results = results[: args.limit]
            for i, result in enumerate(final_results, start=1):
                bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "-"
                semantic_rank = (
                    result["semantic_rank"] if result["semantic_rank"] is not None else "-"
                )
                print(f"{i}. {result['title']}")
                if args.rerank_method == "individual":
                    print(f"   Rerank Score: {result.get('rerank_score', 0.0):.3f}/10")
                if args.rerank_method == "batch":
                    print(f"   Rerank Rank: {result.get('rerank_rank', '-')}")
                if args.rerank_method == "cross_encoder":
                    print(f"   Cross Encoder Score: {result.get('cross_encoder_score', 0.0):.3f}")
                print(f"   RRF Score: {result['rrf']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                document_text = result.get("document", result.get("description", ""))
                print(f"   {document_text[:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
