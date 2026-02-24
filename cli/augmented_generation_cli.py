#!/usr/bin/env python3

import argparse
import contextlib
import io
import os

from dotenv import load_dotenv
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from openai import OpenAI
from openai import OpenAIError


def _build_docs_block(results: list[dict], max_chars_per_doc: int = 220) -> str:
    if not results:
        return "No documents found."
    lines = []
    for i, result in enumerate(results, start=1):
        title = result.get("title", "")
        description = (result.get("description", "") or "").strip()
        if max_chars_per_doc > 0:
            description = description[:max_chars_per_doc]
        lines.append(f"{i}. {title}: {description}")
    return "\n".join(lines)


def _generate_rag_answer(query: str, docs: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is not set."

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()
    except OpenAIError as exc:
        # Retry once with aggressively reduced context if we hit payload/token limits.
        if "Request too large" in str(exc) or "tokens per minute" in str(exc):
            compact_prompt = f"""Answer the query for Hoopla users using ONLY these movie titles.

Query: {query}

Titles:
{docs}

Keep the answer concise and grounded in the provided titles."""
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": compact_prompt}],
                )
                return (response.choices[0].message.content or "").strip()
            except OpenAIError as retry_exc:
                return f"RAG generation failed: {retry_exc}"
        return f"RAG generation failed: {exc}"


def _generate_summary(query: str, results: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is not set."

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()
    except OpenAIError as exc:
        return f"Summary generation failed: {exc}"


def _generate_citations_answer(query: str, documents: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is not set."

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()
    except OpenAIError as exc:
        return f"Citations answer generation failed: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results with an LLM"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of search results"
    )
    citations_parser = subparsers.add_parser(
        "citations", help="Answer with citations based on retrieved documents"
    )
    citations_parser.add_argument("query", type=str, help="Search query to answer")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of search results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                documents = load_movies()
                hybrid = HybridSearch(documents)
                results = hybrid.rrf_search(query, k=60, limit=5)

            docs = _build_docs_block(results, max_chars_per_doc=220)
            answer = _generate_rag_answer(query, docs)

            print("Search Results:")
            for result in results:
                print(f"  - {result.get('title', '')}")

            print("\nRAG Response:")
            print(answer)
        case "summarize":
            query = args.query
            limit = args.limit
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                documents = load_movies()
                hybrid = HybridSearch(documents)
                results = hybrid.rrf_search(query, k=60, limit=limit)

            docs = _build_docs_block(results, max_chars_per_doc=220)
            summary = _generate_summary(query, docs)

            print("Search Results:")
            for result in results:
                print(f"  - {result.get('title', '')}")

            print("\nLLM Summary:")
            print(summary)
        case "citations":
            query = args.query
            limit = args.limit
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                documents = load_movies()
                hybrid = HybridSearch(documents)
                results = hybrid.rrf_search(query, k=60, limit=limit)

            docs = _build_docs_block(results, max_chars_per_doc=220)
            answer = _generate_citations_answer(query, docs)

            print("Search Results:")
            for result in results:
                print(f"  - {result.get('title', '')}")

            print("\nLLM Answer:")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
