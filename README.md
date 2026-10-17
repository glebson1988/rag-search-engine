# RAG Search Engine

Movie search project with keyword, semantic, hybrid, RAG, and multimodal retrieval.

## Tech Stack

- Python 3.14
- `argparse` — CLI commands
- `numpy` — vector math and cosine similarity
- `nltk` — token preprocessing (stopwords, stemming)
- `pickle` — local index/cache persistence
- `python-dotenv` — environment variables from `.env`
- `openai` SDK — OpenAI-compatible client for Groq API
- `Pillow (PIL)` — image loading for multimodal search
- `sentence-transformers` — embedding and reranking models

## Retrieval and Ranking

- Inverted index (`index`, `docmap`) for fast keyword lookup
- TF / IDF / TF-IDF utilities
- BM25 ranking with TF saturation and document-length normalization
- Semantic search with dense embeddings + cosine similarity
- Chunked semantic retrieval (sentence chunking + overlap)
- Hybrid search:
  - Weighted combination (normalized BM25 + semantic)
  - Reciprocal Rank Fusion (RRF)
- Re-ranking options:
  - LLM individual scoring
  - LLM batch ranking
  - Cross-encoder reranking

## Models in Use

- `all-MiniLM-L6-v2`
  - Purpose: semantic text embeddings for documents and queries
  - Used in: `cli/lib/semantic_search.py`

- `cross-encoder/ms-marco-TinyBERT-L2-v2`
  - Purpose: rerank retrieved candidates by query-document relevance
  - Used in: `cli/hybrid_search_cli.py`

- `clip-ViT-B-32`
  - Purpose: embed images and text in a shared space for image-based movie search
  - Used in: `cli/lib/multimodal_search.py`

- `openai/gpt-oss-20b` (via Groq; text-model default)
  - Purpose: query enhancement, LLM reranking, LLM evaluation, RAG generation
  - Used in: `cli/hybrid_search_cli.py`, `cli/augmented_generation_cli.py`
  - Override using `GROQ_MODEL` in the environment or `.env`.
  - The original project used `llama-3.1-8b-instant`. Groq retired that model for free/developer usage on August 16, 2026; the replacement follows [Groq's migration guidance](https://console.groq.com/docs/deprecations).

- `meta-llama/llama-4-scout-17b-16e-instruct` (legacy remote vision workflow)
  - Purpose: multimodal query rewriting from image + text
  - Used in: `cli/describe_image_cli.py`
  - Deprecated for Groq free/developer usage; this command has not been migrated or verified against a replacement. Local CLIP image search is a separate workflow.

## CLI Entrypoints

- `cli/keyword_search_cli.py` — keyword search, TF/IDF/TF-IDF, BM25
- `cli/semantic_search_cli.py` — semantic/chunked search
- `cli/hybrid_search_cli.py` — hybrid search, enhancement, reranking, eval
- `cli/evaluation_cli.py` — Precision@K / Recall@K / F1
- `cli/augmented_generation_cli.py` — `rag`, `summarize`, `citations`, `question`
- `cli/describe_image_cli.py` — image-based query rewrite
- `cli/multimodal_search_cli.py` — image embedding and image search

## Data and Cache

- Data:
  - `data/movies.json`
  - `data/stopwords.txt`
  - `data/golden_dataset.json`
  - `data/paddington.jpeg`
- Cache:
  - `cache/index.pkl`, `cache/docmap.pkl`, `cache/term_frequencies.pkl`, `cache/doc_lengths.pkl`
  - `cache/movie_embeddings.npy`, `cache/chunk_embeddings.npy`, `cache/chunk_metadata.json`

## Quick Start

```bash
uv sync
uv run cli/keyword_search_cli.py build
```

Optional `.env` for LLM features:

```env
GROQ_API_KEY="your_key"
GROQ_MODEL=openai/gpt-oss-20b
```

## Groq model configuration

```dotenv
GROQ_MODEL=openai/gpt-oss-20b
```

An October 17, 2026 smoke test of `openai/gpt-oss-20b` returned HTTP 200 in 1.478 seconds and identified Paddington. The original text model had returned HTTP 404 (`model_not_found`). The successful response also included platform claims unsupported by the supplied descriptions, so API availability and grounded answer quality should be assessed separately.

The separate image-query rewriting command still names the original Scout model, which Groq also retired for free/developer usage. That remote vision workflow has not been migrated or tested. Local CLIP image search is separate and passes the local smoke check.
