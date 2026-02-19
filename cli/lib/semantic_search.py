import os
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies

SCORE_PRECISION = 4


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        cleaned_text = text.strip()
        return self.model.encode(cleaned_text)

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        movie_texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_texts, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        scored_documents: list[tuple[float, dict]] = []
        for i, document_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, document_embedding)
            scored_documents.append((similarity, self.documents[i]))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        top_matches = scored_documents[:limit]
        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in top_matches
        ]


def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("max chunk size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be less than max chunk size")

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    chunks = []
    start = 0
    step = max_chunk_size - overlap
    while start < len(sentences):
        chunk_sentences = sentences[start : start + max_chunk_size]
        # Avoid a trailing 1-sentence overlap-only chunk.
        if start > 0 and len(chunk_sentences) == 1:
            break
        chunks.append(" ".join(chunk_sentences))
        start += step
    return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for movie_idx, document in enumerate(self.documents):
            description = (document.get("description") or "").strip()
            if not description:
                continue

            doc_chunks = semantic_chunk_text(description, max_chunk_size=4, overlap=1)
            total_chunks = len(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=False)
        self.chunk_metadata = chunk_metadata

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def _expected_total_chunks(self) -> int:
        total = 0
        for document in self.documents:
            description = (document.get("description") or "").strip()
            if not description:
                continue
            total += len(semantic_chunk_text(description, max_chunk_size=4, overlap=1))
        return total

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.chunk_metadata = payload.get("chunks", [])
            expected_total_chunks = self._expected_total_chunks()
            if len(self.chunk_embeddings) == expected_total_chunks:
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(chunk_embedding, query_embedding)
            metadata = self.chunk_metadata[i]
            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores: dict[int, dict] = {}
        for item in chunk_scores:
            movie_idx = item["movie_idx"]
            score = item["score"]
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = item

        ranked_movies = sorted(
            movie_scores.items(), key=lambda pair: pair[1]["score"], reverse=True
        )[:limit]

        results = []
        for movie_idx, best_item in ranked_movies:
            movie = self.documents[movie_idx]
            results.append(
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie["description"][:100],
                    "score": round(best_item["score"], SCORE_PRECISION),
                    "metadata": {"chunk_idx": best_item["chunk_idx"]},
                }
            )
        return results


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str) -> list[float]:
    semantic_search = SemanticSearch()
    return semantic_search.generate_embedding(text)


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
