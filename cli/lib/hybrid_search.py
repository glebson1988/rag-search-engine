import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        expanded_limit = max(1, limit * 500)

        bm25_results = self._bm25_search(query, expanded_limit)

        # Use chunked semantic retrieval and aggregate to movie-level scores.
        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)

        doc_scores: dict[int, dict] = {}

        bm25_raw_scores = [score for _, score in bm25_results]
        bm25_norm_scores = normalize_scores(bm25_raw_scores)
        for (doc_id, _), norm_score in zip(bm25_results, bm25_norm_scores):
            document = self.idx.docmap.get(doc_id)
            if document is None:
                continue
            entry = doc_scores.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "title": document["title"],
                    "description": document["description"],
                    "bm25": 0.0,
                    "semantic": 0.0,
                },
            )
            entry["bm25"] = norm_score

        semantic_raw_scores = [result["score"] for result in semantic_results]
        semantic_norm_scores = normalize_scores(semantic_raw_scores)
        for result, norm_score in zip(semantic_results, semantic_norm_scores):
            doc_id = result["id"]
            document = self.idx.docmap.get(doc_id)
            if document is None:
                continue
            entry = doc_scores.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "title": document["title"],
                    "description": document["description"],
                    "bm25": 0.0,
                    "semantic": 0.0,
                },
            )
            entry["semantic"] = norm_score

        results = []
        for entry in doc_scores.values():
            combined = hybrid_score(entry["bm25"], entry["semantic"], alpha=alpha)
            results.append(
                {
                    "id": entry["id"],
                    "title": entry["title"],
                    "description": entry["description"],
                    "hybrid": combined,
                    "bm25": entry["bm25"],
                    "semantic": entry["semantic"],
                }
            )

        results.sort(key=lambda item: item["hybrid"], reverse=True)
        return results[:limit]

    def rrf_search(self, query, k, limit=10):
        expanded_limit = max(1, limit * 500)

        bm25_results = self._bm25_search(query, expanded_limit)
        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)

        ranked_docs: dict[int, dict] = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            document = self.idx.docmap.get(doc_id)
            if document is None:
                continue
            entry = ranked_docs.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "title": document["title"],
                    "description": document["description"],
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf": 0.0,
                },
            )
            if entry["bm25_rank"] is None:
                entry["bm25_rank"] = rank
                entry["rrf"] += rrf_score(rank, k=k)

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            document = self.idx.docmap.get(doc_id)
            if document is None:
                continue
            entry = ranked_docs.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "title": document["title"],
                    "description": document["description"],
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf": 0.0,
                },
            )
            if entry["semantic_rank"] is None:
                entry["semantic_rank"] = rank
                entry["rrf"] += rrf_score(rank, k=k)

        results = sorted(ranked_docs.values(), key=lambda item: item["rrf"], reverse=True)
        return results[:limit]
