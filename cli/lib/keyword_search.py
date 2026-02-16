import os
import pickle
import math
from collections import Counter, defaultdict

from .constants import BM25_K1
from .search_utils import BM25_B, CACHE_DIR, load_movies
from .search_utils import tokenize_text


class InvertedIndex:
    def __init__(self) -> None:
        # token -> set(doc_id)
        self.index: dict[str, set[int]] = defaultdict(set)
        # doc_id -> full document object
        self.docmap: dict[int, dict] = {}
        # doc_id -> Counter(token -> frequency)
        self.term_frequencies: dict[int, Counter[str]] = {}
        # doc_id -> tokenized document length
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize_text(term)
        if not tokens:
            return []
        return sorted(self.index.get(tokens[0], set()))

    def build(self) -> None:
        for movie in load_movies():
            movie_id = movie["id"]
            self.docmap[movie_id] = movie
            self.__add_document(
                movie_id,
                f"{movie['title']} {movie['description']}",
            )

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        doc_lengths_path = self.doc_lengths_path
        if (
            not os.path.exists(index_path)
            or not os.path.exists(docmap_path)
            or not os.path.exists(term_frequencies_path)
            or not os.path.exists(doc_lengths_path)
        ):
            raise FileNotFoundError(
                f"Index files not found: {index_path}, {docmap_path}, {term_frequencies_path}, and/or {doc_lengths_path}"
            )

        with open(index_path, "rb") as f:
            loaded_index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            loaded_docmap = pickle.load(f)
        with open(term_frequencies_path, "rb") as f:
            loaded_term_frequencies = pickle.load(f)
        with open(doc_lengths_path, "rb") as f:
            loaded_doc_lengths = pickle.load(f)

        self.index = defaultdict(set, loaded_index)
        self.docmap = loaded_docmap
        self.term_frequencies = loaded_term_frequencies
        self.doc_lengths = loaded_doc_lengths

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("term must tokenize to a single token")
        if not tokens:
            return 0
        return self.term_frequencies.get(doc_id, Counter()).get(tokens[0], 0)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("term must tokenize to a single token")
        if not tokens:
            return 0.0

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(tokens[0], set()))
        return math.log(
            ((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5))
            + 1
        )

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length == 0.0:
            return 0.0

        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)


def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1, b)
