import os
import pickle
from collections import defaultdict

from .search_utils import CACHE_DIR, load_movies
from .search_utils import tokenize_text


class InvertedIndex:
    def __init__(self) -> None:
        # token -> set(doc_id)
        self.index: dict[str, set[int]] = defaultdict(set)
        # doc_id -> full document object
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        for token in tokenize_text(text):
            self.index[token].add(doc_id)

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

    def load(self) -> None:
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError(
                f"Index files not found: {index_path} and/or {docmap_path}"
            )

        with open(index_path, "rb") as f:
            loaded_index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            loaded_docmap = pickle.load(f)

        self.index = defaultdict(set, loaded_index)
        self.docmap = loaded_docmap
