from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        return self.model.encode(text).tolist()


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str) -> list[float]:
    semantic_search = SemanticSearch()
    return semantic_search.generate_embedding(text)
