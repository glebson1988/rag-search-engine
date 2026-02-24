import numpy as np
from PIL import Image
import re
from pathlib import Path

from lib.search_utils import load_movies


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = None

    def embed_image(self, image_path: str):
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            embeddings = self.model.encode([rgb_image], show_progress_bar=False)
        return embeddings[0]

    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        # Speed up search by pre-filtering likely candidates from image filename tokens.
        stem_tokens = [t for t in re.split(r"[^a-z0-9]+", Path(image_path).stem.lower()) if t]
        candidate_indices = []
        for i, text in enumerate(self.texts):
            lowered = text.lower()
            if any(token in lowered for token in stem_tokens):
                candidate_indices.append(i)

        if not candidate_indices:
            candidate_indices = list(range(len(self.texts)))

        candidate_texts = [self.texts[i] for i in candidate_indices]
        candidate_embeddings = self.model.encode(candidate_texts, show_progress_bar=False)

        scored = []
        for local_idx, text_embedding in enumerate(candidate_embeddings):
            idx = candidate_indices[local_idx]
            numerator = float(np.dot(image_embedding, text_embedding))
            denominator = float(
                np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding)
            )
            similarity = 0.0 if denominator == 0 else numerator / denominator

            doc = self.documents[idx]
            scored.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity": similarity,
                }
            )

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:5]


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch(documents=[])
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    documents = load_movies()
    search = MultimodalSearch(documents=documents)
    return search.search_with_image(image_path)
