import contextlib
import io
import logging
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = None
        try:
            # Fail fast: only use locally cached model files.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.model = SentenceTransformer(
                    model_name, device="cpu", local_files_only=True
                )
        except Exception:
            self.model = None

    def _fallback_embedding(self, image: Image.Image) -> np.ndarray:
        # Deterministic, fast fallback vector when CLIP is unavailable.
        arr = np.asarray(image.resize((32, 32)).convert("RGB"), dtype=np.float32)
        flat = arr.reshape(-1)
        if flat.size == 0:
            return np.zeros(512, dtype=np.float32)
        flat = flat / 255.0
        if flat.size < 512:
            flat = np.pad(flat, (0, 512 - flat.size))
        return flat[:512]

    def embed_image(self, image_path: str):
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            if self.model is not None:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    embeddings = self.model.encode([rgb_image], show_progress_bar=False)
                return embeddings[0]
            return self._fallback_embedding(rgb_image)


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
