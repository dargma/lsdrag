"""Pinned full-dimensional embedding adapters, with explicit task formatting."""
from pathlib import Path
import numpy as np

MODELS = {
    "qwen": ("Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b", 2560),
    "gemma": ("google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b", 768),
}
QWEN_INSTRUCTION = "Given a question, retrieve passages that contain evidence needed to answer it."
GEMMA_QUERY_PREFIX = "task: search result | query: "


def normalized_float32(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite embedding")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero embedding")
    return values / norms


class Embedder:
    def __init__(self, channel, cache_dir, device="cuda", batch_size=16):
        import torch
        from sentence_transformers import SentenceTransformer
        self.channel = channel
        self.batch_size = batch_size
        repo, revision, self.dimension = MODELS[channel]
        path = Path(cache_dir) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        if not path.is_dir():
            raise FileNotFoundError(f"Pinned model snapshot missing: {path}")
        kwargs = {"model_kwargs": {"torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32}}
        if channel == "gemma":
            kwargs["config_kwargs"] = {"audio_config": None}
        self.model = SentenceTransformer(str(path), device=device, **kwargs)
        self.model.max_seq_length = min(self.model.max_seq_length, 8192)

    def encode(self, inputs, role, titles=None):
        if role not in {"query", "document", "image"}:
            raise ValueError(role)
        if role == "image" and self.channel != "gemma":
            raise ValueError("Qwen text embedder cannot encode images")
        if role == "query":
            inputs = [f"Instruct: {QWEN_INSTRUCTION}\nQuery: {s}" if self.channel == "qwen" else GEMMA_QUERY_PREFIX + s for s in inputs]
        elif role == "document" and self.channel == "gemma":
            titles = titles or ["none"] * len(inputs)
            if len(titles) != len(inputs):
                raise ValueError("Title count mismatch")
            inputs = [f"title: {title or 'none'} | text: {text}" for title, text in zip(titles, inputs)]
        # Formatting is already explicit. Prevent any configured default prompt.
        vectors = self.model.encode(inputs, batch_size=self.batch_size, prompt="", normalize_embeddings=False, show_progress_bar=False)
        vectors = normalized_float32(vectors)
        if vectors.shape != (len(inputs), self.dimension):
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
        return vectors
