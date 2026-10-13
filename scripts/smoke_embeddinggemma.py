"""Exercise the actual text and image embedding paths; never a benchmark score."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    model = SentenceTransformer(args.model, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
    images = []
    for color, shape in [("red", "circle"), ("blue", "square")]:
        im = Image.new("RGB", (320, 320), "white")
        draw = ImageDraw.Draw(im)
        getattr(draw, "ellipse" if shape == "circle" else "rectangle")((50, 50, 270, 270), fill=color)
        images.append(im)
    text = model.encode(["a red circle on a white background", "a blue square on a white background"], prompt_name="SearchQuery", normalize_embeddings=True)
    vision = model.encode(images, normalize_embeddings=True)
    docs = model.encode(["A red circle.", "A blue square."], prompt_name="Document", normalize_embeddings=True)
    raw_norms = []
    normalized = []
    for vectors in [text, vision, docs]:
        assert vectors.shape == (2, 768), vectors.shape
        assert np.isfinite(vectors).all()
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        raw_norms.append(norms[:, 0].tolist())
        assert (norms > 0).all()
        vectors = vectors / norms
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        normalized.append(vectors)
    text, vision, docs = normalized
    report = {"purpose": "runtime smoke only; not M3DocVQA", "dtype": "bfloat16; final normalization float32", "raw_norms": raw_norms, "shape": list(text.shape), "text_image_similarity": (text @ vision.T).tolist(), "text_document_similarity": (text @ docs.T).tolist(), "elapsed_seconds": time.monotonic() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
