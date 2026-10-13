"""Mem-Gallery schema adapters with scenario isolation and private evidence labels."""
from collections import defaultdict
from pathlib import Path
import random

from PIL import Image

from .io import dump_json, read_jsonl, sha256, write_jsonl


def select_questions(rows, count, seed):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["qid"]):
        groups[(row["scenario"], bool(row.get("question_image")))].append(row)
    if count > len(rows) or not rows:
        raise ValueError("Insufficient eligible questions")
    quotas = {key: count * len(value) / len(rows) for key, value in groups.items()}
    sizes = {key: int(value) for key, value in quotas.items()}
    for key in sorted(groups, key=lambda k: (-(quotas[k] - sizes[k]), k))[:count - sum(sizes.values())]:
        sizes[key] += 1
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        selected.extend(groups[key][:sizes[key]])
    return sorted(selected, key=lambda r: r["qid"])


def image_reference(reference):
    """Resolve only the actual path references supplied by the dataset."""
    if not reference.startswith("../image/"):
        raise ValueError(f"Unexpected image reference: {reference}")
    relative = Path("data/image") / reference.removeprefix("../image/")
    if ".." in relative.parts or relative.is_absolute():
        raise ValueError("Out-of-root image reference")
    return relative


def convert_scenario(scenario, data, root):
    chunks, figures, documents = [], [], []
    name = data["character_profile"].get("name", "user")
    for session_index, session in enumerate(data["multi_session_dialogues"]):
        document_id = scenario + ":" + session["session_id"]
        metadata = {"document_id": document_id, "document_ids": [document_id], "title": f"{name} — {session['date']} — {session['session_id']}", "scenario": scenario, "session_id": session["session_id"], "date": session["date"], "page_index": session_index}
        documents.append(metadata)
        for turn in session["dialogues"]:
            chunk_id = scenario + ":" + turn["round"]
            ids = turn.get("image_id", [])
            references = turn.get("input_image", [])
            captions = turn.get("image_caption", [])
            if len(ids) != len(references) or len(captions) != len(references):
                raise ValueError(f"Mismatched image/caption lists: {chunk_id}")
            parts = [metadata["title"], "Dialogue ID: " + turn["round"], f"user ({name}): {turn.get('user', '')}", "assistant: " + turn.get("assistant", "")]
            figure_ids = []
            for image_id, reference, caption in zip(ids, references, captions):
                path = Path(root) / image_reference(reference)
                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()
                figure_id = scenario + ":" + image_id
                figure_ids.append(figure_id)
                parts.append(f"Image ID: {image_id}\nProvided image caption: {caption}")
                figures.append({**metadata, "figure_id": figure_id, "chunk_id": chunk_id, "source_image_id": image_id, "source_round": turn["round"], "image_path": str(path.resolve()), "image_ref": reference, "image_sha256": sha256(path), "width": width, "height": height, "caption": caption, "has_caption": bool(caption), "has_description": bool(caption)})
            chunks.append({**metadata, "id": chunk_id, "chunk_id": chunk_id, "source_round": turn["round"], "kind": "image_caption" if figure_ids else "text", "text": "\n".join(parts), "figure_ids": figure_ids})
    # Adjacent turns may cross sessions, preserving the actual conversation order.
    for i, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[i - 1]["chunk_id"] if i else None
        chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i + 1 < len(chunks) else None
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Duplicate dialogue IDs within scenario")
    if len({f["figure_id"] for f in figures}) != len(figures):
        raise ValueError("Duplicate image IDs within scenario")
    return chunks, figures, documents


def save_corpus(output, chunks, figures, documents, source_revision):
    output = Path(output)
    write_jsonl(output / "chunks.jsonl", chunks)
    write_jsonl(output / "figures.jsonl", figures)
    write_jsonl(output / "documents.jsonl", documents)
    manifest = {"benchmark": "Mem-Gallery", "revision": source_revision, "chunks": len(chunks), "figures": len(figures), "documents": len(documents), "scenarios": len({c["scenario"] for c in chunks}), "chunks_sha256": sha256(output / "chunks.jsonl"), "figures_sha256": sha256(output / "figures.jsonl"), "temporal_policy": "all sessions in the question's scenario, matching official store-all-before-QA execution; QA session_id is evidence metadata, not a query-time cutoff", "page_index_semantics": "session ordinal, not a PDF page", "captions": "dataset-provided, no new generated captions"}
    dump_json(output / "manifest.json", manifest)
    return manifest
