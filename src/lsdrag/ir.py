"""Convert verified MinerU 4 structured output to shared retrieval units.

Original document aliases are retained: two MMQA IDs pointing to the same
Wikipedia URL are one physical document, never two independent evidence hits.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from PIL import Image

from .io import dump_json, read_jsonl, sha256, write_jsonl


def split_text(text, max_words=300):
    """Preserve sentence boundaries where possible, without cross-page joins."""
    sentences = re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip())
    part = []
    size = 0
    for sentence in sentences:
        words = sentence.split()
        if size + len(words) > max_words and part:
            yield " ".join(part)
            part, size = [], 0
        while len(words) > max_words:
            yield " ".join(words[:max_words])
            words = words[max_words:]
        if words:
            part.append(" ".join(words))
            size += len(words)
    if part:
        yield " ".join(part)


def split_table(text, max_words=300):
    """Repeat Markdown table headers and keep each row intact.

MinerU structured table bodies are Markdown in the tested 4.x schema. For
unrecognized formats retain the complete body rather than lose cell relations.
An oversized row remains oversized and is explicitly flagged in the IR.
"""
    lines = text.strip().splitlines()
    if len(lines) < 3 or not re.match(r"^[| :\-]+$", lines[1]):
        return [text.strip()] if text.strip() else []
    header = lines[:2]
    parts, rows = [], []
    for row in lines[2:]:
        if rows and len("\n".join(header + rows + [row]).split()) > max_words:
            parts.append("\n".join(header + rows))
            rows = []
        rows.append(row)
    if rows:
        parts.append("\n".join(header + rows))
    return parts


def convert_document(source, metadata, asset_root, max_words=300):
    root = Path(asset_root).resolve()
    document_id = metadata["document_id"]
    chunks, figures = [], []
    for page in source["pages"]:
        page_idx = page["page_idx"]
        if not isinstance(page_idx, int) or page_idx < 0:
            raise ValueError("Invalid zero-based page index")
        blocks = page["blocks"]
        for index, block in enumerate(blocks):
            kind = block["type"]
            if kind in {"header", "footer", "page_number"}:
                continue
            body = block.get("content", "")
            if not isinstance(body, str):
                raise ValueError(f"Unexpected MinerU structured content type: {type(body)}")
            captions = "\n".join(c.get("content", "") for c in block.get("captions", []))
            visual_notes = "\n".join(c.get("content", "") for c in block.get("footnotes", []))
            figure_id = None
            if kind in {"image", "chart"}:
                reference = block.get("image_source")
                if not reference:
                    raise ValueError(f"Visual block has no saved image reference: {document_id}:{page_idx}:{index}")
                image_path = (root / reference).resolve()
                if not image_path.is_relative_to(root) or not image_path.is_file():
                    raise ValueError(f"Missing or out-of-root image reference: {reference}")
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                figure_id = f"{document_id}:p{page_idx}:f{index}"
                context = "\n".join(b.get("content", "") for b in blocks[max(0, index - 2):index + 3] if b.get("type") in {"text", "paragraph_title", "doc_title"})
                # No generated visual description or ground-truth image text.
                description = "\n".join(x for x in (captions, visual_notes) if x.strip())
                body = description or "[No caption. Nearby document text follows.]\n" + " ".join(context.split()[:120])
                kind = "image_caption"
                figures.append({**metadata, "figure_id": figure_id, "page_index": page_idx, "bbox": block.get("bbox"), "image_path": str(image_path), "image_ref": reference, "image_sha256": sha256(image_path), "width": width, "height": height, "caption": captions, "has_caption": bool(captions.strip()), "source_block_type": block["type"]})
                figures[-1].update(footnotes=visual_notes, has_description=bool(description.strip()))
            elif kind == "table":
                pass  # Add captions after row splitting to preserve header recognition.
            elif kind not in {"text", "paragraph_title", "doc_title", "ref_text", "equation", "list", "code", "page_footnote"}:
                # Retain textual uncommon block kinds, without inventing fields.
                kind = "text"
            if not body.strip():
                continue
            parts = [body] if figure_id else split_table(body, max_words) if kind == "table" else list(split_text(body, max_words))
            for part_index, text in enumerate(parts):
                if kind == "table" and captions:
                    text = captions + "\n" + text
                chunk_id = f"{document_id}:p{page_idx}:b{index}:s{part_index}"
                chunk = {**metadata, "chunk_id": chunk_id, "id": chunk_id, "page_index": page_idx, "kind": kind if kind in {"table", "image_caption"} else "text", "text": metadata["title"] + "\n" + text, "bbox": block.get("bbox"), "source_block_index": index, "figure_ids": [figure_id] if figure_id else [], "oversized_unit": len(text.split()) > max_words}
                chunks.append(chunk)
                if figure_id:
                    figures[-1]["chunk_id"] = chunk_id
    for i, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[i - 1]["chunk_id"] if i else None
        chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i + 1 < len(chunks) else None
    return chunks, figures


def build_ir(parsed_root, corpus_manifest, output_dir, max_words=300):
    parsed_root, output_dir = Path(parsed_root), Path(output_dir)
    rows = read_jsonl(corpus_manifest)
    by_url = defaultdict(list)
    for row in rows:
        by_url[row["url"]].append(row)
    chunks, figures, docs, failures = [], [], [], []
    for url, aliases in sorted(by_url.items()):
        aliases.sort(key=lambda r: r["document_id"])
        available = [r for r in aliases if (parsed_root / r["document_id"] / "_SUCCESS.json").exists()]
        if not available:
            continue
        row = available[0]
        directory = parsed_root / row["document_id"]
        path = directory / "structured_content.json"
        metadata = {"document_id": aliases[0]["document_id"], "document_ids": [a["document_id"] for a in aliases], "title": row["title"], "url": url}
        try:
            source = json.loads(path.read_text())
            doc_chunks, doc_figures = convert_document(source, metadata, directory, max_words)
            if not doc_chunks:
                raise ValueError("No retrievable content")
            chunks.extend(doc_chunks)
            figures.extend(doc_figures)
            docs.append({**metadata, "parsed_source_id": row["document_id"], "pages": len(source["pages"]), "parsed_sha256": sha256(path), "parse_identity": json.loads((directory / "_SUCCESS.json").read_text())})
        except Exception as error:
            failures.append({"document_id": row["document_id"], "error": f"{type(error).__name__}: {error}"})
    if failures:
        dump_json(output_dir / "ir_failures.json", failures)
        raise ValueError(f"{len(failures)} documents failed IR validation; refusing incomplete index")
    if not chunks:
        raise ValueError("No completed parsed documents")
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "figures.jsonl", figures)
    write_jsonl(output_dir / "documents.jsonl", docs)
    # Original A-RAG input format, for explicit original/text-only mode.
    dump_json(output_dir / "arag_chunks.json", [{"id": c["chunk_id"], "text": c["text"]} for c in chunks])
    summary = {"documents": len(docs), "document_aliases": sum(len(d["document_ids"]) for d in docs), "pages": sum(d["pages"] for d in docs), "chunks": len(chunks), "chunk_types": dict(Counter(c["kind"] for c in chunks)), "figures": len(figures), "figures_without_caption": sum(not f["has_caption"] for f in figures), "max_words": max_words, "page_index_base": 0, "table_policy": "keep rows and repeat Markdown headers; flag oversized rows", "page_boundary_policy": "no cross-page text merge; explicit adjacent chunk IDs", "chunks_sha256": sha256(output_dir / "chunks.jsonl"), "figures_sha256": sha256(output_dir / "figures.jsonl"), "corpus_manifest_sha256": sha256(corpus_manifest)}
    summary["figures_without_any_description"] = sum(not f["has_description"] for f in figures)
    summary["figure_description_policy"] = "retain original captions and footnotes; nearby text fallback only when both are absent"
    dump_json(output_dir / "manifest.json", summary)
    return summary
