"""Visual evidence metrics over reviewed original-image to figure mappings."""


def visual_evidence_scores(prediction, gold, chunks, figures, reviews):
    required = sorted({c["doc_id"] for c in gold["supporting_context"] if c["doc_part"] == "image"})
    mapped = {key: set(reviews[key]["accepted_figure_ids"]) for key in required if key in reviews and reviews[key]["status"] == "verified_same_visual_content" and reviews[key]["accepted_figure_ids"]}
    for matches in mapped.values():
        if not matches <= set(figures):
            raise ValueError("Visual mapping references a figure outside this corpus")
    complete = bool(required) and len(mapped) == len(required)
    read_pages = {(chunks[key]["document_id"], chunks[key]["page_index"]) for key in prediction["read_chunk_ids"]}
    opened = set(prediction["opened_figure_ids"])
    delivered = set(prediction["delivered_figure_ids"])
    candidates = set()
    for event in prediction["trajectory"]:
        if not event["tool_name"].endswith("search") or not isinstance(event["result"], list):
            continue
        for result in event["result"]:
            if result.get("figure_id"):
                candidates.add(result["figure_id"])
            if result.get("chunk_id") in chunks:
                candidates.update(chunks[result["chunk_id"]]["figure_ids"])
    def recall(identities):
        return sum(bool(matches & identities) for matches in mapped.values()) / len(mapped) if mapped else None
    page_recall = sum(any((figures[f]["document_id"], figures[f]["page_index"]) in read_pages for f in matches) for matches in mapped.values()) / len(mapped) if mapped else None
    return {"gold_image_count": len(required), "mapped_gold_image_count": len(mapped), "visual_mapping_complete": complete, "figure_candidate_recall": recall(candidates) if complete else None, "figure_open_recall": recall(opened) if complete else None, "figure_delivery_recall": recall(delivered) if complete else None, "all_gold_images_delivered": float(all(bool(matches & delivered) for matches in mapped.values())) if complete else None, "image_evidence_page_recall": page_recall if complete else None, "available_mapped_figure_delivery_recall": recall(delivered), "unmapped_gold_image_ids": sorted(set(required) - set(mapped))}
