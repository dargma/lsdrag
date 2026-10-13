"""Track evidence reads separately from images accepted by the Reader."""


class EvidenceContext:
    def __init__(self, retrieval, max_images=8):
        self.retrieval = retrieval
        self.max_images = max_images
        self.read_chunks = set()
        self.opened_figures = set()
        self.delivered_figures = set()
        self.delivered_hashes = set()
        self.pending_images = {}
        self.events = []

    def _figure(self, identity):
        figure = self.retrieval.figures[identity]
        self.opened_figures.add(identity)
        digest = figure["image_sha256"]
        if digest in self.delivered_hashes:
            self.delivered_figures.add(identity)
            return "already_delivered_identical_image"
        if digest in self.pending_images:
            if identity not in self.pending_images[digest]["figure_ids"]:
                self.pending_images[digest]["figure_ids"].append(identity)
            return "queued_identical_image"
        if len(self.delivered_hashes) + len(self.pending_images) >= self.max_images:
            return "image_budget_exhausted"
        self.pending_images[digest] = {"image_path": figure["image_path"], "image_sha256": digest, "figure_ids": [identity]}
        return "queued_for_reader"

    def read(self, chunk_ids):
        if not isinstance(chunk_ids, list) or not 1 <= len(chunk_ids) <= 8:
            raise ValueError("Provide 1–8 chunk IDs")
        # Validate whole request before mutating state.
        for key in chunk_ids:
            if key not in self.retrieval.chunks:
                raise ValueError(f"Unknown chunk ID: {key}")
        results = []
        for key in chunk_ids:
            c = self.retrieval.chunks[key]
            record = {"chunk_id": key, "document_id": c["document_id"], "page_index": c["page_index"], "kind": c["kind"]}
            if key in self.read_chunks:
                record["status"] = "already_read"
            else:
                self.read_chunks.add(key)
                record.update(text=c["text"], previous_chunk_id=c.get("previous_chunk_id"), next_chunk_id=c.get("next_chunk_id"))
            record["figures"] = [{"figure_id": f, "status": self._figure(f)} for f in c["figure_ids"]]
            results.append(record)
        self.events.append({"action": "read_chunk", "chunk_ids": list(chunk_ids)})
        return results

    def figure_read(self, figure_ids):
        if self.retrieval.arm != "V2":
            raise ValueError("figure_read is only exposed in V2")
        if not isinstance(figure_ids, list) or not 1 <= len(figure_ids) <= 8:
            raise ValueError("Provide 1–8 figure IDs")
        for key in figure_ids:
            if key not in self.retrieval.figures:
                raise ValueError(f"Unknown figure ID: {key}")
        return self.read([self.retrieval.figures[f]["chunk_id"] for f in figure_ids])

    def mark_images_accepted(self, images):
        """Call only after a successful Reader response to an image-bearing request."""
        for item in images:
            digest = item["image_sha256"]
            self.delivered_hashes.add(digest)
            self.delivered_figures.update(item["figure_ids"])
            self.pending_images.pop(digest, None)
        self.events.append({"action": "reader_accepted_images", "figure_ids": [f for item in images for f in item["figure_ids"]]})

    def summary(self):
        return {"read_chunk_ids": sorted(self.read_chunks), "opened_figure_ids": sorted(self.opened_figures), "delivered_figure_ids": sorted(self.delivered_figures), "unique_delivered_images": len(self.delivered_hashes), "pending_figure_ids": sorted({f for item in self.pending_images.values() for f in item["figure_ids"]})}
