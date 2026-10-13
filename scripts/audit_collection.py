"""Audit PDF collection by URL aliases without confusing absent text with lost visual evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

from lsdrag.io import dump_json, read_jsonl, sha256

p = argparse.ArgumentParser()
p.add_argument("--corpus-manifest", type=Path, required=True)
p.add_argument("--gold", type=Path, required=True)
p.add_argument("--pdf-root", type=Path, action="append", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
manifest = read_jsonl(a.corpus_manifest)
urls = {r["url"] for r in manifest}
id_to_url = {r["document_id"]: r["url"] for r in manifest}
audits = {}
for root in a.pdf_root:
    for path in sorted(root.glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("url") not in urls:
            continue
        pdf = path.with_suffix(".pdf")
        row["verified_local_pdf"] = bool(pdf.exists() and row.get("sha256") == sha256(pdf))
        previous = audits.get(row["url"])
        if previous is None or (row["verified_local_pdf"], row.get("status") == "ok", row.get("rendered_at", "")) > (previous["verified_local_pdf"], previous.get("status") == "ok", previous.get("rendered_at", "")):
            audits[row["url"]] = row
questions = []
for q in read_jsonl(a.gold):
    required = {id_to_url[c["doc_id"]] for c in q["supporting_context"]}
    missing = sorted(u for u in required if not audits.get(u, {}).get("verified_local_pdf"))
    incomplete = sorted(u for u in required if audits.get(u, {}).get("status") == "image_load_incomplete")
    questions.append({"qid": q["qid"], "required_urls": len(required), "missing_pdf_urls": missing, "incomplete_image_urls": incomplete})
problems = [{k: row.get(k) for k in ("document_id", "url", "title", "status", "error", "article_image_load_failures", "verified_local_pdf")} for row in audits.values() if row["status"] != "ok" or not row["verified_local_pdf"]]
report = {"requested_unique_urls": len(urls), "collected_verified_urls": sum(r["verified_local_pdf"] for r in audits.values()),
          "pdf_pages": sum(r.get("pages", 0) for r in audits.values() if r["verified_local_pdf"]),
          "status_counts": dict(Counter(r["status"] for r in audits.values())), "no_audit_urls": sorted(urls - set(audits)),
          "questions": len(questions), "questions_missing_pdf": sum(bool(r["missing_pdf_urls"]) for r in questions),
          "questions_with_incomplete_image_document": sum(bool(r["incomplete_image_urls"]) for r in questions),
          "problems": problems, "question_audit": questions, "gold_sha256": sha256(a.gold), "corpus_manifest_sha256": sha256(a.corpus_manifest),
          "interpretation": "Image-load failures do not establish loss of a required gold figure. Absence of answer text is not a valid visual-grounding test. Use reviewed original-image mappings separately."}
dump_json(a.output, report)
print(json.dumps({k: v for k, v in report.items() if k not in {"question_audit", "problems"}}))
