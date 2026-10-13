"""Official Mem-Gallery answer metrics plus actual read/delivered evidence metrics."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lsdrag.io import dump_json, read_jsonl, sha256, write_jsonl
from lsdrag.metrics import memgallery_scorer
from lsdrag.agent import AgentConfig, Reader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--judge", action="store_true")
    a = p.parse_args()
    manifest = json.loads((a.run / "run_manifest.json").read_text())
    if sha256(a.corpus / "manifest.json") != manifest["corpus_sha256"]:
        raise ValueError("Wrong corpus")
    gold = {r["qid"]: r for r in read_jsonl(a.gold)}
    chunks = {c["chunk_id"]: c for c in read_jsonl(a.corpus / "chunks.jsonl")}
    figures = {f["figure_id"]: f for f in read_jsonl(a.corpus / "figures.jsonl")}
    scorer = memgallery_scorer(a.upstream)
    reader = Reader(replace(AgentConfig(), max_output_tokens=512)) if a.judge else None
    template_path = Path(a.upstream) / "benchmark/memengine/evaluate/llm_judge.txt"
    template = template_path.read_text() if reader else None
    rows = []
    for qid in manifest["question_ids"]:
        prediction = json.loads((a.run / (qid + ".json")).read_text())
        q = gold[qid]
        if prediction["status"] == "failed":
            raise ValueError("Resolve infrastructure failures before scoring")
        if prediction["question"] != q["question"] or prediction["scenario"] != q["scenario"]:
            raise ValueError("Prediction identity mismatch")
        predicted_answer = ", ".join(prediction["answers"])
        answer = str(q["answer"])
        clue_chunks = {q["scenario"] + ":" + ref for ref in q["clue"]}
        # VS gold image IDs specify the actual targets; VR often has textual
        # answers, so report delivery of images attached to annotated clue turns.
        clue_figures = {fid for cid in clue_chunks for fid in chunks[cid]["figure_ids"]}
        if q["point"] == "VS":
            target_ids = {s.strip() for s in answer.split(",")}
            target_figures = {fid for fid, f in figures.items() if f["scenario"] == q["scenario"] and f["source_image_id"] in target_ids}
            image_label = "explicit_answer_image_ids"
            mapped = len(target_figures) == len(target_ids)
        else:
            target_figures = clue_figures
            image_label = "images_attached_to_gold_clue_turns; relevance within multi-image turn is not separately annotated"
            mapped = bool(target_figures)
        row = {"qid": qid, "scenario": q["scenario"], "task_type": q["point"], "question_has_image": bool(q.get("question_image")), "question_image_provided": prediction.get("question_image_count", 0) > 0, "em": float(scorer.exact_match_score(predicted_answer, answer)), "f1": float(scorer.f1_score(predicted_answer, answer)), "bleu4": float(scorer.bleu_score(predicted_answer, answer)), "bleu1": float(scorer.bleu_score(predicted_answer, answer, weights=(1, 0, 0, 0))), "bleu2": float(scorer.bleu_score(predicted_answer, answer, weights=(.5, .5, 0, 0))), "clue_read_recall": len(clue_chunks.intersection(prediction["read_chunk_ids"])) / len(clue_chunks), "all_clues_read": float(clue_chunks <= set(prediction["read_chunk_ids"])), "image_target_definition": image_label, "image_targets_mapped": mapped, "image_delivery_recall": len(target_figures.intersection(prediction["delivered_figure_ids"])) / len(target_figures) if mapped and target_figures else None, **{key: prediction[key] for key in ("loops", "seconds", "search_tokens", "evidence_tokens", "unique_delivered_images", "prompt_tokens", "completion_tokens")}}
        if reader:
            prompt = template.replace("{{question}}", q["question"]).replace("{{ground_truth}}", answer).replace("{{model_output}}", predicted_answer)
            identity = {"question": q["question"], "gold": answer, "prediction": predicted_answer, "template_sha256": sha256(template_path), "model": reader.config.model}
            cache = a.output / "judge" / (qid + ".json")
            if cache.exists():
                judged = json.loads(cache.read_text())
                if judged["identity"] != identity:
                    raise ValueError("Judge cache differs from current inputs")
            else:
                choice, usage = reader.chat([{"role": "user", "content": prompt}], None)
                response = choice["message"].get("content") or ""
                judged = {"identity": identity, "response": response, "usage": usage, "official_parsed": scorer.parse_judge_response(response)}
                dump_json(cache, judged)
            parsed = judged["official_parsed"]
            row["llm_judge_official"] = parsed.get("score") if parsed else None
            # Preserve the five-level rubric score separately. The official
            # parser currently collapses .25 to .5 and .75 to 1.
            raw_score = None
            response = judged["response"]
            try:
                obj, _ = json.JSONDecoder().raw_decode(response[response.index("{"):])
                raw_score = float(obj["score"])
            except (ValueError, KeyError, TypeError):
                pass
            row["llm_judge_rubric_raw"] = raw_score if raw_score in {0., .25, .5, .75, 1.} else None
        rows.append(row)
    write_jsonl(a.output / "question_metrics.jsonl", rows)
    metrics = ["em", "f1", "bleu1", "bleu2", "bleu4", "clue_read_recall", "all_clues_read", "image_delivery_recall", "loops", "seconds", "search_tokens", "evidence_tokens", "prompt_tokens", "completion_tokens", "unique_delivered_images"]
    if reader:
        metrics.extend(["llm_judge_official", "llm_judge_rubric_raw"])
    def aggregate(group):
        return {"n": len(group), **{key: {"mean": float(np.mean([r[key] for r in group if r[key] is not None])) if any(r[key] is not None for r in group) else None, "scored_n": sum(r[key] is not None for r in group)} for key in metrics}}
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_type"] + ("+question_image" if row["question_has_image"] else "+text_question")].append(row)
    summary = {"arm": manifest["arm"], "overall": aggregate(rows), "by_task_and_question_image": {k: aggregate(v) for k, v in groups.items()}, "gold_sha256": sha256(a.gold), "run_manifest_sha256": sha256(a.run / "run_manifest.json"), "official_judge": "official parser collapses five-level rubric to 0/.5/1; raw rubric scores also retained" if reader else "not run", "metric_scale": "0–1 except counts/time/tokens"}
    summary["question_image_policy"] = manifest.get("question_image_policy", "original")
    summary["subgroup_note"] = "question_has_image describes the original dataset annotation, not whether it was supplied; see question_image_provided"
    summary["token_accounting"] = "search/evidence tokens use common tiktoken gpt-4o encoding; prompt/completion tokens are actual Reader API usage"
    dump_json(a.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
