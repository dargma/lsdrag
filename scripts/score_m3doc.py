"""Score complete recorded runs with pinned official MMQA metrics and A-RAG judge."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lsdrag.agent import AgentConfig, Reader, parse_answer
from lsdrag.io import dump_json, read_jsonl, sha256, write_jsonl
from lsdrag.metrics import answer_scores, document_read_scores, official_scorer


def judge_prompt(prediction, gold):
    return f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {json.dumps(prediction, ensure_ascii=False)}
Gold answer: {json.dumps(gold, ensure_ascii=False)}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

The gold list contains all required answer items, not alternative acceptable answers.
Respond with ONLY 'correct' or 'incorrect'.
Response:"""


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
    gold = {q["qid"]: q for q in read_jsonl(a.gold)}
    chunks = {c["chunk_id"]: c for c in read_jsonl(a.corpus / "chunks.jsonl")}
    if sha256(a.corpus / "manifest.json") != manifest["corpus_sha256"]:
        raise ValueError("Evaluation corpus differs from run corpus")
    scorer = official_scorer(a.upstream)
    reader = Reader(replace(AgentConfig(), max_output_tokens=32)) if a.judge else None
    rows = []
    for qid in manifest["question_ids"]:
        prediction = json.loads((a.run / (qid + ".json")).read_text())
        if prediction["qid"] != qid or prediction["question"] != gold[qid]["question"]:
            raise ValueError("Run and gold question mismatch")
        # Mechanical JSON recovery is shared across all arms and explicitly logged.
        parsed = parse_answer(prediction["answer_raw"])
        prediction = {**prediction, **parsed}
        row = {"qid": qid, "arm": manifest["arm"], **answer_scores(prediction, gold[qid], scorer), **document_read_scores(prediction, gold[qid], chunks), **{k: prediction[k] for k in ("loops", "seconds", "search_tokens", "evidence_tokens", "unique_delivered_images", "prompt_tokens", "completion_tokens", "answer_format_valid")}, "answer_json_recovered": parsed.get("answer_json_recovered", False), "prediction_sha256": sha256(a.run / (qid + ".json"))}
        if reader:
            targets = [str(item["answer"]) for item in gold[qid]["answers"]]
            cache = a.output / "judge" / (qid + ".json")
            identity = {"prediction": prediction["answers"], "gold": targets, "model": reader.config.model, "template_version": 1}
            if cache.exists():
                judged = json.loads(cache.read_text())
                if judged["identity"] != identity:
                    raise ValueError("Judge cache identity mismatch")
            elif not prediction["answers"]:
                judged = {"identity": identity, "response": "incorrect", "skipped_empty_answer": True}
                dump_json(cache, judged)
            else:
                choice, usage = reader.chat([{"role": "system", "content": "You are an expert evaluator."}, {"role": "user", "content": judge_prompt(prediction["answers"], targets)}], None)
                judged = {"identity": identity, "response": choice["message"].get("content") or "", "usage": usage}
                dump_json(cache, judged)
            response = judged["response"].strip().lower()
            row["llm_accuracy"] = float(response == "correct") if response in {"correct", "incorrect"} else None
        rows.append(row)
    write_jsonl(a.output / "question_metrics.jsonl", rows)
    metrics = ["em", "f1", "contain", "document_recall", "all_documents_read", "loops", "seconds", "search_tokens", "evidence_tokens", "unique_delivered_images", "prompt_tokens", "completion_tokens"]
    if reader:
        metrics.append("llm_accuracy")
    def aggregate(group):
        return {"n": len(group), **{key: {"mean": float(np.mean([r[key] for r in group if r.get(key) is not None])) if any(r.get(key) is not None for r in group) else None, "scored_n": sum(r.get(key) is not None for r in group)} for key in metrics}}
    groups = defaultdict(list)
    for row in rows:
        groups["+".join(row["support_modalities"])].append(row)
    summary = {"arm": manifest["arm"], "overall": aggregate(rows), "by_evidence_combination": {k: aggregate(v) for k, v in groups.items()}, "gold_sha256": sha256(a.gold), "run_manifest_sha256": sha256(a.run / "run_manifest.json"), "page_and_figure_metrics": "pending verified mapping; document labels are not page/figure labels", "metric_scale": "0–1, except count/token/time metrics"}
    dump_json(a.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
