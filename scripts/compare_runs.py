"""Paired statistics and auditable case lists; never select examples by answer length."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from lsdrag.io import dump_json, read_jsonl, sha256
from lsdrag.metrics import paired_comparison, cluster_paired_bootstrap

p = argparse.ArgumentParser()
p.add_argument("--baseline", type=Path, required=True, help="Scored run directory")
p.add_argument("--variant", type=Path, action="append", required=True)
p.add_argument("--benchmark", choices=["m3docvqa", "memgallery"], required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
metric_names = (["em", "f1", "contain", "llm_accuracy", "document_recall", "all_documents_read",
                 "figure_candidate_recall", "figure_delivery_recall", "image_evidence_page_recall"]
                if a.benchmark == "m3docvqa" else
                ["em", "f1", "llm_judge_official", "llm_judge_rubric_raw", "clue_read_recall", "image_delivery_recall"])

def load(directory):
    rows = read_jsonl(directory / "question_metrics.jsonl")
    if len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate scored question")
    return {r["qid"]: r for r in rows}

baseline = load(a.baseline)
base_summary = json.loads((a.baseline / "summary.json").read_text())
comparisons = {}
for directory in a.variant:
    variant = load(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if set(baseline) != set(variant) or summary["gold_sha256"] != base_summary["gold_sha256"]:
        raise ValueError("Experiments must have identical questions and gold annotations")
    group_ids = defaultdict(list)
    for qid, row in baseline.items():
        group_ids["overall"].append(qid)
        if a.benchmark == "m3docvqa":
            group_ids["+".join(row["support_modalities"])].append(qid)
        else:
            group_ids[row["task_type"]].append(qid)
            group_ids[row["task_type"] + ("+question_image" if row["question_has_image"] else "+text_question")].append(qid)
    groups = {}
    for name, qids in group_ids.items():
        stats = {}
        for metric in metric_names:
            ids = [q for q in sorted(qids) if baseline[q].get(metric) is not None and variant[q].get(metric) is not None]
            if not ids:
                continue
            first, second = {q: baseline[q] for q in ids}, {q: variant[q] for q in ids}
            stats[metric] = paired_comparison(first, second, metric)
            stats[metric]["excluded_unscored_questions"] = len(qids) - len(ids)
            if a.benchmark == "memgallery":
                stats[metric]["scenario_cluster_bootstrap"] = cluster_paired_bootstrap(first, second, metric)
            if all(first[q][metric] in (0, 1) and second[q][metric] in (0, 1) for q in ids):
                stats[metric]["gain_question_ids"] = [q for q in ids if first[q][metric] == 0 and second[q][metric] == 1]
                stats[metric]["loss_question_ids"] = [q for q in ids if first[q][metric] == 1 and second[q][metric] == 0]
        groups[name] = stats
    comparisons[summary["arm"]] = {"metrics_sha256": sha256(directory / "question_metrics.jsonl"), "groups": groups}

def failure_audit(rows):
    # Heuristic buckets describe observed evidence delivery, not a causal label.
    cases = defaultdict(list)
    for qid, row in sorted(rows.items()):
        if row["em"] == 1:
            continue
        judge = row.get("llm_accuracy" if a.benchmark == "m3docvqa" else "llm_judge_official")
        if judge is not None and judge > 0:
            cases["metric_disagreement_requires_manual_review"].append(qid)
        delivered = row.get("figure_delivery_recall" if a.benchmark == "m3docvqa" else "image_delivery_recall")
        candidate = row.get("figure_candidate_recall")
        if delivered is None:
            bucket = "unmapped_visual_evidence"
        elif delivered == 1:
            bucket = "gold_images_delivered_but_em_incorrect"
        elif candidate is not None and candidate > delivered:
            bucket = "gold_candidate_found_but_not_all_delivered"
        else:
            bucket = "gold_visual_evidence_not_all_retrieved_or_delivered"
        cases[bucket].append(qid)
    return {"counts": {k: len(v) for k, v in cases.items()}, "question_ids": dict(cases),
            "caveat": "Metric disagreement is not a confirmed judge error. Delivered images do not establish that all text/table hops were read. Manual case review is required."}

result = {"benchmark": a.benchmark, "baseline_arm": base_summary["arm"],
          "baseline_metrics_sha256": sha256(a.baseline / "question_metrics.jsonl"),
          "gold_sha256": base_summary["gold_sha256"], "comparisons": comparisons,
          "failure_audits": {base_summary["arm"]: failure_audit(baseline)},
          "inference_limits": "Nominal unadjusted 95% intervals and exact McNemar p-values; subgroup/multiple-endpoint findings are exploratory. A 50-question sample does not establish benchmark-wide performance."}
for directory in a.variant:
    summary = json.loads((directory / "summary.json").read_text())
    result["failure_audits"][summary["arm"]] = failure_audit(load(directory))
dump_json(a.output, result)
print(json.dumps({"output": str(a.output), "arms": list(comparisons), "questions": len(baseline)}))
