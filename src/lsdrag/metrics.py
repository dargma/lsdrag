"""Question-level scoring; official MMQA scorer remains upstream code."""
import importlib
import importlib.util
from pathlib import Path
import re
import string
import subprocess
import sys

import numpy as np
from scipy.stats import binomtest

M3DOC_REVISION = "29e6ac2294d6b87075a1d45b8a8df175b214248a"
MEMGALLERY_REVISION = "a93959e1e978a6a7d77798ae92c2ffe41c538c62"


def official_scorer(checkout):
    checkout = Path(checkout).resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != M3DOC_REVISION:
        raise ValueError("Unexpected M3DocRAG evaluator revision")
    relative = "src/m3docrag/datasets/m3_docvqa"
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", relative], cwd=checkout, text=True).strip():
        raise ValueError("Official evaluation source has local modifications")
    sys.path.insert(0, str(checkout / "src"))
    module = importlib.import_module("m3docrag.datasets.m3_docvqa.evaluate")
    if not Path(module.__file__).resolve().is_relative_to(checkout):
        raise ValueError("Wrong evaluator package loaded")
    return module


def arag_normalize(text):
    text = "".join(c for c in str(text).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def memgallery_scorer(checkout):
    checkout = Path(checkout).resolve()
    path = checkout / "benchmark/memengine/evaluate/evaluation.py"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != MEMGALLERY_REVISION:
        raise ValueError("Unexpected Mem-Gallery evaluator revision")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", str(path)], cwd=checkout, text=True).strip():
        raise ValueError("Official Mem-Gallery evaluation source has modifications")
    spec = importlib.util.spec_from_file_location("lsdrag_official_memgallery_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contain_all(prediction, gold_answers):
    """A-RAG substring metric extended to require every list-answer item."""
    prediction = arag_normalize(prediction)
    gold = [arag_normalize(x) for x in gold_answers]
    return float(bool(prediction and gold) and all(x and x in prediction for x in gold))


def answer_scores(prediction, gold, scorer):
    targets = [str(a["answer"]) for a in gold["answers"]]
    answers = prediction["answers"]
    if prediction["status"] == "failed":
        raise ValueError("Infrastructure failures must be resolved, not scored as wrong answers")
    return {"em": float(scorer.list_em(answers, targets)), "f1": float(scorer.list_f1(answers, targets)), "contain": contain_all("; ".join(answers), targets)}


def document_read_scores(prediction, gold, chunks):
    required = {c["doc_id"] for c in gold["supporting_context"]}
    read = {alias for key in prediction["read_chunk_ids"] for alias in chunks[key]["document_ids"]}
    if not required:
        raise ValueError("Question has no supporting documents")
    return {"document_recall": len(required & read) / len(required), "all_documents_read": float(required <= read), "support_modalities": sorted({c["doc_part"] for c in gold["supporting_context"]}), "question_type": gold["metadata"]["type"]}


def paired_comparison(baseline, variant, metric, seed=20261013, samples=10000):
    if set(baseline) != set(variant) or not baseline:
        raise ValueError("Paired comparison requires identical nonempty question ID sets")
    ids = sorted(baseline)
    a = np.array([baseline[q][metric] for q in ids], dtype=float)
    b = np.array([variant[q][metric] for q in ids], dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Unscored values must not enter paired tests")
    delta = b - a
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    result = {"n": len(ids), "metric": metric, "baseline_mean": float(a.mean()), "variant_mean": float(b.mean()), "difference": float(delta.mean()), "paired_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(), "bootstrap_samples": samples, "seed": seed}
    if set(a) <= {0, 1} and set(b) <= {0, 1}:
        gains = int(((a == 0) & (b == 1)).sum())
        losses = int(((a == 1) & (b == 0)).sum())
        result.update(gains=gains, losses=losses, mcnemar_exact_pvalue=float(binomtest(gains, gains + losses, .5).pvalue) if gains + losses else 1.)
    return result
