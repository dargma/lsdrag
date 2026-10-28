"""Reproduction helper for the head-policy conclusion (baseline / BC / GRPO on dev64).

The frozen 2026-10-23 records pin exact file hashes. A fresh run rebuilds the corpus
vectors and the heads, so this writes new specs with recomputed hashes under
lab/registry/repro_head_policy/ and leaves the frozen records untouched.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO/'lab/registry'
OUT = FROZEN/'repro_head_policy'
CORPUS = ROOT/'corpora/musique_shared_20261023'
BC_RUN = ROOT/'results/decision_shared_initialization_20261023'
BC = BC_RUN/'head.pt'
GRPO = ROOT/'results/decision_shared_grpo192_20261023/grpo_iteration1.pt'
EVAL_RUN = 'decision_shared_validation64_repro'
REFERENCE = REPO/'lab/reports/decision_shared_validation64_20261023/result.json'
LABELS = {'native_fast': 'baseline (native)', 'shared_bc': '모방 (BC head)', 'shared_grpo': 'GRPO 적용'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    print(path)

def rehash_sources(spec):
    spec['source_sha256'] = {p: sha(REPO/p) for p in spec['source_sha256']}
    return spec

def corpus_hashes(spec):
    manifest = read(CORPUS/'manifest.json')
    assert manifest['embeddings_complete'], 'run index_decision_shared_corpus.py first'
    spec.update(document_sha256=manifest['document_sha256'], vectors_sha256=manifest['vectors_sha256'])
    return spec

def split():
    # Teacher collection checks its qids against this frozen train/validation/test split.
    target = ROOT/'results/decision_state_policy_iteration_20261023/manifest.json'
    assert not target.exists(), target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FROZEN/'decision_state_policy_iteration_20261023/manifest.json', target)
    print(target)

def teacher():
    spec = read(FROZEN/'decision_shared_teacher192_20261023/spec.json')
    save(OUT/'teacher_spec.json', rehash_sources(corpus_hashes(spec)))

def grpo():
    manifest = read(FROZEN/'decision_shared_grpo192_20261023/manifest.json')
    assert manifest['reference'] == str(BC)
    manifest['reference_spec']['sha256'] = sha(BC)
    manifest['input_sha256'] = {p: sha(Path(p)) for p in manifest['input_sha256']}
    save(OUT/'grpo_manifest.json', rehash_sources(manifest))

def evaluation():
    spec = read(FROZEN/'decision_shared_outcomes192_20261023/validation_spec.json')
    spec.update(run_name=EVAL_RUN, policies={
        'native_fast': {'kind': 'native'},
        'shared_bc': {'kind': 'original', 'checkpoint': str(BC), 'sha256': sha(BC)},
        'shared_grpo': {'kind': 'original', 'checkpoint': str(GRPO), 'sha256': sha(GRPO)}})
    save(OUT/'eval_spec.json', rehash_sources(corpus_hashes(spec)))

def print_table(path):
    policies = read(path)['policies']
    print(f'\n{path}\n')
    print('| 조건 | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 | 오류 |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, label in LABELS.items():
        p = policies[name]
        print(f"| {label} | {100*p['f1']:.2f} | {p['generated_tokens']:.1f} | {p['prompt_tokens']:,.0f} | {p['tool_calls']:.2f} | {p['errors']} |")

def table(run):
    print_table(REFERENCE)
    if run:
        print_table(ROOT/'results'/run/'summary.json')
    if (BC_RUN/'manifest.json').exists():
        print(f"\nBC train action accuracy: {read(BC_RUN/'manifest.json')['train_accuracy']:.6f} (reference 0.998501 = 666/667)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['split', 'teacher', 'grpo', 'eval', 'table'])
    ap.add_argument('--run', help='table only: evaluation run name under ROOT/results to compare with the reference')
    args = ap.parse_args()
    if args.step == 'table':
        table(args.run)
    else:
        dict(split=split, teacher=teacher, grpo=grpo, eval=evaluation)[args.step]()

if __name__ == '__main__':
    main()
