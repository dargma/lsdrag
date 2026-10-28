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
GRPO_RUN = 'decision_shared_grpo192_20261023'
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

def deadline(spec, seconds):
    # Wall-clock cap only; raise it on a shared GPU so slower rollouts do not abort the run.
    if seconds:
        spec['max_seconds'] = seconds
    return spec

def head(checkpoint):
    return {'kind': 'original', 'checkpoint': str(checkpoint), 'sha256': sha(checkpoint)}

def teacher(args):
    spec = read(FROZEN/'decision_shared_teacher192_20261023/spec.json')
    save(OUT/'teacher_spec.json', deadline(rehash_sources(corpus_hashes(spec)), args.max_seconds))

def grpo(args):
    manifest = read(FROZEN/'decision_shared_grpo192_20261023/manifest.json')
    assert manifest['reference'] == str(BC)
    manifest['reference_spec']['sha256'] = sha(BC)
    manifest['input_sha256'] = {p: sha(Path(p)) for p in manifest['input_sha256']}
    manifest.update(run_name=args.name or GRPO_RUN, kl_coef=args.kl)
    save(OUT/f"{manifest['run_name']}_manifest.json", deadline(rehash_sources(manifest), args.max_seconds))

def evaluation(args):
    spec = read(FROZEN/'decision_shared_outcomes192_20261023/validation_spec.json')
    policies = {'native_fast': {'kind': 'native'}, 'shared_bc': head(BC)} if not args.heads_only else {}
    for run in args.grpo or [GRPO_RUN]:
        policies['shared_grpo' if run == GRPO_RUN else run] = head(ROOT/'results'/run/f'grpo_iteration{args.iteration}.pt')
    spec.update(run_name=args.name or EVAL_RUN, policies=policies)
    save(OUT/f"{spec['run_name']}_spec.json", deadline(rehash_sources(corpus_hashes(spec)), args.max_seconds))

def print_table(path):
    policies = read(path)['policies']
    print(f'\n{path}\n')
    print('| 조건 | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 | 오류 |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, p in policies.items():
        if path == REFERENCE and name not in LABELS:
            continue  # the reference run also scored policies from other experiments
        label = LABELS.get(name, name)
        print(f"| {label} | {100*p['f1']:.2f} | {p['generated_tokens']:.1f} | {p['prompt_tokens']:,.0f} | {p['tool_calls']:.2f} | {p['errors']} |")

def table(runs):
    print_table(REFERENCE)
    for run in runs or []:
        print_table(ROOT/'results'/run/'summary.json')
    if (BC_RUN/'manifest.json').exists():
        print(f"\nBC train action accuracy: {read(BC_RUN/'manifest.json')['train_accuracy']:.6f} (reference 0.998501 = 666/667)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['split', 'teacher', 'grpo', 'eval', 'table'])
    ap.add_argument('--run', nargs='+', help='table: evaluation run names under ROOT/results to print after the reference')
    ap.add_argument('--name', help='grpo/eval: run name (defaults reproduce the original run)')
    ap.add_argument('--kl', type=float, default=.02, help='grpo: KL(pi || pi_BC) coefficient; original .02')
    ap.add_argument('--grpo', nargs='+', help='eval: GRPO run names to evaluate (default: the original-named run)')
    ap.add_argument('--iteration', type=int, default=1, help='eval: GRPO checkpoint iteration to evaluate')
    ap.add_argument('--heads-only', action='store_true', help='eval: skip native_fast and shared_bc')
    ap.add_argument('--max-seconds', type=int, help='teacher/grpo/eval: override the wall-clock cap')
    args = ap.parse_args()
    if args.step == 'table':
        table(args.run)
    elif args.step == 'split':
        split()
    else:
        dict(teacher=teacher, grpo=grpo, eval=evaluation)[args.step](args)

if __name__ == '__main__':
    main()
