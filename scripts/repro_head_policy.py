"""Build run specs from configs/ and print result tables.

Each spec records SHA-256 of its inputs and code; the stage scripts check them
before running. Specs are written to ROOT/specs.
"""
import argparse
import json
from pathlib import Path

from common import ROOT, REPO, CONFIG, SPLIT, sha, save

OUT = ROOT/'specs'
CORPUS = Path(CONFIG['corpus'])
BC = Path(CONFIG['bc_head'])
GRPO_RUN = 'decision_shared_grpo192_20261023'
EVAL_RUN = 'decision_shared_validation64_repro'
REFERENCE = REPO/'configs/reference_dev64.json'
LABELS = {'native_fast': 'baseline (native)', 'shared_bc': '모방 (BC head)', 'shared_grpo': 'GRPO'}
MUSIQUE = [ROOT/f'source/musique/musique_ans_v1.0_{s}.jsonl' for s in ['train', 'dev']]

def sources(script):
    files = [script, 'scripts/common.py', 'prompts/text_rag_integrated_v5.txt']
    files += sorted(str(p.relative_to(REPO)) for p in (REPO/'src/lsdrag').glob('*.py'))
    return {p: sha(REPO/p) for p in files}

def corpus_spec(script, cap):
    manifest = json.loads((CORPUS/'manifest.json').read_text())
    assert manifest['embeddings_complete'], 'run index_decision_shared_corpus.py first'
    return dict(corpus=str(CORPUS), document_sha256=manifest['document_sha256'], vectors_sha256=manifest['vectors_sha256'],
                model=CONFIG['model'], source_sha256=sources(script), **cap)

def head(checkpoint):
    return {'kind': 'original', 'checkpoint': str(checkpoint), 'sha256': sha(checkpoint)}

def write(name, spec, seconds):
    if seconds:
        spec['max_seconds'] = seconds  # wall-clock cap only; raise it on a shared GPU
    save(OUT/f'{name}.json', spec)
    print(OUT/f'{name}.json')

def teacher(args):
    spec = corpus_spec('scripts/collect_decision_shared_teacher.py', CONFIG['teacher'])
    spec.update(run_name='decision_shared_teacher192_20261023', split='train', qids=SPLIT['train'], policies={'native_teacher': {'kind': 'native'}})
    write('teacher_spec', spec, args.max_seconds)

def grpo(args):
    name = args.name or GRPO_RUN
    spec = dict(CONFIG['grpo'], kl_coef=args.kl, run_name=name, train=SPLIT['train'], model=CONFIG['model'], corpus=str(CORPUS),
                reference=str(BC), source_sha256=sources('scripts/run_decision_shared_grpo.py'),
                input_sha256={str(p): sha(p) for p in [BC, CORPUS/'documents.jsonl', CORPUS/'vectors.npy', *MUSIQUE]})
    write(f'{name}_manifest', spec, args.max_seconds)

def evaluation(args):
    policies = {} if args.heads_only else {'native_fast': {'kind': 'native'}, 'shared_bc': head(BC)}
    for run in args.grpo or [GRPO_RUN]:
        policies['shared_grpo' if run == GRPO_RUN else run] = head(ROOT/'results'/run/f'grpo_iteration{args.iteration}.pt')
    spec = corpus_spec('scripts/evaluate_decision_shared_corpus.py', CONFIG['eval'])
    spec.update(run_name=args.name or EVAL_RUN, split='validation', qids=SPLIT['validation'], policies=policies)
    write(f"{spec['run_name']}_spec", spec, args.max_seconds)

def print_table(path):
    data = json.loads(path.read_text())
    policies = data.get('policies', data)
    print(f'\n{path}\n')
    print('| 조건 | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 | 오류 |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, p in policies.items():
        print(f"| {LABELS.get(name, name)} | {100*p['f1']:.2f} | {p['generated_tokens']:.1f} | {p['prompt_tokens']:,.0f} | {p['tool_calls']:.2f} | {p['errors']} |")

def table(runs):
    print_table(REFERENCE)
    for run in runs or []:
        print_table(ROOT/'results'/run/'summary.json')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['teacher', 'grpo', 'eval', 'table'])
    ap.add_argument('--run', nargs='+', help='table: evaluation runs under ROOT/results to print after the reference')
    ap.add_argument('--name', help='grpo/eval: run name')
    ap.add_argument('--kl', type=float, default=CONFIG['grpo']['kl_coef'], help='grpo: KL(pi || pi_BC) coefficient')
    ap.add_argument('--grpo', nargs='+', help='eval: GRPO runs to evaluate')
    ap.add_argument('--iteration', type=int, default=1, help='eval: GRPO checkpoint iteration')
    ap.add_argument('--heads-only', action='store_true', help='eval: skip native_fast and shared_bc')
    ap.add_argument('--max-seconds', type=int, help='teacher/grpo/eval: override the wall-clock cap')
    args = ap.parse_args()
    if args.step == 'table':
        table(args.run)
    else:
        dict(teacher=teacher, grpo=grpo, eval=evaluation)[args.step](args)

if __name__ == '__main__':
    main()
