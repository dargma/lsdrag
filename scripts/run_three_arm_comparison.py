"""Remeasure the three isolated embedder conditions on frozen development inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/content/local_fast/lsdrag-work')
REPO = Path(__file__).resolve().parents[1]
OPTIONS = argparse.ArgumentParser()
OPTIONS.add_argument('--evaluation-set', choices=['development', 'additional50'], default='development')
OPTIONS.add_argument('--benchmark', choices=['all', 'm3docvqa', 'memgallery'], default='all')
ARGS = OPTIONS.parse_args()
SET = ARGS.evaluation_set
OUT = ROOT / ('results/three_arm_arag8_textquery' if SET == 'development' else 'results/three_arm_additional50')
LOGS = OUT / 'logs'
EMBED = ROOT / 'embedding-env/bin/python'
RUNTIME = ROOT / 'runtime/bin/python'
ARMS = ('B0', 'C1', 'G1')
ENV = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}


def command(name, python, script, args):
    print('START', name, flush=True)
    with (LOGS / (name + '.log')).open('a') as log:
        subprocess.run([str(python), str(REPO / 'scripts' / script), *map(str, args)], cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('COMPLETE', name, flush=True)


def one(benchmark, arm):
    mem = benchmark == 'memgallery'
    corpus = ROOT / ('data/mem_gallery_prepared/corpus' if mem else ('data/ir_development' if SET == 'development' else 'data/ir_final288'))
    inputs = ROOT / (('data/mem_gallery_prepared/development' if SET == 'development' else 'data/mem_gallery_prepared/evaluation') if mem else ('data/manifests/development' if SET == 'development' else 'data/manifests/evaluation_50'))
    indexes = ROOT / ('indexes_memgallery' if mem else ('indexes_development' if SET == 'development' else 'indexes_final288'))
    run = OUT / benchmark / arm
    command(f'{benchmark}-{arm}', EMBED, 'run_experiment.py', [
        '--arm', arm, '--benchmark', benchmark, '--corpus', corpus, '--indexes', indexes,
        '--questions', inputs / 'questions.jsonl', '--output', run, '--cache-dir', ROOT / 'hf',
        '--max-loops', '8', '--question-image-policy', 'text-only',
        '--image-delivery-policy', 'agent-read', '--resume'])
    manifest = json.loads((run / 'run_manifest.json').read_text())
    for qid in manifest['question_ids']:
        row = json.loads((run / (qid + '.json')).read_text())
        assert row['status'] != 'failed' and row['loops'] <= 8
        assert row['question_image_count'] == 0
    args = ['--run', run, '--gold', inputs / 'gold.jsonl', '--corpus', corpus,
            '--upstream', ROOT / ('upstream/mem_gallery' if mem else 'upstream/m3docrag'),
            '--output', OUT / benchmark / (arm + '_scores'), '--judge']
    if not mem:
        args += ['--visual-mapping', REPO / ('lab/reports/development_alignment_review.json' if SET == 'development' else 'lab/reports/additional50_alignment_review.json')]
    command(f'{benchmark}-{arm}-score', RUNTIME, 'score_memgallery.py' if mem else 'score_m3doc.py', args)
    command(f'{benchmark}-{arm}-backup', RUNTIME, 'backup_local.py', [
        '--source-root', ROOT, '--destination', f'/content/drive/MyDrive/lsdrag-backups/20261013/{OUT.name}/{benchmark}/{arm}',
        '--include', str(run.relative_to(ROOT)), '--include', str((OUT / benchmark / (arm + '_scores')).relative_to(ROOT)),
        '--report', OUT / benchmark / (arm + '_backup.json')])


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    for benchmark in (('m3docvqa', 'memgallery') if ARGS.benchmark == 'all' else (ARGS.benchmark,)):
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda arm: one(benchmark, arm), ARMS))
        command(benchmark + '-paired', RUNTIME, 'compare_runs.py', [
            '--baseline', OUT / benchmark / 'B0_scores', '--variant', OUT / benchmark / 'C1_scores',
            '--variant', OUT / benchmark / 'G1_scores', '--benchmark', benchmark,
            '--output', OUT / benchmark / 'paired_comparison.json'])
    if not all((OUT / b / 'paired_comparison.json').exists() for b in ('m3docvqa', 'memgallery')):
        print('REQUESTED BENCHMARK COMPLETE; OTHER BENCHMARK PENDING', flush=True)
        return
    (OUT / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'arms': ARMS,
        'benchmarks': ['m3docvqa', 'memgallery'], 'max_agent_iterations': 8, 'question_image_policy': 'text-only'}, indent=2))
    print('ALL SIX RUNS SCORED AND BACKED UP', flush=True)


if __name__ == '__main__':
    main()
