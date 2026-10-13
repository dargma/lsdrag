"""Publish only completed new-condition results; do not launch inference."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/content/local_fast/lsdrag-work')
REPO = ROOT / 'repo'
RESULT = ROOT / 'results/three_arm_arag8_textquery'
REPORT = REPO / 'lab/reports'
ASSETS = REPORT / 'assets/three_arm'
NAMES = {'B0': 'Qwen text', 'C1': 'Google text-only', 'G1': 'Google multimodal'}

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fmt(value):
    return '—' if value is None else f'{100 * value:.1f}'

def main():
    assert (RESULT / 'COMPLETE.json').exists(), 'All six runs must finish before publishing'
    ASSETS.mkdir(parents=True, exist_ok=True)
    summaries, paired, scored = {}, {}, {}
    for benchmark in ('m3docvqa', 'memgallery'):
        destination = REPORT / 'three_arm_arag8' / benchmark
        destination.mkdir(parents=True, exist_ok=True)
        paired[benchmark] = json.loads((RESULT / benchmark / 'paired_comparison.json').read_text())
        shutil.copyfile(RESULT / benchmark / 'paired_comparison.json', destination / 'paired_comparison.json')
        manifests = []
        for arm in NAMES:
            source = RESULT / benchmark / (arm + '_scores')
            summary = json.loads((source / 'summary.json').read_text())
            question_rows = rows(source / 'question_metrics.jsonl')
            summaries[benchmark, arm] = summary
            scored[benchmark, arm] = {x['qid']: x for x in question_rows}
            manifest = json.loads((RESULT / benchmark / arm / 'run_manifest.json').read_text())
            manifests.append(manifest)
            assert manifest['config']['max_loops'] == 8
            assert manifest['question_image_policy'] == 'text-only'
            assert sorted(manifest['index_manifests']) == {'B0': ['qwen_text'], 'C1': ['gemma_text'], 'G1': ['gemma_image', 'gemma_text']}[arm]
            for qid in manifest['question_ids']:
                prediction = json.loads((RESULT / benchmark / arm / (qid + '.json')).read_text())
                assert prediction['question_image_count'] == 0 and prediction['loops'] <= 8
                assert prediction['status'] != 'failed'
            for file in ('summary.json', 'question_metrics.jsonl'):
                shutil.copyfile(source / file, destination / (arm + '_' + file))
            shutil.copyfile(RESULT / benchmark / arm / 'run_manifest.json', destination / (arm + '_run_manifest.json'))
        for key in ('questions_sha256', 'question_ids', 'corpus_sha256', 'prompt_sha256', 'config', 'code_revision', 'question_image_policy'):
            assert all(m[key] == manifests[0][key] for m in manifests), (benchmark, key)
    # Add candidate-image recall for memory search using saved traces, never new model calls.
    corpus = ROOT / 'data/mem_gallery_prepared/corpus'
    chunks = {x['chunk_id']: x for x in rows(corpus / 'chunks.jsonl')}
    figures = {x['figure_id']: x for x in rows(corpus / 'figures.jsonl')}
    gold = {x['qid']: x for x in rows(ROOT / 'data/mem_gallery_prepared/development/gold.jsonl')}
    mem_candidates = {}
    for arm in NAMES:
        records = []
        for qid, q in gold.items():
            if q['point'] == 'VS':
                ids = {s.strip() for s in str(q['answer']).split(',')}
                targets = {fid for fid, f in figures.items() if f['scenario'] == q['scenario'] and f['source_image_id'] in ids}
                mapped = len(targets) == len(ids)
            else:
                targets = {fid for c in q['clue'] for fid in chunks[q['scenario'] + ':' + c]['figure_ids']}
                mapped = bool(targets)
            prediction = json.loads((RESULT / 'memgallery' / arm / (qid + '.json')).read_text())
            found = set()
            for event in prediction['trajectory']:
                if not event['tool_name'].endswith('search') or not isinstance(event['result'], list):
                    continue
                for item in event['result']:
                    if item.get('figure_id'): found.add(item['figure_id'])
                    if item.get('chunk_id') in chunks: found.update(chunks[item['chunk_id']]['figure_ids'])
            recall = len(found & targets) / len(targets) if mapped and targets else None
            records.append({'qid': qid, 'image_candidate_recall': recall, 'gold_figure_ids': sorted(targets), 'candidate_figure_ids': sorted(found)})
        values = [x['image_candidate_recall'] for x in records if x['image_candidate_recall'] is not None]
        mem_candidates[arm] = {'mean': sum(values) / len(values) if values else None, 'scored_n': len(values)}
        (REPORT / 'three_arm_arag8/memgallery' / (arm + '_candidate_recall.json')).write_text(json.dumps({'overall': mem_candidates[arm], 'questions': records}, indent=2) + '\n')
    print(json.dumps({'summaries': {b: {a: summaries[b,a]['overall'] for a in NAMES} for b in ('m3docvqa','memgallery')}, 'mem_candidates':mem_candidates}, indent=2))
    # A separate authoring step chooses interpretations after inspecting paired statistics.
    (ROOT / 'recovery/report_inputs.json').write_text(json.dumps({'mem_candidates': mem_candidates, 'paired': paired}, indent=2))
    print('REPORT INPUTS EXPORTED; prediction and comparison invariants verified')

if __name__ == '__main__':
    main()
