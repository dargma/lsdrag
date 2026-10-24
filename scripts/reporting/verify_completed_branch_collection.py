"""Independently re-score complete training collections and verify archived bytes."""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import torch

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(name):
    assert Path(name).name == name
    run = ROOT / 'results' / name
    status = read(run / 'status.json')
    assert status['phase'] == 'complete', 'Never inspect partial outcomes for fitting'
    manifest = read(run / 'manifest.json')
    assert manifest == read(REPO / 'lab/registry' / name / 'manifest.json')
    train = set(manifest['train'])
    assert len(train) == len(manifest['train']) == 48
    assert train <= set(manifest['train_pool'])
    assert not train & set(manifest['validation'] + manifest['test'])
    for path, expected in manifest['input_sha256'].items():
        assert sha(Path(path)) == expected, path
    # Historical source hashes are provenance. Check current sources only when
    # identical source execution is required, separately from archived integrity.
    archive = read(REPO / 'lab/registry' / name / 'archive.json')
    assert sha(Path(archive['path'])) == archive['sha256']
    archived = set()
    with tarfile.open(archive['path']) as bundle:
        for member in bundle:
            if member.isdir():
                continue
            assert member.isfile() and not member.issym() and not member.islnk()
            relative = Path(member.name)
            assert relative.parts[0] == name and '..' not in relative.parts
            assert str(relative) not in archived
            archived.add(str(relative))
            assert hashlib.sha256(bundle.extractfile(member).read()).hexdigest() == sha(run.parent / relative)
    assert archived == {str(p.relative_to(run.parent)) for p in run.rglob('*') if p.is_file()}
    raw = {}
    for split in ('train', 'dev'):
        for line in (ROOT / 'source/musique' / f'musique_ans_v1.0_{split}.jsonl').open():
            question = json.loads(line)
            if question['id'] in train:
                assert question['id'] not in raw
                raw[question['id']] = question
    assert set(raw) == train
    sys.path.insert(0, str(ROOT / 'source/musique_scorer'))
    from metrics.answer import compute_f1
    episodes = [read(p) for p in (run / 'episodes').glob('*.json')]
    by_key = {(e['qid'], e['tag']): e for e in episodes}
    assert len(by_key) == len(episodes) == status['episodes']
    assert {e['qid'] for e in episodes} == train
    for episode in episodes:
        question = raw[episode['qid']]
        score = max(compute_f1(answer, episode['answer'])
                    for answer in [question['answer']] + question['answer_aliases'])
        assert score == episode['f1'], (episode['qid'], episode['tag'])
    for field, key in [('generated_tokens', 'generated'), ('prompt_tokens', 'inputs')]:
        assert sum(e[field] for e in episodes) == status[key]
    assert status['episodes'] <= manifest['budget_episodes']
    assert status['generated'] <= manifest['max_generated']
    assert status['inputs'] <= manifest['max_inputs']
    states = torch.load(run / 'states.pt', weights_only=True, map_location='cpu')
    assert len(states) == status['states']
    assert {r['qid'] for r in states} == train
    assert len({(r['qid'], r['turn']) for r in states}) == len(states)
    expected_keys = {(qid, 'collector') for qid in train}
    for record in states:
        assert record['pack'].dtype == torch.float32 and torch.isfinite(record['pack']).all()
        assert len(record['samples']) == int(record['legal'].sum()) * manifest['repeats_per_action']
        sample_keys = {(s['action'], s['rep']) for s in record['samples']}
        assert len(sample_keys) == len(record['samples'])
        assert sample_keys == {(a, rep) for a in range(4) if record['legal'][a]
                               for rep in range(manifest['repeats_per_action'])}
        for sample in record['samples']:
            tag = f"t{record['turn']}_a{sample['action']}_r{sample['rep']}"
            key = (record['qid'], tag)
            expected_keys.add(key)
            episode = by_key[key]
            for source, target in [('f1', 'f1'), ('generated', 'generated_tokens'),
                                   ('inputs', 'prompt_tokens'), ('errors', 'errors')]:
                assert sample[source] == episode[target]
            assert episode['reader_calls'][0]['messages_sha256'] == record['prefix_sha256']
            assert episode['decisions'][0]['action'] == sample['action']
            assert episode['decisions'][0]['branch_intervention']
    assert expected_keys == set(by_key), 'Missing or unaccounted episode'
    summary = read(run / 'summary.json')
    assert summary['state_tensor_sha256'] == sha(run / 'states.pt')
    assert summary['counts'] == {key: status[key] for key in ['episodes', 'generated', 'inputs', 'states']}
    result = dict(verified=True, run=name, questions=len(train), states=len(states),
                  episodes=len(episodes), generated=status['generated'], inputs=status['inputs'],
                  errors=sum(bool(e['errors']) for e in episodes), archive_sha256=archive['sha256'],
                  archived_files=len(archived), scores_recomputed=True,
                  independent_quality_claim=False)
    target = REPO / 'lab/reports' / name / 'independent_verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
