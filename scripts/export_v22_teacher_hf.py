"""Export audited teacher data for portable Hugging Face reuse; no network writes."""
import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import ROOT, save, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--source-snapshot', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repo-id', required=True)
    a = ap.parse_args()
    audit = json.loads(a.audit.read_text())
    assert audit['audit_passed']
    assert json.loads((a.teacher/'status.json').read_text())['phase'] == 'complete'
    assert sha(a.teacher/'teacher_states.pt') == audit['teacher_sha256']
    split = json.loads(a.split.read_text())
    memberships = {qid: part for part, key in [('train', 'fit'), ('validation', 'validation')] for qid in split[key]}
    raw = {q['id']: q for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open()
           if (q := json.loads(line))['id'] in memberships}
    assert len(raw) == len(memberships) == 576
    a.output.mkdir(parents=True, exist_ok=False)
    episode_rows = {k: [] for k in ('train', 'validation')}
    sys_path = __import__('sys').path
    sys_path.insert(0, str(ROOT/'source/musique_scorer'))
    from metrics.answer import compute_exact
    for p in sorted((a.teacher/'episodes').glob('*.json')):
        ep = json.loads(p.read_text()); qid = ep['qid']; q = raw[qid]
        episode_rows[memberships[qid]].append(dict(
            qid=qid, question=q['question'], reference_answer=q['answer'], answer_aliases=q['answer_aliases'],
            prediction=ep['answer'], em=max(compute_exact(ans, ep['answer']) for ans in [q['answer']]+q['answer_aliases']),
            f1=ep['f1'], tool_calls=ep['tool_calls'], generated_tokens=ep['generated_tokens'],
            prompt_tokens=ep['prompt_tokens'], errors=ep['errors'],
            trajectory_json=json.dumps(ep['trajectory'], ensure_ascii=False),
            decisions_json=json.dumps(ep['decisions'], ensure_ascii=False),
            raw_episode_json=json.dumps(ep, ensure_ascii=False)))
    labels = {k: [] for k in ('train', 'validation')}
    states = torch.load(a.teacher/'teacher_states.pt', weights_only=True, map_location='cpu')
    for i, r in enumerate(states):
        labels[memberships[r['qid']]].append(dict(
            state_index=i, qid=r['qid'], step_id=r['step_id'], action=int(r['action']),
            legal_actions=r['legal'].tolist(), candidate_ids=r['candidate_ids'],
            candidate_spans=[list(s) for s in r['candidate_spans']], context_ids=list(r['context_ids']),
            pointer_target=r['pointer_target'], pointer_source=r['pointer_source'],
            teacher_chunk_ids=r['teacher_chunk_ids'], gold_candidate_indices=r['gold_candidate_indices']))
    for config, data in [('episodes', episode_rows), ('labels', labels)]:
        folder = a.output/config; folder.mkdir()
        # One shared schema prevents a null-only field in one split from changing types.
        schema = pa.Table.from_pylist(data['train']+data['validation']).schema
        for part, rows in data.items():
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder/f'{part}.parquet', compression='zstd')
    artifacts = a.output/'artifacts'; artifacts.mkdir()
    shutil.copy2(a.teacher/'teacher_states.pt', artifacts/'teacher_states.pt')
    provenance = a.output/'provenance'; provenance.mkdir()
    for name in ('spec.json', 'summary.json', 'status.json', 'pointer_coverage.json'):
        shutil.copy2(a.teacher/name, provenance/name)
    shutil.copy2(a.audit, provenance/'teacher_audit.json')
    shutil.copy2(a.split, provenance/'kd_split.json')
    shutil.copytree(a.source_snapshot, provenance/'teacher_source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT/'source/musique_scorer/LICENSE', a.output/'LICENSE')
    counts = {config: {part: len(rows) for part, rows in data.items()}
              for config, data in [('episodes', episode_rows), ('labels', labels)]}
    readme = f'''---
language:
- en
license: cc-by-4.0
task_categories:
- question-answering
configs:
- config_name: episodes
  default: true
  data_files:
  - split: train
    path: episodes/train.parquet
  - split: validation
    path: episodes/validation.parquet
- config_name: labels
  data_files:
  - split: train
    path: labels/train.parquet
  - split: validation
    path: labels/validation.parquet
---

# lsdrag v2.2 generated teacher data

Completed native greedy teacher trajectories from Qwen/Qwen3-4B-Instruct-2507
(revision cdbee75f17c01a7cc42f958dc650907174af0554), MuSiQue-Ans v1.0 train,
text-only shared corpus, keyword_search / semantic_search / read_chunk / FINAL.
Generated outputs may be incorrect. All episodes, including failures, are retained.
Source: [MuSiQue by Trivedi et al.](https://github.com/StonyBrookNLP/musique),
CC BY 4.0. This dataset adds generated trajectories, action labels and offline
gold-present pointer labels; it is not an official MuSiQue release.

## Load

```python
from datasets import load_dataset
episodes = load_dataset("{a.repo_id}", "episodes", token=True)
labels = load_dataset("{a.repo_id}", "labels", token=True)
```

Pin `revision` to the upload commit for reproducibility. `train` is KD fit459;
`validation` is the preregistered internal117 split. Both come from original
MuSiQue train. They are qid-disjoint, but 59 validation questions share atomic
question families with fit. This is NOT an independent benchmark holdout.
Original dev/test data are not uploaded in this package.

Episodes expose question, reference answer/aliases, generated answer, EM/F1,
costs and errors. Variable tool argument/result structures are serialized as
JSON strings in trajectory_json and raw_episode_json to preserve exact values.
Labels expose stable state_index, action IDs (0 keyword, 1 semantic, 2 read,
3 FINAL), masks, current visible candidate IDs/spans, input token IDs, pointer
targets and label source. Candidate K varies per state. Pointer target indices
refer to that row's candidate_ids, not global document numbers. STOP is implicit
after the last target unless max8 or candidate exhaustion forces termination.
Gold information is offline supervision/scoring ONLY; never put it into inputs.

For exact KD features and tensor dtypes:
```python
import torch
from huggingface_hub import hf_hub_download
path = hf_hub_download("{a.repo_id}", "artifacts/teacher_states.pt", repo_type="dataset", token=True)
states = torch.load(path, weights_only=True, map_location="cpu")
# labels[state_index] maps to states[state_index]; pack/candidate_features are included.
```

The tensor artifact contains all splits. Always apply provenance/kd_split.json
before training; do not fit internal validation states. Tool labels imitate the
teacher; read labels use eligible gold candidates when present, otherwise valid
teacher choices. Invalid read labels are excluded; their QA/tool states remain.
This is teacher KD plus gold supervision, not pure teacher imitation.

Counts: `{json.dumps(counts)}`.
Coverage gate passed: `{audit['coverage']['passed']}`. Data audit passed: true.
These facts do not establish trained-policy quality or GRPO success.
`provenance/` contains original specs, audits and the exact collection source.
Absolute paths in source specs describe the original machine; use local restored
paths on a new machine. Model/corpus weights are not bundled. All packaged file
hashes are in manifest.json. No authentication credentials are included.
'''
    (a.output/'README.md').write_text(readme)
    manifest = dict(format='lsdrag-v22-teacher-v1', repo_id=a.repo_id, counts=counts,
                    audit_passed=True, coverage=audit['coverage'],
                    files={str(p.relative_to(a.output)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                           for p in sorted(a.output.rglob('*')) if p.is_file()})
    save(a.output/'manifest.json', manifest)
    print(json.dumps({'output': str(a.output), 'counts': counts, 'files': len(manifest['files'])}, indent=2))


if __name__ == '__main__':
    main()
