"""Freeze the sole dev comparison only after the expanded full train gate passes."""
import hashlib,json
from pathlib import Path
R=Path('/content/local_fast/lsdrag-code');ROOT=Path('/content/local_fast/lsdrag-text-v1/results')
def read(p):return json.loads(p.read_text())
def main():
    train=read(R/'lab/reports/decision_constrained_stop120_train_eval_20261024/verification.json')
    pipeline=read(ROOT/'decision_constrained_stop120_pipeline_20261024.json')
    assert train['verified'] and train['train_gate'] and pipeline['phase']=='complete'
    name='decision_constrained_stop120_20261024'
    fit=read(ROOT/name/'summary.json');assert fit['learning_progress']
    spec=read(R/'lab/registry/decision_constrained_stop72_20261024/validation_spec.json')
    spec.update(run_name='decision_constrained_stop120_validation64_20261024',policies={'h3':spec['policies']['h3'],'constrained_stop':fit['candidate']})
    for p,h in spec['source_sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    for p in spec['policies'].values():assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest()==p['sha256']
    out=R/'lab/registry'/name/'validation_spec.json';assert not out.exists();out.write_text(json.dumps(spec,indent=2)+'\n');print(out)
if __name__=='__main__':main()
