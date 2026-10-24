"""Freeze dependent stage checkpoint/input/source hashes before execution."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1];REG=REPO/'lab/registry/qwen35_9b_h3_20261024'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['outcomes','stop','evaluation'],required=True);args=ap.parse_args()
    t=read(REG/'teacher_spec.json');plan=read(REG/'training_plan.json');sources=list(t['source_sha256'])+['scripts/collect_qwen35_outcomes.py','scripts/fit_qwen35_outcome_heads.py','scripts/evaluate_qwen35_h3.py','src/lsdrag/decision_budget_targets.py','src/lsdrag/decision_stop_value.py']
    frozen=read(REG/'pipeline_sources.json')
    for f,h in frozen.items():assert sha(REPO/f)==h,f
    if args.stage=='evaluation':
        spec={k:t[k] for k in ['corpus','document_sha256','vectors_sha256','model']}
        spec.update(run_name='qwen35_9b_h3_validation64_20261024',qids=plan['development64'],split='validation',max_seconds=7200,max_generated=800000,max_inputs=50000000,policies={},source_sha256={f:sha(REPO/f) for f in sources})
        for name,run,kind in [('base','qwen35_9b_base_20261024','qwen35_base'),('h3','qwen35_9b_h3_20261024','qwen35_stop'),('constant','qwen35_9b_h3_20261024','qwen35_stop')]:
            path=ROOT/'results'/run/f'{name}.pt';spec['policies'][name]=dict(kind=kind,checkpoint=str(path),sha256=sha(path))
    else:
        name='qwen35_9b_outcomes48_20261024' if args.stage=='outcomes' else 'qwen35_9b_stop48_20261024'
        checkpoint=ROOT/'results'/('qwen35_9b_bootstrap_20261024/head.pt' if args.stage=='outcomes' else 'qwen35_9b_base_20261024/base.pt')
        spec=dict(run_name=name,mode='all_actions' if args.stage=='outcomes' else 'stop_pairs',qids=plan['outcome_train48'],model=t['model'],corpus=t['corpus'],checkpoint=str(checkpoint),max_seconds=14400,max_generated=2000000,max_episodes=2000 if args.stage=='outcomes' else 1000)
        files=[str(REPO/f) for f in sources]+[str(checkpoint),str(REG/'training_plan.json'),str(Path(t['corpus'])/'documents.jsonl'),str(Path(t['corpus'])/'vectors.npy')]
        spec['hashes']={f:sha(f) for f in files}
    out=REG/f'{args.stage}_spec.json';assert not out.exists(),'Do not replace a frozen stage spec';out.write_text(json.dumps(spec,indent=2)+'\n');print(out)
if __name__=='__main__':main()
