"""Finalize only the completed collection hash; no adaptive candidate selection."""
import hashlib,json
from pathlib import Path
REPO=Path('/content/local_fast/lsdrag-code');ROOT=Path('/content/local_fast/lsdrag-text-v1/results')
NAME='decision_constrained_stop120_20261024';COLLECTION='decision_h3_stop_coverage48_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=read(REPO/'lab/reports'/COLLECTION/'independent_verification.json');assert audit['verified']
    m=read(REPO/'lab/registry'/COLLECTION/'manifest.json')
    p=read(REPO/'lab/registry/decision_constrained_stop72_20261024/plan.json')
    assert m['fit120']==p['fit']+m['train'] and not set(m['train'])&set(m['prior_outcome_questions'])
    p['fit']=m['fit120'];p['inputs'] += [str(ROOT/COLLECTION/'states.pt')];p['collections'] += [COLLECTION]
    assert sha(Path(p['inputs'][-1]))==audit['states_sha256']
    for name,h in p['hashes'].items():assert sha(Path(name))==h
    for name in [p['inputs'][-1],str(REPO/'scripts/prepare_decision_constrained_stop120.py'),str(REPO/'scripts/fit_decision_constrained_stop120.py'),str(REPO/'docs/JEV_H3_STOP_COVERAGE48_20261024.md')]:p['hashes'][name]=sha(Path(name))
    assert len(set(p['fit']))==120 and not set(p['fit'])&set(p['excluded'])
    out=REPO/'lab/registry'/NAME;out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
    print(out/'plan.json')
if __name__=='__main__':main()
