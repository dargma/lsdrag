"""Same frozen constrained learner on the preregistered expanded training union."""
import json
from pathlib import Path
import fit_decision_constrained_stop72 as trainer
NAME='decision_constrained_stop120_20261024'
COLLECTION='decision_h3_stop_coverage48_20261024'
if __name__=='__main__':
    audit=json.loads((trainer.REPO/'lab/reports'/COLLECTION/'independent_verification.json').read_text())
    assert audit['verified']
    plan=json.loads((trainer.REPO/'lab/registry'/NAME/'plan.json').read_text())
    path=trainer.ROOT/COLLECTION/'states.pt'
    assert plan['hashes'][str(path)]==audit['states_sha256']==trainer.sha(path)
    assert len(plan['fit'])==len(set(plan['fit']))==120
    trainer.NAME=NAME
    trainer.main()
