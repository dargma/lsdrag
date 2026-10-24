"""Native H2 reference collection with passive causal document-feature capture."""
import json
import sys
from pathlib import Path
import evaluate_decision_shared_corpus as evaluator
from evaluate_decision_stop_value_h2 import load_head
from lsdrag.decision_read_capture import ReadCaptureReader
from run_decision_policy_pilot import ROOT

if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[sys.argv.index('--spec')+1]).read_text())
    questions = [json.loads(x) for x in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open()]
    selected = [q for q in questions if q['id'] in spec['qids']]
    assert len(selected) == len(spec['qids']) == 48
    assert len({q['question'] for q in selected}) == 48
    ReadCaptureReader.question_ids = {q['question']:q['id'] for q in selected}
    ReadCaptureReader.output_dir = ROOT/'results'/spec['run_name']/'read_features'
    evaluator.RecoveryPolicyReader = ReadCaptureReader
    evaluator.load_head = load_head
    evaluator.main()
