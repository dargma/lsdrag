import pytest
from lsdrag.decision_budget_targets import budget_targets


def record(qid, anchor, cont_f1, stop_f1, cont_cost, stop_cost):
    return dict(qid=qid, continue_action=0, reference_probs=([.9, .02, .02, .06] if anchor == 0
                else [.06, .02, .02, .9]), samples=[
        dict(action=0, f1=cont_f1, generated=cont_cost, errors=[]),
        dict(action=3, f1=stop_f1, generated=stop_cost, errors=[])])


def test_budget_targets_save_positive_answers_without_cheap_failure_labels():
    rows = [record('a', 0, 1, 1, 100, 10),
            record('b', 0, 1, 0, 100, 10),
            record('c', 0, 0, 0, 100, 1)]
    labels, report = budget_targets(rows)
    assert labels == {0: 1}
    assert report['target_suffix_f1'] == pytest.approx(report['reference_suffix_f1'])
    assert report['target_suffix_generated'] < report['reference_suffix_generated']


def test_quality_floor_and_neutral_ties_keep_reference():
    rows = [record('a', 0, 1, .5, 100, 10), record('b', 3, 1, 1, 10, 10)]
    labels, report = budget_targets(rows)
    assert labels == {}
    assert report['target_suffix_generated'] == pytest.approx(report['reference_suffix_generated'])
