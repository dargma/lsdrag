import copy
from types import SimpleNamespace
import numpy as np
import pytest

from lsdrag.native_distillation import native_target, NativeTeacherStudentReader
from lsdrag.tool_registry import tool_schemas, tool_contract_sha256
from lsdrag.direct_reader import DirectOutputError

TOOLS = tool_schemas('G1', 'search-linked', 'multimodal_full')


def choice(action='semantic_search', arguments='{"query":"entity"}'):
    return {'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': None,
        'tool_calls': [{'id': 'call_0', 'type': 'function',
                        'function': {'name': action, 'arguments': arguments}}]}}


def test_native_target_preserves_arguments_and_rejects_invalid_labels():
    from jsonschema import ValidationError
    valid = choice()
    target = native_target(valid, TOOLS)
    assert target['action'] == 'semantic_search'
    assert target['arguments'] == {'query': 'entity'}
    bad = copy.deepcopy(valid)
    bad['message']['tool_calls'] *= 2
    with pytest.raises(ValueError, match='multiple_actions'):
        native_target(bad, TOOLS)
    with pytest.raises(ValueError, match='truncated'):
        native_target(dict(valid, finish_reason='length'), TOOLS)
    with pytest.raises(ValueError, match='unknown_action'):
        native_target(choice('invented_tool'), TOOLS)
    with pytest.raises(ValidationError):
        native_target(choice(arguments='{}'), TOOLS)
    for content in ['', 'A', 'not JSON', '{"reasoning":"done"}']:
        with pytest.raises((ValueError, ValidationError)):
            native_target({'finish_reason': 'stop', 'message': {'content': content}}, TOOLS)
    assert native_target({'finish_reason': 'stop', 'message': {'content': '{"answers":[]}'}}, TOOLS)['action'] == 'FINAL'


def test_features_are_student_pre_output_and_invalid_teacher_has_no_label(monkeypatch):
    import hashlib
    import lsdrag.native_distillation as module
    history = [{'role': 'system', 'content': 'student system'},
               {'role': 'user', 'content': [{'type': 'text', 'text': 'question'},
                                            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}]
    original = copy.deepcopy(history)
    events, saved = [], []
    student = SimpleNamespace(torch=None, model=None, contract_sha256=tool_contract_sha256(TOOLS),
        expected_feature_contract={'allowed_system_prompt_sha256s': [hashlib.sha256(b'student system').hexdigest()]})
    vector = np.array([1., 2.])
    def make_probe(reader, mode, state_sink):
        assert reader is student and mode == 'first_token'
        def select(messages, tools):
            assert messages == original
            events.append('student')
            state_sink({'action': 'FINAL', 'index': 5, 'qid': 'q', 'messages_sha256': 'hash'}, vector.copy())
            return 'FINAL', [], {'prompt_tokens': 3, 'generated_tokens': 0}
        return SimpleNamespace(_select=select)
    monkeypatch.setattr(module.GeneratedChoiceReader, 'from_reader', make_probe)
    response = choice()
    def teacher_chat(messages, tools):
        assert messages[0]['content'] == 'teacher system'
        assert messages[1:] == original[1:]
        assert events[-1] == 'student' or tools is None
        events.append('teacher')
        vector[:] = 99  # Must not corrupt the captured student feature.
        return response, {'prompt_tokens': 7, 'generated_tokens': 4}
    wrapper = NativeTeacherStudentReader(student, SimpleNamespace(chat=teacher_chat), 'teacher system', {'revision': 'teacher'}, verify_readout=False)
    wrapper.current_qid = 'q'
    wrapper.state_sink = lambda record, feature: saved.append((record, feature))
    _, usage = wrapper.chat(history, TOOLS)
    assert history == original and events == ['student', 'teacher']
    assert saved[0][0]['action'] == 'semantic_search'
    assert saved[0][0]['student_probe_action'] == 'FINAL'
    np.testing.assert_array_equal(saved[0][1], [1., 2.])
    assert usage['prompt_tokens'] == 10
    response['finish_reason'] = 'length'
    with pytest.raises(DirectOutputError):
        wrapper.chat(history, TOOLS)
    assert len(saved) == 1
    wrapper.chat(history, None)
    assert len(saved) == 1  # Forced FINAL excluded.


def test_training_rejects_mixed_target_formats(tmp_path):
    from lsdrag.head_experiment import train_behavior_heads
    records = [{'qid': 'q', 'action': 'FINAL', 'index': 5, 'selector': 'native_single',
                'forced_final': False, 'format_valid': True,
                'feature_origin': 'student_pre_teacher_output', 'teacher_contract': {'revision': 'a'}}]
    with pytest.raises(ValueError, match='selected teacher format'):
        train_behavior_heads(records, [[1., 2.]], tmp_path, {}, TOOLS)
    with pytest.raises(ValueError, match='provenance'):
        train_behavior_heads(records, [[1., 2.]], tmp_path, {}, TOOLS,
            teacher_selector='native_single', teacher_contract={'revision': 'b'})
    result = train_behavior_heads(records, [[1., 2.]], tmp_path, {}, TOOLS,
        teacher_selector='native_single', teacher_contract={'revision': 'a'})
    assert not result['gate']['passed']


def test_source_recall_uses_modality_denominators_not_answer_f1():
    from lsdrag.modality_recall import source_recall
    chunks = {'c': {'document_id': 't', 'document_ids': ['t', 'alias']}}
    figures = {'f': {'document_id': 'i'}}
    row = {'trajectory': [{'tool_name': 'semantic_search', 'result': [{'chunk_id': 'c'}, {'figure_id': 'f'}]}],
           'read_chunk_ids': [], 'delivered_figure_ids': ['f']}
    gold = {'supporting_context': [{'doc_id': 't', 'doc_part': 'text'}, {'doc_id': 'i', 'doc_part': 'image'}]}
    result = source_recall(row, gold, chunks, figures)
    assert result['text']['first_search_source_recall_at_k']['1'] == 1
    assert result['image']['first_search_source_recall_at_k']['1'] == 0
    assert result['image']['first_search_source_recall_at_k']['3'] == 1
    assert result['image']['image_delivered_source_recall'] == 1
    assert result['table']['trajectory_retrieved_source_recall'] is None
    assert result['text']['explicit_read_source_recall'] == 0
