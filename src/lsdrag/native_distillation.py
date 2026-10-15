"""Native single-action teacher targets on pre-output student states.

Teacher arguments drive teacher rollouts. The learned student still uses its
own argument generator: that distribution shift must be measured, not hidden.
"""
import copy
import hashlib
import json
import time

from jsonschema import validate, ValidationError

from .direct_contract import verify_system_prompts
from .direct_reader import DirectOutputError
from .head_experiment import GeneratedChoiceReader, selection_messages
from .tool_registry import action_names, tool_contract_sha256

SINGLE_ACTION_INSTRUCTION = (
    "Choose the next operation using the available native tools. "
    "Issue exactly one tool call per turn, or return the final answer JSON. "
    "Do not issue parallel or multiple tool calls. Do not output action letters.")
FINAL_SCHEMA = {'type': 'object', 'properties': {
    'answers': {'type': 'array', 'items': {'type': 'string'}},
    'citations': {'type': 'array', 'items': {'type': 'string'}},
    'reasoning': {'type': 'string'}}, 'required': ['answers'],
    'additionalProperties': False}


def native_target(choice, tools):
    """Never map malformed, truncated or multi-action output to FINAL."""
    if choice.get('finish_reason') != 'stop':
        raise ValueError('teacher_output_truncated')
    message = choice['message']
    calls = message.get('tool_calls') or []
    if len(calls) > 1:
        raise ValueError('teacher_multiple_actions')
    if calls:
        call = calls[0]
        if call.get('type') != 'function' or not call.get('id'):
            raise ValueError('teacher_invalid_call_envelope')
        function = call['function']
        action = function['name']
        schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
        if action not in schemas:
            raise ValueError('teacher_unknown_action')
        arguments = function['arguments']
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        validate(arguments, schemas[action])
        return {'action': action, 'index': action_names(tools).index(action),
                'arguments': copy.deepcopy(arguments), 'format_valid': True}
    answer = json.loads(message.get('content') or '')
    validate(answer, FINAL_SCHEMA)
    return {'action': 'FINAL', 'index': action_names(tools).index('FINAL'),
            'arguments': None, 'format_valid': True}


class NativeTeacherStudentReader:
    """Teacher rollout with student-only feature capture and auditable targets."""
    def __init__(self, student, teacher, teacher_system_prompt, teacher_contract, *, verify_readout=True):
        self.student, self.teacher = student, teacher
        self.verify_readout = verify_readout
        self.readout_verified = False
        self.teacher_system_prompt = teacher_system_prompt
        self.teacher_contract = copy.deepcopy(teacher_contract)
        self.current_qid, self.natural_loop = None, 0
        self.state_sink = None
        self.torch, self.model = student.torch, student.model
        self.expected_feature_contract = student.expected_feature_contract

    def chat(self, messages, tools):
        start = time.perf_counter()
        verify_system_prompts(messages, self.expected_feature_contract)
        # Snapshot student input BEFORE asking the teacher. No teacher answer,
        # rationale or selected arguments may enter this state.
        student_messages = copy.deepcopy(messages)
        teacher_messages = copy.deepcopy(messages)
        for message in teacher_messages:
            if message['role'] == 'system':
                message['content'] = self.teacher_system_prompt
        captured = []
        student_usage = {}
        if tools:
            if tool_contract_sha256(tools) != self.student.contract_sha256:
                raise ValueError('Teacher/student tool registry mismatch')
            probe = GeneratedChoiceReader.from_reader(
                self.student, 'first_token',
                state_sink=lambda record, feature: captured.append((record, feature)))
            probe.current_qid = self.current_qid
            _, _, student_usage = probe._select(student_messages, tools)
            if len(captured) != 1:
                raise ValueError('Student state capture failed')
            if self.verify_readout and not self.readout_verified:
                check_start = time.perf_counter()
                inputs = self.student.prepare(selection_messages(student_messages), tools)
                with self.torch.inference_mode():
                    output = self.model.model(**inputs, use_cache=False, return_dict=True)
                    actual = output.last_hidden_state[0, -1, :].float().cpu()
                self.student._sync()
                expected = self.torch.from_numpy(captured[0][1])
                self.torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                student_usage = dict(student_usage)
                student_usage['prompt_tokens'] += int(inputs['input_ids'].shape[1])
                student_usage['prefill_seconds'] = student_usage.get('prefill_seconds', 0)+time.perf_counter()-check_start
                student_usage['readout_equivalence_verified'] = True
                self.readout_verified = True
                del inputs, output, actual, expected
        # The probe is a same-state first-token baseline, NOT the training label.
        choice, teacher_usage = self.teacher.chat(teacher_messages, tools)
        usage = {key: student_usage.get(key, 0)+teacher_usage.get(key, 0)
                 for key in ('prompt_tokens', 'completion_tokens', 'generated_tokens',
                             'prefill_seconds', 'decode_seconds', 'head_seconds')}
        usage.update(student_probe_usage=student_usage, teacher_usage=teacher_usage,
                     reader_wall_seconds=time.perf_counter()-start,
                     teacher_target_format='native_single', direct_dispatch=True)
        if not tools:
            # Forced FINAL is never a natural action target.
            return choice, usage
        self.natural_loop += 1
        try:
            target = native_target(choice, tools)
        except (ValueError, KeyError, TypeError, ValidationError) as error:
            usage.update(target_rejection=str(error), teacher_choice=choice)
            raise DirectOutputError('Rejected native teacher target: '+str(error), usage) from error
        probe_record, feature = captured[0]
        record = dict(probe_record, **target, loop=self.natural_loop,
                      selector='native_single', forced_final=False,
                      student_probe_action=probe_record['action'],
                      teacher_contract=self.teacher_contract,
                      teacher_messages_sha256=hashlib.sha256(json.dumps(
                          teacher_messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      feature_origin='student_pre_teacher_output', usage=usage)
        if self.state_sink:
            self.state_sink(record, feature.copy())
        return choice, usage
