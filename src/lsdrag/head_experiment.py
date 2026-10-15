"""Six-action behavior-cloning pilot with a shared argument realization path.

Generated letters are a controlled controller baseline, not native A-RAG tool
serialization. All natural teacher decisions, including failed episodes, count.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .direct_contract import build_feature_contract
from .direct_reader import DirectHFReader, DirectOutputError
from .hf_reader import HFReader
from .tool_registry import action_names, tool_contract_sha256

SEEDS = (20261015, 20261016, 20261017)
SELECTOR_INSTRUCTION = ('At each natural decision choose the next operation using ONLY one letter: '
    'A = keyword_search; B = semantic_search; C = read_chunk; D = image_search; '
    'E = figure_read; F = FINAL (answer now and stop). Do not generate arguments in the decision. '
    'The runtime will separately request argument JSON or the final answer after selection. '
    'When that subsequent request appears, follow it and return JSON instead of a letter.')


def experiment_prompt(base_prompt):
    return base_prompt.rstrip()+'\n\n'+SELECTOR_INSTRUCTION+'\n'


class GeneratedChoiceReader(DirectHFReader):
    """Shared frozen model; generated/first-token/learned selection, common JSON path."""
    def __init__(self, model_path, config, tools, feature_contract, mode='generated', state_sink=None):
        expected = build_feature_contract(model_path, config, tools, feature_contract.get('allowed_system_prompt_sha256s', []))
        if feature_contract != expected:
            raise ValueError('Generated controller feature contract mismatch')
        HFReader.__init__(self, model_path, config, head_path=None)
        self.actions = action_names(tools)
        if len(self.actions) != 6:
            raise ValueError('Full head experiment requires five tools plus FINAL')
        self.registry_tools = copy.deepcopy(tools)
        self.contract_sha256 = tool_contract_sha256(tools)
        self.expected_feature_contract = feature_contract
        self.mode, self.state_sink = mode, state_sink
        self.call_counter = 0
        self.current_qid, self.natural_loop = None, 0
        self._capture_owner = self
        self.label_ids = []
        for label in 'ABCDEF':
            ids = self.processor.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError('Action label must encode as one token')
            self.label_ids.append(ids[0])

    @classmethod
    def from_reader(cls, reader, mode, checkpoint=None, state_sink=None):
        obj = cls.__new__(cls)
        obj.__dict__ = reader.__dict__.copy()
        obj.mode, obj.state_sink = mode, state_sink
        obj.natural_loop = 0
        obj._capture_owner = reader._capture_owner
        if mode == 'head':
            saved = obj.torch.load(checkpoint, map_location='cpu', weights_only=True)
            obj._validate_checkpoint(saved)
            width = obj.model.config.get_text_config().hidden_size
            if saved['weight'].shape[1] != width:
                raise ValueError('Wrong backbone width')
            obj.head = obj.torch.nn.Linear(width, 6).to(obj.model.device).float()
            obj.head.load_state_dict({k: saved[k] for k in ('weight', 'bias')})
            obj.head.eval().requires_grad_(False)
        return obj

    def _select(self, messages, tools):
        if self.mode not in {'generated', 'first_token', 'head'}:
            raise ValueError(self.mode)
        torch = self.torch
        start = time.perf_counter()
        inputs = self.prepare(messages, tools)
        self._sync()
        preprocessing = time.perf_counter()-start
        budget = 8 if self.mode == 'generated' else 1
        if inputs['input_ids'].shape[1]+budget > self.config.context_limit:
            raise ValueError('Selector exceeds context budget')
        tick = time.perf_counter()
        generated, raw = [], None
        with torch.inference_mode():
            if self.mode == 'head':
                output = self.model.model(**inputs, use_cache=True, return_dict=True)
                feature = output.last_hidden_state[:, -1, :].detach().clone()
            else:
                output = self.model(**inputs, use_cache=True, logits_to_keep=1)
                feature = self._capture_owner.feature.detach().clone()
            self._sync()
            prefill = time.perf_counter()-tick
            tick = time.perf_counter()
            if self.mode == 'head':
                logits = self.head(feature.float())[0]
            else:
                logits = output.logits[0, -1, self.label_ids].float()
            probabilities = logits.softmax(-1).cpu().tolist()
            index = int(logits.argmax().item())
            self._sync()
            head_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            if self.mode == 'generated':
                mask = inputs['attention_mask']
                for _ in range(budget):
                    token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                    value = int(token.item())
                    generated.append(value)
                    if value in self.stop_ids:
                        break
                    mask = torch.cat([mask, torch.ones_like(token)], -1)
                    output = self.model(input_ids=token, attention_mask=mask, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
                raw = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
            self._sync()
            decode_seconds = time.perf_counter()-tick
        usage = {'preprocess_seconds': preprocessing, 'prefill_seconds': prefill,
                 'head_seconds': head_seconds, 'decode_seconds': decode_seconds,
                 'prompt_tokens': int(inputs['input_ids'].shape[1]),
                 'generated_tokens': len(generated), 'completion_tokens': len(generated),
                 'forced_prefix_tokens': 0, 'selector': self.mode, 'raw_selector': raw,
                 'wall_seconds': time.perf_counter()-start}
        if self.mode == 'generated':
            if raw not in tuple('ABCDEF') or not generated or generated[-1] not in self.stop_ids:
                raise DirectOutputError('Malformed or truncated generated six-way decision', usage)
            index = 'ABCDEF'.index(raw)
        feature_array = feature[0].float().cpu().numpy().copy()
        self.natural_loop += 1
        if self.state_sink:
            self.state_sink({'qid': self.current_qid, 'loop': self.natural_loop,
                             'action': self.actions[index], 'index': index,
                             'forced_final': False, 'selector': self.mode,
                             'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                             'usage': usage}, feature_array)
        return self.actions[index], probabilities, usage


def train_behavior_heads(records, features, outputdir, feature_contract, tools, *, seeds=SEEDS, epochs=100):
    import torch
    torch.set_num_threads(2)
    actions = action_names(tools)
    if len(actions) != 6:
        raise ValueError('Expected complete six-action registry')
    records = list(records)
    x = np.asarray(features, dtype=np.float32)
    if not records or x.ndim != 2 or len(x) != len(records) or not np.isfinite(x).all():
        raise ValueError('Invalid feature/record matrix')
    if any(not isinstance(r.get('qid'), str) or not r['qid'] for r in records):
        raise ValueError('Missing training question identity')
    if any('index' in r and r['index'] != actions.index(r['action']) for r in records):
        raise ValueError('Action label/index mismatch')
    if any(r.get('forced_final') or r.get('selector') != 'generated' for r in records):
        raise ValueError('BC labels require natural generated teacher decisions')
    y = np.asarray([actions.index(r['action']) for r in records], dtype=np.int64)
    coverage = {a: len({r['qid'] for r in records if r['action'] == a}) for a in actions}
    question_count = len({r['qid'] for r in records})
    gate = {'passed': question_count >= 30 and all(n >= 5 for n in coverage.values()),
            'distinct_teacher_questions': question_count, 'minimum_distinct_teacher_questions': 30, 'minimum_distinct_questions_per_action': 5,
            'distinct_questions_per_action': coverage, 'note': 'No forced or historical labels; failed episodes included.'}
    directory = Path(outputdir)
    directory.mkdir(parents=True, exist_ok=True)
    result = {'gate': gate, 'runs': [], 'examples': len(records), 'teacher_objective': 'behavior cloning, not outcome improvement',
              'actions': list(actions), 'epochs': int(epochs), 'seeds': list(seeds),
              'action_histogram': {action: sum(r['action'] == action for r in records) for action in actions},
              'question_coverage': coverage, 'teacher_episode_status_by_question': {r['qid']: r.get('episode_status', 'unknown') for r in records}}
    if not gate['passed']:
        (directory/'training.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.
    tx, ty = torch.from_numpy((x-mean)/scale), torch.from_numpy(y)
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = torch.nn.Linear(x.shape[1], 6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        curve = []
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(tx), ty)
            loss.backward()
            optimizer.step()
            curve.append(float(loss.detach()))
        # Fold train-only standardization into raw-feature Linear weights.
        weight = model.weight.detach()/torch.from_numpy(scale)[None, :]
        bias = model.bias.detach()-(weight*torch.from_numpy(mean)[None, :]).sum(-1)
        with torch.inference_mode():
            original_logits = model(tx)
            folded_logits = torch.from_numpy(x) @ weight.T + bias
        if not torch.allclose(original_logits, folded_logits, atol=1e-4, rtol=1e-4) or not torch.equal(original_logits.argmax(-1), folded_logits.argmax(-1)):
            raise ValueError('Folding train normalization changed logits/actions')
        checkpoint = {'format': 'jev-direct-head-v1', 'actions': actions,
                      'tool_contract_sha256': tool_contract_sha256(tools), 'feature_contract': feature_contract,
                      'weight': weight, 'bias': bias, 'seed': int(seed), 'epochs': int(epochs),
                      'purpose': 'six_action_behavior_cloning', 'train_qids': sorted({r['qid'] for r in records}),
                      'normalization': {'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale), 'folded_into_linear': True},
                      'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .01},
                      'labels': 'all natural generated decisions, including unsuccessful trajectories'}
        path = directory/f'head_seed_{seed}.pt'
        torch.save(checkpoint, path)
        result['runs'].append({'checkpoint': str(path), 'seed': int(seed), 'loss_curve': curve})
    (directory/'training.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def validate_teacher_artifact(artifact, qid, feature_contract):
    """Reject mismatched provenance, identities and state/feature alignment."""
    if artifact['feature_contract'] != feature_contract:
        raise ValueError('Teacher feature provenance mismatch')
    records, features = artifact['records'], artifact['features']
    if not isinstance(qid, str) or not qid or any(record['qid'] != qid for record in records):
        raise ValueError('Teacher record qid differs from selected training question')
    if len(records) != len(features) or features.ndim != 2:
        raise ValueError('Teacher record/features shape mismatch')
    if len({record['loop'] for record in records}) != len(records):
        raise ValueError('Duplicate teacher state loop')
    return records, features


def infrastructure_tool_errors(trajectory):
    """Tool transport/index failures must never become imitation-quality zeros."""
    failures = []
    policy_value_prefixes = ('Unknown chunk ID:', 'Unknown figure ID:', 'Provide 1–8 chunk IDs',
                            'Provide 1–8 figure IDs', 'keywords must be nonempty strings', 'Unknown tool')
    for step in trajectory:
        result = step.get('result')
        if not isinstance(result, dict) or 'error' not in result:
            continue
        error = str(result['error'])
        if error.startswith('ValidationError:'):
            continue
        if error.startswith('ValueError:') and any(error[len('ValueError:'):].strip().startswith(prefix) for prefix in policy_value_prefixes):
            continue
        failures.append(step)
    return failures


def validate_completed_prefix(completed_qids, train_qids):
    completed = set(completed_qids)
    allowed = [list(train_qids)[:count] for count in (60, 80, 100) if len(train_qids) >= count]
    if len(completed) != len(completed_qids) or not any(completed == set(prefix) for prefix in allowed):
        raise ValueError('Completed teacher identities must equal the fixed first 60, 80 or 100 training questions')
