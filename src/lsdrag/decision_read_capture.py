"""Capture causal document features without altering native actions or arguments."""
import hashlib
import json
from pathlib import Path
import torch
from .decision_policy_recovery import RecoveryPolicyReader
from .text_decision_benchmark import canonical


def candidate_spans(tokenizer, ids, history):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    if encoded['input_ids'] != ids:
        raise ValueError('Unsafe tokenizer roundtrip')
    records = []
    seen = set()
    for step in reversed(history):
        result = step['result']
        if not isinstance(result, list):
            continue
        content = canonical(result)
        base = text.rfind(content)
        if base < 0:
            raise ValueError('Tool observation absent from prefix')
        for row in result:
            if not isinstance(row, dict) or not isinstance(row.get('chunk_id'), str):
                continue
            cid = row['chunk_id']
            if cid in seen:
                continue
            seen.add(cid)
            serialized = canonical(row)
            offset = content.find(serialized)
            if offset < 0:
                raise ValueError('Candidate not in observation')
            start = base + offset; end = start + len(serialized)
            positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > start and a < end]
            if not positions:
                raise ValueError('Empty candidate span')
            records.append(dict(chunk_id=cid, span=[positions[0], positions[-1]+1], text_sha256=hashlib.sha256(serialized.encode()).hexdigest()))
    return records


class ReadCaptureReader(RecoveryPolicyReader):
    output_dir = None
    question_ids = None

    def capture_pack(self, messages, history, deadline):
        captured = []
        handle = self.model.model.register_forward_hook(lambda module, args, output: captured.append(output.last_hidden_state[0].detach()))
        try:
            pack, usage = super().capture_pack(messages, history, deadline)
        finally:
            handle.remove()
        assert len(captured) == 1
        self._capture_hidden = captured[0]
        self._capture_pack = pack.detach()
        self._capture_question = next(m['content'] for m in messages if m['role']=='user')
        self._capture_prefix = usage['messages_sha256']
        return pack, usage

    def continue_cached(self, action, trajectory, deadline=float('inf')):
        record = None
        if action == 'read_chunk':
            candidates = candidate_spans(self.tokenizer, self.pending['ids'], trajectory)
            assert candidates
            h = self._capture_hidden
            features = torch.stack([h[a:b].float().mean(0) for a,b in (r['span'] for r in candidates)]).cpu()
            record = dict(qid=self.question_ids[self._capture_question], turn=len(trajectory), prefix_sha256=self._capture_prefix,
                          candidates=candidates, candidate_features=features, pack=self._capture_pack.float().cpu())
        self._capture_hidden = None
        self._capture_pack = None
        value, usage, error = super().continue_cached(action, trajectory, deadline)
        if record is not None:
            record.update(arguments=value, error=error, generated_tokens=usage['generated_tokens'])
            dest = Path(self.output_dir); dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"{record['qid']}_{record['turn']}.pt"
            if path.exists(): raise FileExistsError(path)
            torch.save(record, path)
        return value, usage, error
