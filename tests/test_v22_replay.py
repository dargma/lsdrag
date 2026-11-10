import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from lsdrag.chunk_pointer import ChunkPointer, pool_candidates
from lsdrag.decision_policy import PolicyHead, distribution
from lsdrag.fullft import replay_v22_step


def tiny_model():
    torch.manual_seed(9)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config).float().eval()


def test_pointer_and_tool_replay_uses_current_backbone_and_backpropagates():
    model = tiny_model()
    tool, pointer = PolicyHead(dim=16), ChunkPointer(16, 8)
    context = list(range(1, 15)); spans = [(1, 4), (6, 9)]
    legal = torch.ones(4, dtype=torch.bool)
    with torch.no_grad():
        collected = model.model(input_ids=torch.tensor([context]), use_cache=True).last_hidden_state[0]
        action_lp = distribution(tool(collected[-1][None, None], legal[None])[0], legal[None]).log_prob(torch.tensor([2]))
        _, old_ptr = pointer.select(collected[-1], pool_candidates(collected, spans), targets=[0])
    step = dict(capture={'context_ids': context, 'candidate_spans': spans},
                segment={'context_ids': context, 'tokens': [], 'kinds': [], 'old_logp': [], 'choices': {}},
                pack=torch.full((37, 16), float('nan')), legal=legal, action=2, forced=False,
                temperature=1., old_logp=float(action_lp),
                pointer_events=[{'choice': e['choice'], 'stop': e['stop'], 'old_logp': float(e['logp'])} for e in old_ptr])
    events, gap = replay_v22_step(model, step, tool, pointer)
    assert gap.abs().max() < 1e-4
    assert [e['kind'] for e in events] == ['tool', 'pointer', 'pointer']
    (-torch.stack([e['new'] for e in events]).sum()).backward()
    assert model.model.embed_tokens.weight.grad.abs().sum() > 0
    assert tool.trunk[0].weight.grad.abs().sum() > 0
    assert pointer.key.weight.grad.abs().sum() > 0
    assert pointer.stop.weight.grad.abs().sum() > 0


def test_native_cached_generation_matches_full_prefix_replay_fp32():
    model = tiny_model(); context = list(range(1, 12)); tokens = [5, 6, 7]
    kinds = ['branch', 'forced', 'free']; dt, pt = 2., 1.
    old = []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([context]), use_cache=True)
        for token, kind in zip(tokens, kinds):
            logits = out.logits[0, -1].float()
            if kind == 'branch':
                old.append(float((logits[[4, 5]]/dt).log_softmax(-1)[1]))
            elif kind == 'free':
                old.append(float((logits/pt).log_softmax(-1)[token]))
            else:
                old.append(0.)
            out = model(input_ids=torch.tensor([[token]]), past_key_values=out.past_key_values, use_cache=True)
    step = dict(capture={'context_ids': context, 'candidate_spans': []},
                segment={'context_ids': context, 'tokens': tokens, 'kinds': kinds, 'old_logp': old,
                         'choices': {0: [4, 5]}, 'decision_temperature': dt},
                pointer_events=[], temperature=dt, forced=False)
    events, gap = replay_v22_step(model, step, payload_temperature=pt)
    assert len(events) == 2 and gap.abs().max() <= 1e-4
    assert {e['kind'] for e in events} == {'free', 'native_tool'}
    (-torch.stack([e['new'] for e in events]).sum()).backward()
    assert model.lm_head.weight.grad.abs().sum() > 0
