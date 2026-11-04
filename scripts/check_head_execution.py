"""Check that head-ARAG executes exactly the head's action and that the prefilled/generated tool call
matches the context the next step sees.

CPU part (saved episodes): action == executed tool, forced prefix decodes to the tool-call syntax for that
action, generated tokens never contain a tool name, and the next step's rendered context contains the
tool call exactly as it was prefilled+generated (otherwise the LLM conditions on text it did not produce).

Usage: python scripts/check_head_execution.py --run RUN [--limit N]
"""
import argparse, collections, glob, json
from pathlib import Path
from common import ROOT, REPO, CONFIG
from lsdrag.text_decision_benchmark import TOOLS, canonical
from lsdrag.text_integrated import ACTIONS
from lsdrag.text_integrated_aligned import syntax_prefix

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--limit', type=int, default=200); a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CONFIG['model'])
    raw = {}
    for name in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
            q = json.loads(line); raw[q['id']] = q
    prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
    render = lambda msgs: tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    C = collections.Counter(); examples = []
    files = sorted(glob.glob(str(ROOT/'results'/a.run/'episodes'/'*.json')))[:a.limit]
    for p in files:
        e = json.loads(Path(p).read_text()); C['episodes'] += 1
        conts = [u for u in e['reader_calls'] if 'raw_json' in u]   # one continuation per decision
        dec = e['decisions']; hist = e['trajectory']
        executed = [h['tool_name'] for h in hist] + (['FINAL'] if not e['errors'] else [])
        C['decisions'] += len(dec)
        C['action_eq_executed'] += sum(ACTIONS[d['action']] == x for d, x in zip(dec, executed))
        for d, u in zip(dec, conts):
            name = ACTIONS[d['action']]; pre = tok.decode(u['prefix_token_ids'])
            C['prefix_ok'] += pre == syntax_prefix(name)[0]
            gen = tok.decode(u['token_ids'])
            C['gen_has_toolname'] += any(f'"name": "{t}"' in gen for t in ACTIONS)
        # context consistency: rendered next-step prompt vs what was actually prefilled+generated
        q = raw[e['qid']]['question']
        msgs = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': q}]
        for i, h in enumerate(hist):
            before = render(msgs)
            produced = tok.decode(conts[i]['prefix_token_ids'] + conts[i]['token_ids'])
            tc = {'id': f'call_{i}', 'type': 'function', 'function': {'name': h['tool_name'], 'arguments': canonical(h['arguments'])}}
            msgs += [{'role': 'assistant', 'content': None, 'tool_calls': [tc]}, {'role': 'tool', 'tool_call_id': tc['id'], 'content': canonical(h['result'])}]
            after = render(msgs)
            C['steps'] += 1
            C['prefix_preserved'] += after.startswith(before)
            seen = after[len(before):] if after.startswith(before) else ''
            ok = seen.startswith(produced.rstrip('\n'))
            C['context_matches_produced'] += ok
            if not ok and len(examples) < 3: examples.append({'produced': produced, 'next_context': seen[:len(produced)+40]})
    print(json.dumps({'counts': C, 'mismatch_examples': examples}, indent=1, ensure_ascii=False))

if __name__ == '__main__':
    main()
