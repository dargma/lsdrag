"""Train-only JSON and cached-continuation functional check, not a speed claim."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from lsdrag.text_integrated import StrictReader, argument_schema, native_schema, options_for
from lsdrag.text_decision_benchmark import public_paragraphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, default=Path('prompts/text_rag_integrated_v4.txt'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    qid = json.loads(Path('lab/registry/text_integrated_20261016/split.json').read_text())['train'][0]
    with Path('/content/local_fast/lsdrag-text-v1/source/musique/musique_ans_v1.0_train.jsonl').open() as stream:
        row = next(r for line in stream if (r := json.loads(line))['id'] == qid)
    paragraph = public_paragraphs(row)[0]
    history = [{'tool_name': 'semantic_search', 'arguments': {'query': row['question']},
                'result': [{'chunk_id': paragraph['chunk_id'], 'title': paragraph['title'],
                            'snippet': paragraph['text'][:600]}]}]
    messages = [{'role': 'system', 'content': args.prompt.read_text()},
                {'role': 'user', 'content': row['question']}]
    state = messages + [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'check', 'type': 'function',
            'function': {'name': 'semantic_search', 'arguments': json.dumps(history[0]['arguments'])}}]},
        {'role': 'tool', 'tool_call_id': 'check', 'content': json.dumps(history[0]['result'])}]
    reader = StrictReader('/content/local_fast/lsdrag-text-v1/hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554',
                          SimpleNamespace(max_output_tokens=512, context_limit=16384))
    checks = []
    deadline = time.perf_counter() + 600
    def save(kind, action, value, usage, failure):
        checks.append(dict(kind=kind, action=action, value=value, usage=usage, failure=failure))
        (args.output / 'checks.json').write_text(json.dumps({'qid': qid, 'checks': checks,
            'passed': all(c['failure'] is None for c in checks), 'quality_or_speed_evidence': False}, indent=2))
        print(kind, action, failure or 'PASS', flush=True)
        if failure:
            raise RuntimeError(failure)
    for action in ['keyword_search', 'semantic_search']:
        value, usage, failure = reader.generate_schema(messages, argument_schema(action, []), 'arguments',
            'Generate arguments for '+action, deadline)
        save('fresh_argument', action, value, usage, failure)
    value, usage, failure = reader.generate_native(messages, [], deadline)
    save('native', value.get('action') if value else None, value, usage, failure)
    for action in ['keyword_search', 'semantic_search', 'read_chunk', 'FINAL']:
        _, _, capture = reader.capture(state, options_for(row['question'], history), 'teacher', deadline)
        value, usage, failure = reader.continue_cached(action, history, deadline)
        usage['capture'] = capture
        assert usage['full_prefill_count'] == 0 and usage['cache_reused']
        save('cached', action, value, usage, failure)


if __name__ == '__main__':
    main()
