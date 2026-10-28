"""Bounded text-only RAG comparison; no historical labels or image transport."""
import hashlib
import json
from pathlib import Path

import numpy as np
from jsonschema import validate

# Tool schemas given to the chat template (frozen copy of the original A-RAG text tools).
TOOLS = json.loads((Path(__file__).with_name('text_tools.json')).read_text())
FINAL_SCHEMA = {'type': 'object', 'properties': {
    'answer': {'type': 'string'},
    'support_ids': {'type': 'array', 'items': {'type': 'string'}}},
    'required': ['answer', 'support_ids'], 'additionalProperties': False}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def select_questions(rows, count, seed=20261016):
    eligible = [r for r in rows if r['answerable'] and len(r['question_decomposition']) == 2]
    return sorted(eligible, key=lambda r: hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest())[:count]


def public_paragraphs(row):
    # Explicit allowlist: no is_supporting, answers or decomposition in runtime.
    return [{'chunk_id': f'p{p["idx"]}', 'title': p['title'], 'text': p['paragraph_text']}
            for p in row['paragraphs']]


class TextCorpus:
    def __init__(self, rows, vectors, embedder):
        self.rows, self.vectors, self.embedder = rows, vectors, embedder
        self.by_id = {r['chunk_id']: r for r in rows}
        self.query_cache = {}

    def call(self, name, args):
        schema = next(t['function']['parameters'] for t in TOOLS if t['function']['name'] == name)
        validate(args, schema)
        if name == 'read_chunk':
            if any(k not in self.by_id for k in args['chunk_ids']):
                raise ValueError('Unknown chunk ID')
            return [dict(self.by_id[k]) for k in args['chunk_ids']]
        if name == 'keyword_search':
            terms = [s.lower() for s in args['keywords']]
            scores = [sum(r['text'].lower().count(t) * len(t) for t in terms) for r in self.rows]
        else:
            query = args['query']
            if query not in self.query_cache:
                self.query_cache[query] = self.embedder.encode([query], 'query')[0]
            scores = self.vectors @ self.query_cache[query]
        order = sorted(range(len(self.rows)), key=lambda i: (-float(scores[i]), self.rows[i]['chunk_id']))
        return [{'chunk_id': self.rows[i]['chunk_id'], 'title': self.rows[i]['title'],
                 'snippet': self.rows[i]['text'][:300]}
                for i in order[:args.get('top_k', 5)] if name != 'keyword_search' or scores[i] > 0]


def execution_options(question, trajectory):
    """One D3 choice jointly selects action and complete reusable args."""
    options = []
    observed, groups = [], []
    for step in reversed(trajectory):
        result = step['result']
        ids = [r['chunk_id'] for r in result if isinstance(r, dict) and isinstance(r.get('chunk_id'), str)] if isinstance(result, list) else []
        if ids:
            groups.append(list(dict.fromkeys(ids))[:8])
            observed.extend(ids)
    observed = list(dict.fromkeys(observed))
    for tool in TOOLS:
        name = tool['function']['name']
        options.append({'action': name, 'arguments': None, 'source': 'GENERATE_ARGS'})
        candidates = []
        if name == 'read_chunk':
            candidates = [{'chunk_ids': ids} for ids in groups] + [{'chunk_ids': [k]} for k in observed]
        else:
            candidates = [dict(step['arguments']) for step in reversed(trajectory) if step['tool_name'] == name]
            if name == 'semantic_search':
                candidates.append({'query': question, 'top_k': 5})
        seen = set()
        for value in candidates:
            identity = canonical(value)
            if identity in seen:
                continue
            validate(value, tool['function']['parameters'])
            seen.add(identity)
            options.append({'action': name, 'arguments': value, 'source': 'observed_or_original_question'})
            if len(seen) == 4:
                break
    options.append({'action': 'FINAL', 'arguments': None, 'source': 'answer_generation'})
    return options
