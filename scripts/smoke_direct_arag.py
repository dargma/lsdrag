"""Real-backbone functional dispatch smoke. Synthetic head/index, no quality/timing claim."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from lsdrag.agent import Agent, AgentConfig
from lsdrag.direct_contract import build_feature_contract
from lsdrag.direct_reader import DirectHFReader
from lsdrag.io import dump_json, sha256
from lsdrag.retrieval import Retrieval
from lsdrag.tool_registry import action_names, synchronize_prompt, tool_contract_sha256, tool_schemas


class FixedEmbedder:
    def encode(self, inputs, role):
        return np.array([[1., 0.] for _ in inputs], dtype=np.float32)


class FixedIndex:
    def __init__(self, chunk_ids, figure_ids=()):
        self.chunk_ids, self.figure_ids = chunk_ids, figure_ids
        self.last_matches = {}

    def search(self, query_vector, limit, identity='chunk_id', allowed=None, row_limit=None):
        ids = self.figure_ids if identity == 'figure_id' else self.chunk_ids
        return [(key, 1.-i*.1) for i, key in enumerate(ids) if allowed is None or key in allowed][:limit]


def synthetic_retrieval(directory):
    directory.mkdir(parents=True, exist_ok=True)
    retrieval = Retrieval.__new__(Retrieval)
    retrieval.arm, retrieval.scoped = 'G1', False
    retrieval.chunks = {}
    retrieval.figures = {}
    for number, color in enumerate(('red', 'blue'), 1):
        path = directory/f'figure{number}.png'
        Image.new('RGB', (224, 224), color).save(path)
        chunk_id, figure_id = f'c{number}', f'f{number}'
        retrieval.chunks[chunk_id] = {'chunk_id': chunk_id, 'document_id': 'doc1', 'document_ids': ['doc1'],
            'page_index': number, 'kind': 'image-caption', 'text': 'Sample shape illustration; inspect pixels for its color.', 'figure_ids': [figure_id]}
        retrieval.figures[figure_id] = {'figure_id': figure_id, 'chunk_id': chunk_id,
            'document_id': 'doc1', 'document_ids': ['doc1'], 'image_path': str(path), 'image_sha256': sha256(path)}
    retrieval.chunks['c0'] = {'chunk_id': 'c0', 'document_id': 'doc1', 'document_ids': ['doc1'], 'page_index': 0,
        'kind': 'text', 'text': 'Sample note: inspect the two illustrations. Available text chunk c0 and figures f1, f2.', 'figure_ids': []}
    retrieval.indexes = {'gemma_text': FixedIndex(['c1']), 'gemma_image': FixedIndex(['c1'], ['f2'])}
    retrieval.embedders = {'gemma': FixedEmbedder()}
    retrieval.query_cache, retrieval.candidate_count = {}, 2
    return retrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-free-gb', type=float, default=32.)
    args = parser.parse_args()
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    free_gb = float(free.splitlines()[0])/1024.
    if free_gb < args.min_free_gb:
        raise RuntimeError(f'Insufficient free VRAM: {free_gb:.1f} GiB')
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    config = AgentConfig(max_loops=8, max_output_tokens=128, context_limit=32768,
                         image_soft_tokens=1120, image_delivery_policy='search-linked', tool_profile='multimodal_full')
    tools = tool_schemas('G1', config.image_delivery_policy, config.tool_profile)
    actions = action_names(tools)
    prompt = ('You are running a synthetic interface test. Use discovered IDs exactly. '
              'Available text chunk c0; figure IDs f1 and f2. Return valid arguments for each selected operation. '
              'When the runtime asks for a final answer, return JSON with answers and optionally citations/reasoning.')
    effective = synchronize_prompt(prompt, tools, config.image_delivery_policy)
    prompt_hash = hashlib.sha256(effective.encode()).hexdigest()
    feature_contract = build_feature_contract(args.model_path, config, tools, [prompt_hash])
    import json
    model_config = json.loads((args.model_path/'config.json').read_text())
    width = model_config.get('text_config', model_config)['hidden_size']
    checkpoint = {'format': 'jev-direct-head-v1', 'purpose': 'synthetic_dispatch_smoke',
                  'actions': actions, 'tool_contract_sha256': tool_contract_sha256(tools),
                  'feature_contract': feature_contract, 'weight': torch.zeros(len(actions), width), 'bias': torch.zeros(len(actions))}
    checkpoint_path = args.output/'SYNTHETIC_NOT_TRAINED.pt'
    torch.save(checkpoint, checkpoint_path)
    reader = DirectHFReader(str(args.model_path), config, checkpoint_path, tools, diagnostic=True)
    class ScriptedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.position = 0
        def forward(self, hidden):
            logits = torch.full((hidden.shape[0], len(actions)), -30., device=hidden.device)
            logits[:, min(self.position, len(actions)-1)] = 30.
            self.position += 1
            return logits
    reader.head = ScriptedHead()
    retrieval = synthetic_retrieval(args.output/'synthetic_corpus')
    result = Agent(retrieval, prompt, config, reader).run('synthetic_dispatch_only',
        'Inspect the sample note and both shape illustrations. Search using the keyword sample; read c0 when asked for chunk IDs. '
        'For image search request one result; inspect the returned figure f2 when asked for figure IDs. What colors are shown?')
    observed = [row['tool_name'] for row in result['trajectory']]
    handler_errors = [row for row in result['trajectory'] if isinstance(row.get('result'), dict) and 'error' in row['result']]
    summary = {'purpose': 'synthetic_dispatch_smoke', 'learned_head': False, 'retrieval_quality_evaluated': False,
        'latency_comparison_valid': False, 'shared_gpu_execution_authorized': True,
        'expected_actions': list(actions), 'observed_tools': observed, 'free_vram_before_gb': free_gb,
        'all_five_handlers_executed': observed == list(actions[:-1]), 'handler_errors': handler_errors,
        'answer_format_valid': result['answer_format_valid'], 'reader_errors': result['errors'],
        'delivered_figure_ids': result['delivered_figure_ids'], 'result': result,
        'feature_contract': feature_contract,
        'source_hashes': {name: sha256(Path(__file__).resolve().parents[1]/name) for name in
                          ['scripts/smoke_direct_arag.py', 'src/lsdrag/direct_reader.py', 'src/lsdrag/direct_contract.py', 'src/lsdrag/agent.py', 'src/lsdrag/tool_registry.py']}}
    summary['passed'] = (summary['all_five_handlers_executed'] and not handler_errors and not result['errors']
                         and result['answer_format_valid'] and result['status'] == 'ok'
                         and set(result['delivered_figure_ids']) == {'f1', 'f2'})
    dump_json(args.output/'functional_smoke.json', summary)
    print(json.dumps({key: summary[key] for key in ('passed', 'all_five_handlers_executed', 'answer_format_valid', 'delivered_figure_ids', 'reader_errors')}))
    if not summary['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
