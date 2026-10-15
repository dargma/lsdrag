"""Pinned feature provenance for direct-controller checkpoints."""
import hashlib
from pathlib import Path
import re


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_feature_contract(model_path, config, tools, prompt_sha256s):
    from .tool_registry import tool_contract_sha256
    prompts = sorted(set(prompt_sha256s))
    if not prompts or any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) for value in prompts):
        raise ValueError('Allowed system prompt SHA256 list must be nonempty and valid')
    path = Path(model_path).resolve()
    revision = path.name
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Direct head requires a pinned 40-hex model snapshot')
    names = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'processor_config.json', 'preprocessor_config.json', 'image_processor_config.json',
             'chat_template.json', 'chat_template.jinja', 'vocab.json', 'merges.txt', 'tokenizer.model'}
    assets = {file.relative_to(path).as_posix(): _sha(file) for file in path.rglob('*') if file.is_file() and
              (file.name in names or file.suffix == '.jinja' or file.name.startswith('chat_template'))}
    if 'config.json' not in assets or 'tokenizer_config.json' not in assets:
        raise ValueError('Missing pinned model/tokenizer configuration assets')
    return {'format': 'jev-direct-features-v1', 'model_revision': revision,
            'assets_sha256': assets, 'prepare_hf_reader_sha256': _sha(Path(__file__).with_name('hf_reader.py')),
            'readout': 'full-multimodal-backbone-final-norm-last-token-v1', 'backbone_dtype': 'bfloat16',
            'image_soft_tokens': int(config.image_soft_tokens), 'context_limit': int(config.context_limit),
            'actions': [tool['function']['name'] for tool in tools]+['FINAL'],
            'tool_contract_sha256': tool_contract_sha256(tools), 'allowed_system_prompt_sha256s': prompts}


def verify_system_prompts(messages, feature_contract):
    allowed = set(feature_contract['allowed_system_prompt_sha256s'])
    system = [message.get('content') for message in messages if message.get('role') == 'system']
    if not system or any(not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() not in allowed for text in system):
        raise ValueError('System prompt is not allowed by checkpoint feature contract')
