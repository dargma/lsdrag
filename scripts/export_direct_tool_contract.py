"""Export the exact full-tool prompt and feature contract without using a GPU."""
import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

from lsdrag.agent import AgentConfig
from lsdrag.direct_contract import build_feature_contract
from lsdrag.io import dump_json
from lsdrag.tool_registry import action_names, synchronize_prompt, tool_contract_sha256, tool_schemas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', type=Path)
    parser.add_argument('--image-delivery-policy', choices=['agent-read', 'search-linked'], default='search-linked')
    args = parser.parse_args()
    config = AgentConfig(tool_profile='multimodal_full', image_delivery_policy=args.image_delivery_policy)
    prompt_path = args.prompt or Path(__file__).resolve().parents[1]/'prompts'/(
        'search_linked.txt' if args.image_delivery_policy == 'search-linked' else 'visual.txt')
    tools = tool_schemas('G1', config.image_delivery_policy, config.tool_profile)
    prompt = synchronize_prompt(prompt_path.read_text(), tools, config.image_delivery_policy)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    dump_json(args.output, {'purpose': 'contract only; no trained head or measured policy',
        'arm': 'G1', 'config': asdict(config), 'actions': list(action_names(tools)),
        'tools': tools, 'system_prompt': prompt, 'system_prompt_sha256': prompt_sha,
        'tool_contract_sha256': tool_contract_sha256(tools),
        'feature_contract': build_feature_contract(args.model_path, config, tools, [prompt_sha])})
    print(args.output)


if __name__ == '__main__':
    main()
