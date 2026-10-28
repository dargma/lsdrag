"""Shared paths and helpers. All runtime data lives under ROOT; the repo holds code and configs."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO/'configs/experiment.json').read_text())
SPLIT = json.loads((REPO/'configs/split.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def load_head(spec, device='cpu'):
    from lsdrag.decision_policy import PolicyHead
    head = PolicyHead()
    head.load_state_dict(torch.load(spec['checkpoint'], weights_only=True, map_location=device))
    return head.to(device).eval()
