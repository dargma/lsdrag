"""Verified wrapper around the pinned, unmodified official MultiModalQA scorer.

Answers are lists of distinct answer spans, not alternative aliases. Returns
fractions in [0, 1]. Source files live outside the repository and are checked
against pinned SHA256 values before loading.
"""
from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

REVISION = '4dd14328c6d02a4daa357cc6032915a0b14602e3'
DEFAULT_SOURCE_DIR = Path('/content/local_fast/lsdrag-v11/scorer')
_SHA256 = {
    'evaluate.py': '18622a55661bd6841d9cf6c8051da23bd0a21e69253e7a340cd6702340413660',
    'common_utils.py': '1f5eb00d529627aafe1ead9895419cd79b9a7f4e35b69571bd9cbb3ea637b0ff',
}


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load official scorer: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4)
def load_official_scorer(source_dir: str = str(DEFAULT_SOURCE_DIR)) -> ModuleType:
    directory = Path(source_dir)
    for filename, expected in _SHA256.items():
        path = directory / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Official MMQA scorer SHA256 mismatch: {path}')
    common = _module('_lsdrag_mmqa_official_common', directory / 'common_utils.py')
    # Upstream uses an absolute sibling import. Restore any unrelated module.
    previous = sys.modules.get('common_utils')
    sys.modules['common_utils'] = common
    try:
        return _module('_lsdrag_mmqa_official_evaluate', directory / 'evaluate.py')
    finally:
        if previous is None:
            sys.modules.pop('common_utils', None)
        else:
            sys.modules['common_utils'] = previous


def score(predanswers: Sequence[str], goldanswers: Sequence[str], *,
          source_dir: str | Path = DEFAULT_SOURCE_DIR) -> dict[str, float]:
    """Return official list-aware ``em`` and ``f1`` fractions for one question.

    Gold strings comprise one reference answer list. The caller must parse model
    output into spans; this wrapper does not split commas or normalize formatting.
    """
    if isinstance(predanswers, str) or isinstance(goldanswers, str):
        raise TypeError('Pass lists of answer spans, not raw strings')
    predicted, gold = list(predanswers), list(goldanswers)
    if not gold:
        raise ValueError('MMQA gold reference cannot be empty')
    if not all(isinstance(x, str) for x in predicted + gold):
        raise TypeError('Every answer span must be a string')
    official = load_official_scorer(str(Path(source_dir).resolve()))
    return {'em': float(official.list_em(predicted, gold)),
            'f1': float(official.list_f1(predicted, gold))}
