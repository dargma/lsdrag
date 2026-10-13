"""Inspect case integrity using only the Python standard library; no downloads."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
for case in sorted(path for path in root.iterdir() if path.is_dir()):
    manifest = json.loads((case / 'manifest.json').read_text())
    for item in manifest['files']:
        path = case / item['path']
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
    metrics = {arm: json.loads((case / 'metrics' / (arm + '.json')).read_text()) for arm in ('B0', 'G1')}
    assert metrics['B0']['em'] == 0 and metrics['G1']['em'] == 1, case
    predictions = {arm: json.loads((case / 'predictions' / (arm + '.json')).read_text()) for arm in ('B0', 'C1', 'G1')}
    assert all(p['question_image_count'] == 0 for p in predictions.values()), case
    for figure in json.loads((case / 'figures_local.json').read_text()):
        assert (case / figure['image_path']).is_file(), figure
    counts = [len(predictions[a]['delivered_figure_ids']) for a in ('B0', 'G1')]
    print(f"{case.name}: hashes OK; Qwen EM 0 -> Google multimodal EM 1; images {counts[0]} -> {counts[1]}")
