"""Predeclared CPU-only matched-capacity decision-head training.

Only training rows determine normalization or gradient updates. The auxiliary
regression is enabled at S1, where access effects were measured.
"""
import json
from pathlib import Path

import numpy as np

ACTIONS = ('ANSWER_AND_STOP', 'TEXT_ACQUIRE', 'IMAGE_ACQUIRE')
DEFAULT_SEEDS = (20261015, 20261016, 20261017)


def training_arrays(rows, cost_lambda):
    features, targets, auxiliaries, masks = [], [], [], []
    seen = set()
    for row in rows:
        if row['qid'] in seen:
            raise ValueError('Duplicate training qid')
        seen.add(row['qid'])
        if len(row['features']) != 2 or len(row['stops']) != 2:
            raise ValueError('Each row requires S0 and S1')
        aux = [float(row['actions'][action][name]) for action in ACTIONS[1:] for name in ('N0', 'J')]
        for state in (0, 1):
            stop = row['stops'][state]
            values = [float(stop['f1']) - cost_lambda * float(stop['cost_seconds'])]
            for action in ACTIONS[1:]:
                cell = row['actions'][action]['cells']['00' if state == 0 else '11']
                values.append(float(cell['f1']) - cost_lambda * float(cell['cost_seconds']))
            features.append(row['features'][state])
            targets.append(values)
            auxiliaries.append(aux)
            masks.append(float(state == 1))
    if not features:
        raise ValueError('No training examples')
    arrays = [np.asarray(value, dtype=np.float32) for value in (features, targets, auxiliaries, masks)]
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError('Nonfinite training data')
    if arrays[0].ndim != 2:
        raise ValueError('Features must form a matrix')
    return tuple(arrays)


def _model(dimension):
    import torch

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Sequential(torch.nn.Linear(dimension, 128), torch.nn.ReLU())
            self.q = torch.nn.Linear(128, 3)
            self.access = torch.nn.Linear(128, 4)

        def forward(self, x):
            hidden = self.shared(x)
            return self.q(hidden), self.access(hidden)

    return Head()


def train_heads(rows, outputdir, lambdas=(0., .01), seeds=DEFAULT_SEEDS, epochs=200):
    import torch
    torch.set_num_threads(2)
    rows = list(rows)
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    runs = []
    for cost_lambda in lambdas:
        if cost_lambda < 0:
            raise ValueError('Negative cost coefficient')
        x, y, aux, mask = training_arrays(rows, float(cost_lambda))
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-6] = 1.
        tx = torch.from_numpy((x-mean)/scale)
        ty, ta, tm = map(torch.from_numpy, (y, aux, mask))
        for seed in seeds:
            for beta in (0., 1.):
                torch.manual_seed(int(seed))
                model = _model(x.shape[1]).cpu()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
                curve = []
                for epoch in range(int(epochs)):
                    optimizer.zero_grad(set_to_none=True)
                    q, access = model(tx)
                    return_loss = torch.nn.functional.huber_loss(q, ty)
                    per_state = torch.nn.functional.huber_loss(access, ta, reduction='none').mean(-1)
                    access_loss = (per_state*tm).sum()/tm.sum().clamp_min(1.)
                    loss = return_loss + beta*access_loss
                    loss.backward()
                    optimizer.step()
                    curve.append({'epoch': epoch+1, 'loss': float(loss.detach()), 'return_loss': float(return_loss.detach()), 'access_loss': float(access_loss.detach())})
                model.eval()
                with torch.inference_mode():
                    predicted = model(tx)[0].argmax(-1).numpy()
                stem = f'lambda_{cost_lambda:g}_beta_{beta:g}_seed_{seed}'
                path = outputdir/f'{stem}.pt'
                checkpoint = {'format': 'jev-v11-head-v1', 'dimension': x.shape[1], 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale), 'state_dict': model.state_dict(), 'actions': ACTIONS, 'lambda': float(cost_lambda), 'beta': beta, 'seed': int(seed), 'epochs': int(epochs), 'train_qids': [r['qid'] for r in rows], 'train_source_groups': sorted({r['source_group'] for r in rows}), 'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .01}, 'architecture': 'Linear(d,128)-ReLU; Q3 and access4', 'auxiliary_state': 'S1_only'}
                torch.save(checkpoint, path)
                metadata = {key: checkpoint[key] for key in ('lambda', 'beta', 'seed', 'epochs', 'dimension', 'architecture', 'optimizer', 'auxiliary_state')}
                metadata.update(checkpoint=str(path), examples=len(x), target_action_distribution=np.bincount(y.argmax(-1), minlength=3).tolist(), predicted_action_distribution=np.bincount(predicted, minlength=3).tolist(), loss_curve=curve)
                (outputdir/f'{stem}.json').write_text(json.dumps(metadata, indent=2)+'\n')
                runs.append(metadata)
    result = {'runs': runs, 'training_questions': len(rows), 'actions': list(ACTIONS), 'normalization': 'training-only per-feature mean/std', 'hyperparameter_selection': 'none', 'target_definition': 'S0=stop,00 text,00 image; S1=stop,11 text,11 image; subtract lambda*cost_seconds'}
    (outputdir/'training_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def load_head(checkpoint):
    import torch
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['format'] != 'jev-v11-head-v1' or tuple(checkpoint['actions']) != ACTIONS:
        raise ValueError('Incompatible decision-head checkpoint')
    model = _model(int(checkpoint['dimension'])).cpu().eval()
    model.load_state_dict(checkpoint['state_dict'])
    model.requires_grad_(False)
    return {'model': model, 'mean': checkpoint['mean'], 'scale': checkpoint['scale'], 'checkpoint': checkpoint}


def infer_head(checkpoint, features):
    import torch
    loaded = checkpoint if isinstance(checkpoint, dict) and 'model' in checkpoint else load_head(checkpoint)
    x = np.asarray(features, dtype=np.float32)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    if x.ndim != 2 or x.shape[-1] != len(loaded['mean']) or not np.isfinite(x).all():
        raise ValueError('Invalid inference features')
    with torch.inference_mode():
        tx = (torch.from_numpy(x)-loaded['mean'])/loaded['scale']
        result = loaded['model'](tx)[0].numpy()
    return result[0] if single else result
