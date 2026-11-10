"""Write one frozen v2.2 S4 manifest (collection, update or dev evaluation) for an arm and iteration.

Iteration k collection samples from the iteration k-1 checkpoint (k=1: base LLM + KD heads); the update
that follows produces checkpoint k. The KL reference is always the iteration-0 policy.
"""
import argparse
import json
from pathlib import Path

from common import ROOT, REPO, save, sha

SPECS = ROOT/'specs/v22/s4'
KD_HEAD = ROOT/'results/v22_kd_20261110/head.pt'
SEED = 20261110


def run_dir(tag, arm, kind, iteration):
    return ROOT/'results'/f'v22_s4_{tag}_{arm}_{kind}{iteration}'


def checkpoint(tag, arm, iteration, base):
    """(model path, head path or None) of the policy after `iteration` updates."""
    if iteration == 0:
        return base, (str(KD_HEAD) if arm == 'head' else None)
    out = run_dir(tag, arm, 'update', iteration)
    assert json.loads((out/'status.json').read_text())['phase'] == 'complete', out
    return str(out/'model'), (str(out/'head.pt') if arm == 'head' else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['collect', 'update', 'eval'])
    ap.add_argument('--arm', choices=['head', 'native'], required=True)
    ap.add_argument('--iteration', type=int, required=True, help='collect/update: 1..; eval: 0..')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--last', type=int, default=2, help='final iteration: its optimizer state is not written (disk)')
    ap.add_argument('--proofs', nargs='*', default=[], help='path:field for gate proof records whose field is true')
    args = ap.parse_args()
    pre = json.loads((ROOT/'specs/v22/preflight.json').read_text())
    ref = json.loads((ROOT/'specs/v22/s2_dev192.json').read_text())
    fit = pre['fit']
    qids = fit[:64] + fit[153:217] + fit[306:370]
    assert all(sum(q.startswith(h) for q in qids) == 64 for h in ('2hop', '3hop', '4hop'))
    sources = {name: sha(REPO/name) for name in ref['source_sha256']}
    commit = (REPO/'SOURCE_COMMIT').read_text().strip()
    common = dict(profile='v22', arm=args.arm, seed=SEED+1000003*args.iteration, source_commit=commit,
                  source_sha256=sources, decision_temperature=1.0, iteration=args.iteration)
    i = args.iteration
    if args.kind == 'collect':
        model, head = checkpoint(args.tag, args.arm, i-1, ref['model'])
        spec = dict(common, stage='training_collection', repeats=8, qids=qids, model=model, corpus=ref['corpus'],
                    answer_max_chars=60, output=str(run_dir(args.tag, args.arm, 'collect', i)),
                    max_seconds=6*3600, max_generated=12_000_000, max_inputs=480_000_000)
        if head:
            spec.update(head_checkpoint=head, head_sha256=sha(Path(head)))
    elif args.kind == 'update':
        col = run_dir(args.tag, args.arm, 'collect', i)
        ev = json.loads((col/'evidence.json').read_text())
        model, head = checkpoint(args.tag, args.arm, i-1, ref['model'])
        proofs = []
        for item in args.proofs:
            path, field = item.rsplit(':', 1)
            assert json.loads(Path(path).read_text())[field] is True, item
            proofs.append(dict(path=path, sha256=sha(Path(path)), field=field, value=True))
        spec = dict(common, stage='grpo_update', repeats=8, batch=16, update_passes=1, qids=qids,
                    collection_manifest=str(col/'manifest.json'), collection_manifest_sha256=sha(col/'manifest.json'),
                    model=model, reference_model=ref['model'], gate_proofs=proofs,
                    group_files=ev['group_files'], group_sha256=ev['group_sha256'],
                    train_support_ids=ev['train_support_ids'], process_reward=True, call_weight=.15, token_weight=.10,
                    minimum_free_vram_gib=75, disk_reserve_gib=8, save_resume=i < args.last, output=str(run_dir(args.tag, args.arm, 'update', i)))
        if head:
            spec.update(head_checkpoint=head, head_sha256=sha(Path(head)),
                        reference_head=str(KD_HEAD), reference_head_sha256=sha(KD_HEAD))
        if i > 1:
            resume = run_dir(args.tag, args.arm, 'update', i-1)/'resume.pt'
            spec.update(optimizer_resume=str(resume), optimizer_resume_sha256=sha(resume),
                        consume_optimizer_resume=True)
    else:
        model, head = checkpoint(args.tag, args.arm, i, ref['model'])
        name = f'{"H" if args.arm == "head" else "N"}RL{i}'
        policy = dict(kind='v22_pointer', checkpoint=head, sha256=sha(Path(head))) if head else dict(kind='native')
        spec = dict(ref, run_name=f'v22_s4_{args.tag}_{args.arm}_dev192_i{i}', model=model, policies={name: policy},
                    source_commit=commit, source_sha256=sources)
    path = SPECS/f'{args.tag}_{args.arm}_{args.kind}{i}.json'
    save(path, spec)
    print(path)


if __name__ == '__main__':
    main()
