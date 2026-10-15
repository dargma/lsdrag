"""Run first2 then first20 native-target audit; never auto-train on this audit."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from lsdrag.io import read_jsonl
from lsdrag.modality_recall import source_recall
from lsdrag.evidence_metrics import visual_evidence_scores
from lsdrag.head_experiment import validate_teacher_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-args', type=Path, required=True)
    args = parser.parse_args()
    common = json.loads(args.common_args.read_text())
    value = lambda name: Path(common[common.index('--'+name)+1])
    root, repo = value('output'), Path(__file__).resolve().parents[1]
    backup = value('backup')
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root/'pilot.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    gold = {row['qid']: row for row in read_jsonl(value('train-questions').parent/'train_gold_private.jsonl')}
    chunks = {row['chunk_id']: row for row in read_jsonl(value('corpus')/'chunks.jsonl')}
    figures = {row['figure_id']: row for row in read_jsonl(value('corpus')/'figures.jsonl')}
    # Different accepted sets are unresolved, not silently unioned or called wrong.
    reviews, conflicts = {}, set()
    for name in ['development_alignment_review.json', 'additional50_alignment_review.json', 'final_alignment_partial_review.json']:
        for review in json.loads((repo/'lab/reports'/name).read_text()):
            identity = review['gold_image_id']
            if review.get('status') != 'verified_same_visual_content' or not review.get('accepted_figure_ids'):
                continue
            if identity in reviews and set(reviews[identity]['accepted_figure_ids']) != set(review['accepted_figure_ids']):
                conflicts.add(identity)
            reviews[identity] = review
    for identity in conflicts:
        reviews.pop(identity, None)
    def audit(limit):
        manifest = json.loads((root/'manifest.json').read_text())
        episodes, records, modality = [], [], []
        for qid in manifest['train_question_order'][:limit]:
            artifact = torch.load(root/'teacher'/f'{qid}.pt', map_location='cpu', weights_only=True)
            states, _ = validate_teacher_artifact(artifact, qid, manifest['feature_contract'])
            for record in states:
                if record.get('selector') != 'native_single' or record.get('teacher_contract') != manifest['teacher_contract'] or record.get('feature_origin') != 'student_pre_teacher_output' or not record.get('format_valid'):
                    raise ValueError('Target provenance mismatch')
            episode = json.loads((root/'teacher'/f'{qid}.json').read_text())
            if episode['qid'] != qid:
                raise ValueError('Episode identity mismatch')
            episodes.append(episode)
            records.extend(states)
            modality.append({'qid': qid, 'source_recall_proxies': source_recall(episode, gold[qid], chunks, figures),
                             'reviewed_visual_evidence': visual_evidence_scores(episode, gold[qid], chunks, figures, reviews)})
        rejections = [call['usage']['target_rejection'] for row in episodes for call in row['reader_calls'] if call['usage'].get('target_rejection')]
        attempted = len(records)+len(rejections)
        return {'questions': len(episodes), 'accepted_states': len(records),
            'target_rejections': rejections, 'target_acceptance_rate': len(records)/attempted if attempted else None,
            'action_counts': dict(Counter(r['action'] for r in records)),
            'native_vs_first_token_agreement_on_accepted_states': sum(r['action'] == r['student_probe_action'] for r in records)/len(records) if records else None,
            'em': sum(r['em'] for r in episodes)/len(episodes),
            'f1': sum(r['f1'] for r in episodes)/len(episodes),
            'answer_format_valid_count': sum(r['answer_format_valid'] for r in episodes),
            'reader_errors': [e for row in episodes for e in row['errors']],
            'per_question_modality_recall': modality,
            'unresolved_visual_mapping_ids': sorted(conflicts),
            'visual_mapping_complete_questions': sum(row['reviewed_visual_evidence']['visual_mapping_complete'] for row in modality),
            'modality_summary': {kind: {
                'eligible_questions': sum(row['source_recall_proxies'][kind]['required_source_count'] > 0 for row in modality),
                'trajectory_retrieved_source_macro_recall': (
                    sum(row['source_recall_proxies'][kind]['trajectory_retrieved_source_recall'] or 0 for row in modality)/
                    sum(row['source_recall_proxies'][kind]['required_source_count'] > 0 for row in modality)
                    if any(row['source_recall_proxies'][kind]['required_source_count'] for row in modality) else None)
            } for kind in ('text', 'table', 'image')},
            'recall_scope': 'Source-document proxies, first search ranked results and whole trajectory separately. NOT exact text spans, tables or images. Search snippets may contain evidence without read calls.',
            'quality_claim': 'Shared-backbone native single-action teacher audit; not an isolated model-size comparison or learned-head quality claim.',
            'training_completed': False, 'evaluation_completed': False}
    def checkpoint(tag, report):
        (root/(tag+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        reportdir = repo/'lab/reports/native_distillation_20261015'
        reportdir.mkdir(parents=True, exist_ok=True)
        (reportdir/(tag+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        backup.mkdir(parents=True, exist_ok=True)
        archive = root.parent/(tag+'.tar.gz')
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(root, arcname=root.name)
            for name in ['docs/JEV_DISTILLATION_PROTOCOL_20261015.md', 'lab/registry/native_distillation_20261015']:
                tar.add(repo/name, arcname=name)
        target = backup/archive.name
        temporary = target.with_suffix('.tmp')
        shutil.copy2(archive, temporary)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
            raise ValueError('Drive checkpoint hash mismatch')
        temporary.replace(target)
        status = {'phase': tag, 'run': str(root), 'archive': str(target), 'sha256': digest,
                  'updated_epoch': time.time(), 'pid': os.getpid(),
                  'next': ('Monitor running collection' if tag.endswith('_starting') else 'native20 audit' if tag == 'native2_complete' else 'Review targets and reader-size separation before teacher60/head training'),
                  'training_completed': False, 'evaluation_completed': False}
        (root/'checkpoint.json').write_text(json.dumps(status, indent=2)+'\n')
        (backup/(tag+'_provenance.json')).write_text(json.dumps(status, indent=2)+'\n')
        # Avoid committing unrelated in-progress work from another conversation.
        if subprocess.run(['git', 'diff', '--quiet'], cwd=repo).returncode or subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=repo).returncode:
            (root/'git_checkpoint_deferred.json').write_text(json.dumps(status, indent=2)+'\n')
            return
        (repo/'memory/EXPERIMENT_STATUS.json').write_text(json.dumps(status, indent=2)+'\n')
        note = '# Native distillation checkpoint: '+tag+'\n\n'+json.dumps(status)+'\n\n'
        for name in ['CURRENT.md', 'HEAD_EXPERIMENT.md']:
            path = repo/'memory'/name
            path.write_text(note+path.read_text())
        with (repo/'memory/SESSION_LOG.md').open('a') as stream:
            stream.write('\n'+note)
        subprocess.run(['git', 'add', 'memory/EXPERIMENT_STATUS.json', 'memory/CURRENT.md', 'memory/HEAD_EXPERIMENT.md', 'memory/SESSION_LOG.md', str(reportdir.relative_to(repo))], cwd=repo, check=True)
        subprocess.run(['git', 'commit', '-m', 'Checkpoint '+tag+' native teacher audit'], cwd=repo, check=True)
        subprocess.run(['git', 'push', 'origin', 'jev_v1.1'], cwd=repo, check=True)
    for limit in (2, 20):
        log = root/f'teacher_{limit:03d}.log'
        command = [sys.executable, '-u', 'scripts/run_decision_rag_stages.py', *common,
                   '--phase', 'teacher', '--start', '0', '--limit', str(limit)]
        checkpoint(f'native{limit}_starting', {'command': command, 'log': str(log), 'active_process_record': str(root/'active_process.json'), 'training_completed': False})
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
            active = {'phase': f'native{limit}_running', 'pid': os.getpid(), 'child_pid': process.pid,
                      'command': command, 'log': str(log), 'updated_epoch': time.time()}
            (root/'active_process.json').write_text(json.dumps(active, indent=2)+'\n')
            code = process.wait()
            active.update(phase=f'native{limit}_process_exited', exit_code=code, updated_epoch=time.time())
            (root/'active_process.json').write_text(json.dumps(active, indent=2)+'\n')
        if code:
            checkpoint(f'native{limit}_execution_failed', {'exit_code': code, 'log': str(log), 'training_completed': False})
            raise RuntimeError('Teacher collection failed; see '+str(log))
        report = audit(limit)
        tag = f'native{limit}_complete'
        if report['target_rejections'] or report['reader_errors'] or not report['accepted_states']:
            checkpoint(f'native{limit}_review_required', report)
            return
        checkpoint(tag, report)


if __name__ == '__main__':
    main()
