"""Continue the bounded experiment after collection, publishing real milestones."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from lsdrag.io import dump_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--reader-server-pid',type=int,required=True)
    a=p.parse_args();r=a.work_root;repo=Path(__file__).resolve().parents[1]
    output=r/'results/jev_v1.0';report=repo/'lab/reports/jev_v1.0';drive=Path('/content/drive/MyDrive/lsdrag/outputs/jev_v1.0')
    def backup():
        shutil.copytree(output,drive,dirs_exist_ok=True)
    def publish(stage,text,files):
        if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
            raise RuntimeError('Uncommitted changes; preserve results and stop before publishing')
        targets=[]
        for source,name in files:
            shutil.copy2(source,report/name);targets.append(str((report/name).relative_to(repo)))
        progress=report/'PROGRESS.md'
        with progress.open('a') as f: f.write('\n## '+stage+'\n\n'+text+'\n')
        targets.append(str(progress.relative_to(repo)))
        subprocess.run(['git','add',*targets],cwd=repo,check=True)
        subprocess.run(['git','commit','-m',stage],cwd=repo,check=True)
        subprocess.run(['git','push','origin','jev_v1.0'],cwd=repo,check=True)
        backup()
    # Copy raw trajectories regularly so local-fast loss does not lose labels.
    while not (output/'collection_complete.json').exists():
        backup();time.sleep(30)
    collection=json.loads((output/'collection_complete.json').read_text())
    audit=json.loads((output/'pooled_label_audit.json').read_text())
    gate=audit['gate']
    publish('Teacher label collection completed',f"성공한 answerable 학습 문항 **{gate['answerable_successful_questions']}개**. 행동별 서로 다른 성공 문항: `{json.dumps(gate['distinct_questions_per_action'],ensure_ascii=False)}`. 라벨 진입 기준 통과: **{gate['passed']}**. [수집 감사](pooled_label_audit.json).",[(output/'pooled_label_audit.json','pooled_label_audit.json'),(output/'collection_complete.json','collection_complete.json')])
    # Stop only the exact Reader server started for this experiment.
    cmdline=Path(f'/proc/{a.reader_server_pid}/cmdline')
    if cmdline.exists():
        command=cmdline.read_bytes().replace(b'\0',b' ')
        if b'vllm' not in command or b'--port 8102' not in command or b'gemma-reader' not in command:
            raise RuntimeError('Server PID identity changed; do not signal it')
        os.kill(a.reader_server_pid,signal.SIGTERM)
    if not gate['passed']:
        print('NoGo: label shortage; full head comparison not run.',flush=True)
        return
    # Allow our server to release memory. Other GPU jobs are never terminated.
    for _ in range(60):
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not processes: break
        time.sleep(2)
    else: raise RuntimeError('GPU still occupied; timing evaluation deferred')
    head=output/'head.pt'
    subprocess.run([str(r/'runtime/bin/python'),'scripts/train_decision_head.py','--work-root',str(r),'--audit',str(output/'pooled_label_audit.json'),'--output',str(head)],cwd=repo,check=True)
    trained=json.loads(head.with_suffix('.json').read_text())
    publish('Frozen Gemma decision head trained',f"backbone 동결·linear head만 CE로 한 번 학습했다. 성공 궤적의 **{trained['turns']}턴**을 사용했다. [학습 설정·출처](head_training.json). 학습 loss는 정답률이나 폐루프 성능을 의미하지 않는다. 최종 비교용 새 M3 20문항은 아직 학습/설정 선택에 사용하지 않았다.",[(head.with_suffix('.json'),'head_training.json')])
    evaluation=output/'paired_m3_test20'
    subprocess.run([str(r/'runtime/bin/python'),'scripts/evaluate_decision_pair.py','--work-root',str(r),'--head',str(head),'--output',str(evaluation)],cwd=repo,check=True)
    result=json.loads((evaluation/'summary.json').read_text())
    arms=result['arms'];b=arms['B-HF'];d=arms['D-head']
    table='| 조건 | 문항 | EM | F1 | 평균 전체 시간 | 중앙값 | 평균 fallback |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,row in arms.items():
        table+=f"| {name} | {row['n']} | {100*row['em']:.1f}% | {100*row['f1']:.1f}% | {row['seconds']:.3f}s | {row['median_seconds']:.3f}s | {row['fallback_count']:.2f} |\n"
    table+=f"\n평균 전체 시간 비율 B-HF/D-head: **{b['seconds']/d['seconds']:.3f}배**. 인자·prefix·검색·fallback·최종 답변 시간을 포함한다. [문항별 결과·시간 분해](paired_summary.json). 20문항의 작은 PoC이므로 정확도 비열등성·일반적 속도 개선·보정된 정답 확률을 주장하지 않는다. 보완 자료 사용 여부와 행동별 학습 수는 수집 감사에 명시했다."
    publish('Paired Gemma decision comparison completed',table,[(evaluation/'summary.json','paired_summary.json'),(evaluation/'manifest.json','paired_manifest.json')])
    print('Completed and pushed final paired report.',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        print(f'Experiment stopped for inspection: {type(error).__name__}: {error}',flush=True)
        raise
