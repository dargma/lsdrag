"""CPU-only aggregation of recorded v1.1 pilot artifacts; never calls a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from lsdrag.io import dump_json, read_jsonl
from lsdrag.v11_protocol import contrast_summary


def cluster_interval(rows, key, samples=10000):
    groups = {}
    for r in rows:
        groups.setdefault(r['source_group'], []).append(r[key])
    values = list(groups.values())
    if not values: return None
    sums = np.array([sum(x) for x in values]); counts = np.array([len(x) for x in values])
    idx = np.random.default_rng(20261015).integers(0, len(values), size=(samples,len(values)))
    bootstrap = sums[idx].sum(1)/counts[idx].sum(1)
    return {'mean': float(sums.sum()/counts.sum()), 'ci95': np.quantile(bootstrap,[.025,.975]).tolist(),
            'source_groups': len(values), 'unit': 'support+public-table URL component'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();a.report.mkdir(parents=True,exist_ok=True)
    result={'scope':'table-conditioned text+image one-boundary continuation pilot; development only'}
    for name in ['oracle_summary','contrast_stage1','contrast_train','status',
                 'training_exposure_and_target_audit','development_summary']:
        path=a.run/(name+'.json')
        if path.exists(): result[name]=json.loads(path.read_text())
    train=[json.loads(p.read_text()) for p in (a.run/'train').glob('*.json')] if (a.run/'train').exists() else []
    result['current_training_collection']=contrast_summary(train)
    if train:
        mismatches=[];same_results=0
        for r in train:
            for action,rec in r['actions'].items():
                for i in (0,1):
                    c0,c1=rec['cells'][f'{i}0'],rec['cells'][f'{i}1']
                    if c0['input_hash']==c1['input_hash']:
                        same_results+=1
                        if c0['answers']!=c1['answers'] or c0['f1']!=c1['f1']:
                            mismatches.append({'qid':r['qid'],'action':action,'i':i})
        result['identical_input_audit']={'paired_equal_hashes':same_results,'mismatches':mismatches}
        result['collection_seconds']=sum(r['collection_seconds'] for r in train)
    if (a.run/'development').exists():
        rows=[json.loads(p.read_text()) for p in (a.run/'development').glob('*/*.json')]
        result['paired_auxiliary_comparisons']={}
        for lam in (0.,.01):
            paired=[]
            for q in read_jsonl(a.data/'development_questions.jsonl'):
                arms={}
                for beta in (0.,1.):
                    rs=[r for r in rows if r['qid']==q['qid'] and r['policy'].startswith(f'head_l{lam}_b{beta}_')]
                    if len(rs)==3:
                        arms[beta]={key:float(np.mean([r[key] for r in rs])) for key in ('em','f1','total_seconds')}
                if len(arms)==2:
                    paired.append({'qid':q['qid'],'source_group':next(r['source_group'] for r in rows if r['qid']==q['qid']),
                                   **{k:arms[1.][k]-arms[0.][k] for k in ('em','f1','total_seconds')}})
            result['paired_auxiliary_comparisons'][str(lam)]={
                'questions':len(paired),'seed_policy':'average all three predetermined seeds per question before resampling',
                **{k:cluster_interval(paired,k) for k in ('em','f1','total_seconds')}}
    dump_json(a.report/'stage12_summary.json',result)
    oracle=result.get('oracle_summary',{})
    text=['# jev_v1.1 — 1·2단계 실행 기록','',
          '공식 MMQA의 공개 표를 공통 배경으로 사용한 텍스트·이미지 continuation pilot이다. '
          '최종 benchmark test나 독창성·비열등성 검증이 아니다.','',
          '- GPU: RTX PRO 6000 Blackwell Server Edition, 96GB.',
          '- Reader: pinned Gemma 4 E4B-it; backbone 동결.',
          '- 질문: train 300 / development 80. 최초 진단 40은 train의 사전 고정 부분집합.',
          '- 공통 축소 corpus: 텍스트 503개·이미지 427개. 실제 이미지 벡터와 텍스트 벡터를 분리.',
          '- 최초 실행에서 누락된 공개 표를 발견해 원결과를 보존하고 동일 40문항·gate로 한 번 재진단.',
          '- 수정 전 EM 6/40·F1 23.3%는 공개 문맥이 누락된 진단이며 완전한 evidence oracle로 해석하지 않는다.','']
    if oracle:
        text+=['## 1A. 수정된 Reader 진단','',
               '| 조건 | EM | F1 | 평균 답변 호출 시간 |','|---|---:|---:|---:|']
        for label,name in [('질문+공개 표','public_background'),('공개 표+정답 근거','gold_evidence')]:
            r=oracle['arms'][name]
            text.append(f'| {label} | {r["em"]:.1%} | {r["f1"]:.1%} | {r["cost_seconds"]:.3f}s |')
        text+=['',f'기존 gate 판정: **{oracle["gate"]["passed"]}**. '
               '기준은 40문항 중 완전 정답 10개 이상이며 F1 순개선 0.05 이상이다.','']
    text+=['## 1B. 모듈 접근 개입','',
           '자연 검색의 첫 증거를 query generator와 final reader에 보이거나 숨기는 2×2를 수집한다. '
           '공개 표·사용한 budget은 모든 조건에서 동일하다. 검색어·중간 답은 final reader에 전달하지 않는다.','',
           '현재 수집/효과 요약:','',
           '```json',json.dumps(result['current_training_collection'],ensure_ascii=False,indent=2),'```','']
    if 'status' in result:
        text+=['## 단계 판정','', '```json',json.dumps(result['status'],ensure_ascii=False,indent=2),'```','']
    if 'development_summary' in result:
        text+=['## 2. Head 학습 및 development 실행','',
               'return-only와 return+access는 같은 MLP·라벨·학습 횟수이며 β만 다르다. '
               'λ=0/0.01, 사전 고정 seed 3개를 모두 보고한다.','',
               '| 정책 | EM | F1 | 전체 시간 | 결정 시간 |','|---|---:|---:|---:|---:|']
        for name,r in result['development_summary'].items():
            text.append(f'| {name} | {r["em"]:.1%} | {r["f1"]:.1%} | {r["total_seconds"]:.3f}s | {r["decision_seconds"]:.3f}s |')
        text+=['','보조학습 대비 차이와 source-component bootstrap 구간은 '
               '[전체 요약](stage12_summary.json)에 있다. 좋은 seed만 고르지 않는다.','']
    text+=['## 해석의 한계','',
           '이미지 관측은 pixels+title package다. 시각 정보만의 필요성은 아직 분리하지 않았다. '
           '공유 corpus와 공개 표로 인해 엄격한 unseen-source 평가가 아니며, 실제 train 노출을 별도로 감사한다. '
           '동일 입력의 셀 재사용은 비용 측정값을 유지한 counterfactual 기록이며 실제 새 정책 평가는 답변을 다시 생성한다. '
           '학습 완료와 가설 지지는 별개다.','',
           '[실행 재검토](../../../docs/JEV_V11_EXECUTION_REVIEW.md) · '
           '[고정 계획](../../registry/jev_v1.1/execution_plan.json) · '
           '[전체 집계](stage12_summary.json)','']
    (a.report/'STAGE12_PROGRESS.md').write_text('\n'.join(text))
    print(json.dumps({'train_collected':len(train),'report':str(a.report),'status':result.get('status')}))


if __name__=='__main__': main()
