"""Final paired rollout audit including exact before replay and EM-stratified costs."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2];NAME='qwen35_outcome_rerank_validation64_20261025'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(es):
 n=len(es);t=sum(e['generated_tokens'] for e in es);return dict(n=n,total_generated=t,mean_generated=t/n if n else None)
def main():
 run=ROOT/'results'/NAME;spec=read(run/'spec.json');status=read(run/'status.json');assert status['phase']=='complete' and status['episodes']==128
 for f,h in spec['source_sha256'].items():assert sha(REPO/f)==h
 for p in spec['policies'].values():assert sha(p['checkpoint'])==p['sha256']
 plan=read(REPO/'lab/registry/qwen35_outcome_rerank_20261025/plan.json');split=read(REPO/'lab/registry/qwen35_9b_h3_20261024/training_plan.json');protected=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json');assert spec['qids']==split['development64'];assert not set(spec['qids'])&set(split['train']+protected['reserved_qids']+protected['test_qids'])
 oldrun=ROOT/'results/qwen35_9b_two_promising_validation64_20261024';oldspec=read(oldrun/'spec.json');oldverify=read(REPO/'lab/reports'/oldrun.name/'verification.json')
 for k in ['model','corpus','document_sha256','vectors_sha256','qids']:assert spec[k]==oldspec[k]
 assert spec['policies']['before']['sha256']==oldspec['policies']['value_aux']['sha256']==sha(plan['checkpoint'])
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 gold={q['id']:[q['answer']]+q['answer_aliases'] for sp in ['train','dev'] for q in map(json.loads,(ROOT/f'source/musique/musique_ans_v1.0_{sp}.jsonl').open())}
 rows={};hashes={};tokens=inputs=0;replay=0
 for path in sorted((run/'episodes').glob('*.json')):
  e=read(path);assert e['qid'] in spec['qids'] and e['policy'] in ['before','after'];key=e['policy'],e['qid'];assert key not in rows
  assert abs(e['f1']-max(compute_f1(a,e['answer']) for a in gold[e['qid']]))<1e-10;e['em']=max(compute_exact(a,e['answer']) for a in gold[e['qid']]);assert e['generated_tokens']==sum(u.get('generated_tokens',0) for u in e['reader_calls'])
  for u in e['reader_calls']:
   if 'token_ids' in u:assert len(u['token_ids'])==u['generated_tokens']
  if e['policy']=='before':
   previous=oldrun/'episodes'/f"value_aux_{e['qid']}.json";assert sha(previous)==oldverify['episode_sha256'][str(previous)];old=read(previous)
   for k in ['answer','support_ids','f1','generated_tokens','prompt_tokens','tool_calls','errors']:assert e[k]==old[k],(e['qid'],k)
   assert [d['action'] for d in e['decisions']]==[d['action'] for d in old['decisions']];replay+=1
  rows[key]=e;hashes[str(path)]=sha(path);tokens+=e['generated_tokens'];inputs+=e['prompt_tokens']
 assert len(rows)==128 and tokens==status['generated'] and inputs==status['inputs'] and replay==64
 policies={}
 for name in ['before','after']:
  es=[rows[name,q] for q in spec['qids']];policies[name]=dict(f1=sum(e['f1'] for e in es)/64,all=stats(es),correct=stats([e for e in es if e['em']]),incorrect=stats([e for e in es if not e['em']]),partial=stats([e for e in es if not e['em'] and 0<e['f1']<1]),f1_one_nonexact=stats([e for e in es if not e['em'] and e['f1']==1]),mean_input=sum(e['prompt_tokens'] for e in es)/64,mean_tools=sum(e['tool_calls'] for e in es)/64,errors=sum(bool(e['errors']) for e in es),forced_final=sum(e['forced_final'] for e in es))
 groups={}
 for a,b,label in [(1,1,'both_correct'),(0,1,'gained_correct'),(1,0,'lost_correct'),(0,0,'both_incorrect')]:
  qs=[q for q in spec['qids'] if rows['before',q]['em']==a and rows['after',q]['em']==b];ref=stats([rows['before',q] for q in qs]);cand=stats([rows['after',q] for q in qs]);groups[label]=dict(qids=qs,before=ref,after=cand,saved_tokens=ref['total_generated']-cand['total_generated'])
 effect=dict(f1_delta=policies['after']['f1']-policies['before']['f1'],token_ratio=policies['after']['all']['total_generated']/policies['before']['all']['total_generated'],f1_wins=sum(rows['after',q]['f1']>rows['before',q]['f1'] for q in spec['qids']),f1_losses=sum(rows['after',q]['f1']<rows['before',q]['f1'] for q in spec['qids']))
 passed=effect['f1_delta']>=0 and policies['after']['correct']['n']>=policies['before']['correct']['n'] and policies['after']['all']['mean_generated']<policies['before']['all']['mean_generated'] and policies['after']['correct']['mean_generated']<policies['before']['correct']['mean_generated'] and groups['both_correct']['saved_tokens']>=0 and policies['after']['errors']==0
 report=dict(verified=True,replayed_reference64=True,policies=policies,groups=groups,effect=effect,screen_passed=passed,episode_sha256=hashes,scope='Final fixed experiment; repeated dev64, no independent confirmation; no controlled latency claim')
 out=REPO/'lab/reports'/NAME;out.mkdir(parents=True,exist_ok=True);(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# 행동 후 정답 가능성 기반 정책 실험 결과','', '동일9B 가치 보조학습 head의 전후64문항 비교. 학습전64개 원본이 기존실행과 답변·점수·행동·비용까지 정확히 재현됐다.','', '| 정책 | F1(%) | 정답/64 | 평균 생성토큰 | 평균 입력토큰 | 평균 도구호출 | 오류 |','|---|---:|---:|---:|---:|---:|---:|']
 for n,p in policies.items():lines.append(f"| {n} | {100*p['f1']:.2f} | {p['correct']['n']} | {p['all']['mean_generated']:.2f} | {p['mean_input']:.2f} | {p['mean_tools']:.2f} | {p['errors']} |")
 lines+=['',f"F1 변화 {effect['f1_delta']*100:+.4f}pp, 생성토큰 변화 {(effect['token_ratio']-1)*100:+.2f}%. 문항별F1 개선{effect['f1_wins']}/하락{effect['f1_losses']}.",'','| 정오 변화 | 문항 | 전 총토큰 | 후 총토큰 | 절약 |','|---|---:|---:|---:|---:|']
 for n,g in groups.items():lines.append(f"| {n} | {g['before']['n']} | {g['before']['total_generated']} | {g['after']['total_generated']} | {g['saved_tokens']} |")
 lines+=['','| 정책 | 분류 | 문항 | 총토큰 | 평균토큰 |','|---|---|---:|---:|---:|']
 for n,p in policies.items():
  for k in ['correct','incorrect','partial','f1_one_nonexact']:
   z=p[k];mean='—' if z['mean_generated'] is None else f"{z['mean_generated']:.2f}";lines.append(f"| {n} | {k} | {z['n']} | {z['total_generated']} | {mean} |")
 lines+=['',f'고정한 탐색 기준(전 대비F1/EM유지, 전체평균과정답평균생성토큰감소, 공통정답비용비증가, 오류0): {passed}.','', '정답은공식EM이며 부분F1오답은incorrect에포함된다. 공유동결표현 위 행동별EM/비용readout만학습, 동결LLM/도구/인자생성 동일. 반복사용한개발64이므로 독립일반화증거는아니다. 사용자가승인한추가본실험1개이며계수탐색/확장은없다.']
 (REPO/'docs/JEV_OUTCOME_RERANK_RESULT_20261025.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='episode_sha256'},indent=2))
if __name__=='__main__':main()
