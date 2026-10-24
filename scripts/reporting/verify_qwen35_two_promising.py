"""Raw official scores, paired correctness costs and cached comparator identity."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
NAME='qwen35_9b_two_promising_validation64_20261024'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(es):
 n=len(es);total=sum(e['generated_tokens'] for e in es)
 return dict(n=n,total_generated=total,mean_generated=total/n if n else None)
def main():
 run=ROOT/'results'/NAME;spec=read(run/'spec.json');status=read(run/'status.json');assert status['phase']=='complete' and status['episodes']==128
 oldrun=ROOT/'results/qwen35_9b_h3_validation64_20261024';old=read(oldrun/'spec.json');audit=read(REPO/'lab/reports'/oldrun.name/'verification.json');assert audit['verified']
 for k in ['model','corpus','document_sha256','vectors_sha256','qids']:assert spec[k]==old[k],k
 for f,h in old['source_sha256'].items():assert sha(REPO/f)==h,f
 for f,h in spec['source_sha256'].items():assert sha(REPO/f)==h,f
 for p,h in audit['episode_sha256'].items():assert sha(p)==h,p
 for s in [spec,old]:
  for p in s['policies'].values():assert sha(p['checkpoint'])==p['sha256']
 plan=read(REPO/'lab/registry/qwen35_9b_two_promising_20261024/plan.json');assert plan['dev64']==spec['qids'];assert not set(plan['train192'])&set(spec['qids'])
 protected=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json');assert not set(spec['qids'])&set(protected['reserved_qids']+protected['test_qids'])
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 gold={q['id']:[q['answer']]+q['answer_aliases'] for s in ['train','dev'] for q in map(json.loads,(ROOT/f'source/musique/musique_ans_v1.0_{s}.jsonl').open())}
 rows={};hashes={};newtotal=0;newcount=0
 for directory in [oldrun,run]:
  for p in sorted((directory/'episodes').glob('*.json')):
   e=read(p);assert e['qid'] in spec['qids'];key=e['policy'],e['qid'];assert key not in rows
   assert abs(e['f1']-max(compute_f1(a,e['answer']) for a in gold[e['qid']]))<1e-10
   assert e['generated_tokens']==sum(c.get('generated_tokens',0) for c in e['reader_calls'])
   for c in e['reader_calls']:
    if 'token_ids' in c:assert len(c['token_ids'])==c['generated_tokens']
   e['em']=max(compute_exact(a,e['answer']) for a in gold[e['qid']]);rows[key]=e;hashes[str(p)]=sha(p)
   if directory==run:newtotal+=e['generated_tokens'];newcount+=1
 assert newcount==128 and newtotal==status['generated'] and len(rows)==320
 policies={}
 for name in list(old['policies'])+list(spec['policies']):
  es=[rows[name,q] for q in spec['qids']]
  policies[name]=dict(f1=sum(e['f1'] for e in es)/64,all=stats(es),correct=stats([e for e in es if e['em']]),incorrect=stats([e for e in es if not e['em']]),errors=sum(bool(e['errors']) for e in es),mean_tools=sum(e['tool_calls'] for e in es)/64)
 paired={};screen={}
 for cand in spec['policies']:
  for ref in old['policies']:
   groups={}
   for a,b,label in [(1,1,'both_correct'),(0,1,'gained_correct'),(1,0,'lost_correct'),(0,0,'both_incorrect')]:
    qs=[q for q in spec['qids'] if rows[ref,q]['em']==a and rows[cand,q]['em']==b];rs=stats([rows[ref,q] for q in qs]);cs=stats([rows[cand,q] for q in qs]);groups[label]=dict(reference=rs,candidate=cs,saved_tokens=rs['total_generated']-cs['total_generated'])
   paired[f'{cand}_vs_{ref}']=groups
  screen[cand]=policies[cand]['f1']>=policies['base']['f1'] and policies[cand]['correct']['n']>=policies['base']['correct']['n'] and paired[f'{cand}_vs_base']['both_correct']['saved_tokens']>0 and policies[cand]['errors']==0
 out=REPO/'lab/reports'/NAME;out.mkdir(parents=True,exist_ok=True)
 report=dict(verified=True,policies=policies,paired=paired,screen=screen,episode_sha256=hashes,cached_references=True,scope='Two fixed adapted hypotheses on repeatedly exposeddev64; no independent confirmation')
 (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# 9B 두 후보 결과','', '동일 개발64문항. base/H3/constant는 동일 프로토콜의 검증된 기존 실행을 재사용했다.','', '| 정책 | F1(%) | EM/64 | 평균 생성토큰 | 정답 평균 | 오답 평균 | 오류 |','|---|---:|---:|---:|---:|---:|---:|']
 for n,p in policies.items():lines.append(f"| {n} | {100*p['f1']:.2f} | {p['correct']['n']} | {p['all']['mean_generated']:.2f} | {p['correct']['mean_generated']} | {p['incorrect']['mean_generated']} | {p['errors']} |")
 lines+=['','| 후보 vs 기준 | 정오 분류 | 문항 | 기준 토큰 | 후보 토큰 | 절약 |','|---|---|---:|---:|---:|---:|']
 for comp,groups in paired.items():
  for group,g in groups.items():lines.append(f"| {comp} | {group} | {g['reference']['n']} | {g['reference']['total_generated']} | {g['candidate']['total_generated']} | {g['saved_tokens']} |")
 lines+=['','고정한 탐색 통과 기준(base보다 F1·EM 유지/개선 + 공통 정답 토큰 절감 + 오류0): '+json.dumps(screen),'','두 후보는 기존 방법의9B 적용이다. 고정 이전 정책의 후속 결과로 학습했으며, value_aux는 표현 공동학습도 포함하므로 순수 auxiliary 효과를 분리하지 않는다. 반복 개발셋 결과이며 독립 검증 또는 지연시간 주장을 하지 않는다.']
 (REPO/'docs/JEV_QWEN35_TWO_PROMISING_RESULTS_20261024.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'policies':policies,'screen':screen},indent=2))
if __name__=='__main__':main()
