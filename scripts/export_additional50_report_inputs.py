import hashlib,json,shutil
from pathlib import Path
r=Path('/content/local_fast/lsdrag-work');p=r/'repo';out=r/'results/three_arm_additional50';dest=p/'lab/reports/additional50';assert (out/'COMPLETE.json').exists()
all_results={}
for bench in ['m3docvqa','memgallery']:
 d=dest/bench;d.mkdir(parents=True,exist_ok=True);manifests=[]
 old=r/'results/three_arm_arag8_textquery'/bench/'B0/run_manifest.json';oldids=set(json.loads(old.read_text())['question_ids'])
 for arm in ['B0','C1','G1']:
  run=out/bench/arm;m=json.loads((run/'run_manifest.json').read_text());assert len(m['question_ids'])==50 and len(set(m['question_ids']))==50;assert not oldids.intersection(m['question_ids']);assert m['config']['max_loops']==8 and m['question_image_policy']=='text-only';assert sorted(m['index_manifests'])=={'B0':['qwen_text'],'C1':['gemma_text'],'G1':['gemma_image','gemma_text']}[arm]
  for qid in m['question_ids']:
   z=json.loads((run/(qid+'.json')).read_text());assert z['qid']==qid and z['status']!='failed' and z['question_image_count']==0 and z['loops']<=8
  manifests.append(m)
  for name in ['summary.json','question_metrics.jsonl']:
   shutil.copyfile(out/bench/(arm+'_scores')/name,d/(arm+'_'+name))
  shutil.copyfile(run/'run_manifest.json',d/(arm+'_run_manifest.json'))
  all_results.setdefault(bench,{})[arm]=json.loads((d/(arm+'_summary.json')).read_text())['overall']
 for key in ['question_ids','questions_sha256','corpus_sha256','prompt_sha256','config','code_revision','question_image_policy']:
  assert all(x[key]==manifests[0][key] for x in manifests), (bench,key)
 shutil.copyfile(out/bench/'paired_comparison.json',d/'paired_comparison.json')
shutil.copyfile(out/'COMPLETE.json',dest/'COMPLETE.json')
(r/'recovery/additional50-report-inputs.json').write_text(json.dumps(all_results,indent=2)+'\n')
print(json.dumps(all_results,indent=2))
print('All additional predictions unique, disjoint, successful; fair manifests verified.')
