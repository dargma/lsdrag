"""CPU-only audit of the saved head, cached training features and recorded runs.

No backbone forward, generation, optimization, calibration or test-time tuning.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import torch
from lsdrag.decision import ACTIONS,action_label,trajectory_states
from lsdrag.io import read_jsonl,dump_json,sha256

p=argparse.ArgumentParser();p.add_argument('--work-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.work_root
repo=Path(__file__).resolve().parents[1];run=r/'results/jev_v1.0';torch.set_num_threads(4)
checkpoint=torch.load(run/'head.pt',map_location='cpu',weights_only=True)
cache=torch.load(run/'head.features.pt',map_location='cpu',weights_only=True)
metadata=json.loads((run/'head.json').read_text());paired=json.loads((run/'paired_m3_test20/manifest.json').read_text())
assert sha256(run/'head.pt')==metadata['checkpoint_sha256']==paired['head_sha256']
x,y=cache['features'],cache['labels'];w,b=checkpoint['weight'],checkpoint['bias'];logits=x@w.T+b;prob=logits.softmax(-1);pred=prob.argmax(-1)
assert len(x)==len(metadata['features'])==197
assert [ACTIONS.index(f['action']) for f in metadata['features']]==y.tolist()
conf=torch.zeros((4,4),dtype=torch.int64)
for label,chosen in zip(y,pred):conf[label,chosen]+=1
counts=Counter(y.tolist());groups=defaultdict(list);input_counts=Counter();image_qids=set()
for feature in metadata['features']:
 source=feature['source']
 assert sha256(source)==feature['source_sha256']
 recorded=json.loads(Path(source).read_text())
 has_image=bool(recorded.get('question_image_count')) or any(c.get('new_figure_ids') for c in recorded['reader_calls'] if c['loop']<=feature['loop'])
 input_counts['with_image' if has_image else 'text_only']+=1
 if has_image:image_qids.add(feature['qid'])
 group='MultiHop-RAG' if 'teacher_multihop' in source else ('MMQA train' if 'teacher_mmqa' in source else ('new M3 train82' if 'teacher_m3' in source else 'old M3 train60'))
 groups[group].append(feature)
data=[{'source':name,'distinct_questions':len({f['qid'] for f in features}),'trajectories':len({f['source'] for f in features}),'turns':len(features),'action_turns':dict(Counter(f['action'] for f in features))} for name,features in groups.items()]
plan=json.loads((repo/'lab/registry/jev_v1.0/label_collection_plan.json').read_text());split=json.loads((repo/'lab/registry/jev_v1.0/split.json').read_text())['splits'];trainids={f['qid'] for f in metadata['features']}
assert not trainids.intersection(set(plan['fresh_m3_test_qids'])|set(split['test'])|set(split['development']))
figures={f['figure_id']:f for f in read_jsonl(r/'data/ir_final288/figures.jsonl')};prompt=(repo/'prompts/search_linked.txt').read_text()
metrics=json.loads((run/'paired_m3_test20/summary.json').read_text())['rows'];scores={(m['qid'],m['arm']):m['em'] for m in metrics}
rollouts={};shared=[];finals=[];conf_shared=torch.zeros((4,4),dtype=torch.int64)
for arm in ['B-HF','D-head']:
 histogram=Counter();qs=0
 for path in sorted((run/'paired_m3_test20'/arm).glob('*.json')):
  z=json.loads(path.read_text());qs+=1
  states=list(trajectory_states(z,prompt,figures))
  for state,call in zip(states,z['reader_calls']):
   action=action_label(call['message']);histogram[action]+=1
   if state['forced_final']:continue
   # Hash exact reconstructed input, including actual image bytes. Comparison
   # only when both policies reached the identical state of the same question.
   import hashlib
   state_hash=hashlib.sha256(json.dumps(state['messages'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
   key=(z['qid'],state_hash)
   if arm=='B-HF': rollouts[key]={'action':action,'loop':call['loop']}
   else:
    selected=call['usage']['selected_action']
    if selected=='FINAL':finals.append({'qid':z['qid'],'loop':call['loop'],'action_probability':call['usage']['action_probabilities'][3],'answer_em':scores[(z['qid'],arm)]})
    if key in rollouts:
     target=rollouts[key]['action'];conf_shared[ACTIONS.index(target),ACTIONS.index(selected)]+=1
     shared.append({'qid':z['qid'],'state_sha256':state_hash,'baseline_loop':rollouts[key]['loop'],'head_loop':call['loop'],'baseline_action':target,'head_action':selected,'match':target==selected})
 groups_key=arm
 if arm=='B-HF':base_hist=dict(histogram)
 else:head_hist=dict(histogram)
normalized=torch.nn.functional.normalize(w,dim=1);center=w-w.mean(dim=0)
report={'audit_kind':'posthoc CPU inspection; no new solves, GPU calls, retraining, threshold tuning or independent development feature extraction','checkpoint_sha256':sha256(run/'head.pt'),'checkpoint_matches_deployed_head':True,'question_split_overlap':False,'head':{'input_dimension':w.shape[1],'classes':list(ACTIONS),'parameter_count':w.numel()+b.numel(),'weight_shape':list(w.shape),'bias':b.tolist(),'all_finite':bool(torch.isfinite(w).all() and torch.isfinite(b).all()),'weight_min':float(w.min()),'weight_max':float(w.max()),'weight_mean':float(w.mean()),'weight_std':float(w.std()),'zero_fraction':float((w==0).float().mean()),'row_l2_norms':w.norm(dim=1).tolist(),'row_cosines':(normalized@normalized.T).tolist(),'singular_values':torch.linalg.svdvals(w).tolist(),'centered_singular_values':torch.linalg.svdvals(center).tolist(),'weight_rank':int(torch.linalg.matrix_rank(w)),'class_relative_rank':int(torch.linalg.matrix_rank(center))},'training':{'distinct_questions':len(trainids),'trajectories':len({f['source'] for f in metadata['features']}),'turns':len(x),'dataset_breakdown':data,'training_input_composition':dict(input_counts),'distinct_train_questions_with_image_states':len(image_qids),'target_action_counts':{action:counts[i] for i,action in enumerate(ACTIONS)},'predicted_action_counts':{action:int((pred==i).sum()) for i,action in enumerate(ACTIONS)},'teacher_action_accuracy':float((pred==y).float().mean()),'majority_class_accuracy':max(counts.values())/len(y),'confusion_rows_true_columns_predicted':conf.tolist(),'recomputed_ce_after_final_step':float(torch.nn.functional.cross_entropy(logits,y)),'logged_last_ce_before_final_step':metadata['training_loss'],'learning_curve_available':False,'initial_checkpoint_available':False,'fit':metadata['fit']},'recorded_heldout_rollouts':{'questions':20,'baseline_action_counts':base_hist,'head_action_counts':head_hist,'head_FINAL_probability_ge_0_9':sum(f['action_probability']>=.9 for f in finals),'head_FINAL_calls':len(finals),'head_FINAL_at_first_call':sum(f['loop']==1 for f in finals),'head_FINAL_high_probability_but_answer_EM0':sum(f['action_probability']>=.9 and f['answer_em']==0 for f in finals),'head_final_calls':finals},'same_input_posthoc_action_comparison':{'states':len(shared),'distinct_questions':len({s['qid'] for s in shared}),'agreement':sum(s['match'] for s in shared)/len(shared),'confusion_rows_baseline_columns_head':conf_shared.tolist(),'states_detail':shared,'limitation':'Existing final-test traces only; shared prefixes are selectively observed, not an independent development evaluation or optimal-action labels.'},'conclusion_limits':['Finite, non-collapsed weights and perfect training fit do not establish generalization.','45 questions / 197 turns are small relative to 10244 head parameters; overfitting remains possible.','Training is dominated by MultiHop-RAG (160/197 turns); 55/57 read_chunk labels are from that text-only fallback.','Only EM0->EM1 heldout case is semantically correct in both arms; answer format explains exact-match improvement.','No independent heldout teacher-action benchmark, repeated training seeds, or FINAL-only formatting control was run.','Do not interpret action softmax as answer correctness probability.']}
dump_json(a.output,report)
print(json.dumps({'training_accuracy':report['training']['teacher_action_accuracy'],'weights_finite':report['head']['all_finite'],'rollout_actions':head_hist,'shared_states':len(shared),'shared_agreement':report['same_input_posthoc_action_comparison']['agreement'],'high_final_but_EM0':report['recorded_heldout_rollouts']['head_FINAL_high_probability_but_answer_EM0']}))
