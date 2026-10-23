"""CPU same-hidden logit-change decomposition; no new LLM inference."""
import json
from pathlib import Path
import numpy as np
import torch
from lsdrag.decision_policy import PolicyHead
REPO=Path(__file__).resolve().parents[2];BASE=Path('/content/local_fast/lsdrag-text-v1/results')
def read(p):return json.loads(p.read_text())
def main():
 torch.set_num_threads(2);rows=torch.load(BASE/'decision_policy_20261023_pilot2/round0/counterfactual_data.pt',weights_only=True,map_location='cpu');x=torch.stack([r['pack'] for r in rows]).float();legal=torch.stack([r['legal'] for r in rows]);qids=[r['qid'] for r in rows];w=np.array([1/qids.count(q) for q in qids]);w/=w.sum();anchor=PolicyHead(2560,'mlp');anchor.load_state_dict(torch.load(BASE/'decision_policy_20261023_pilot2/reference_round0.pt',weights_only=True,map_location='cpu'));anchor.eval();folds=read(REPO/'lab/registry/decision_value_probe_20261023/manifest.json')['folds']
 with torch.no_grad():al=anchor(x,legal)[0].numpy()
 output=[]
 for seed,run in enumerate(['decision_ppo_tempered_20261023','decision_ppo_tempered_seed2_20261023'],1):
  full=PolicyHead(2560,'mlp');full.load_state_dict(torch.load(BASE/run/'ppo_iteration2.pt',weights_only=True,map_location='cpu'));full.eval()
  assert all(torch.equal(v,full.state_dict()[k]) for k,v in anchor.state_dict().items() if k.startswith('trunk.') or k in ['mean','scale'])
  with torch.no_grad():fl=full(x,legal)[0].numpy()
  delta=fl-al;design=[];target=[];pw=[];state=[]
  for i,mask in enumerate(legal.numpy()):
   acts=np.flatnonzero(mask);pairs=[(a,b) for j,a in enumerate(acts) for b in acts[j+1:]]
   for a,b in pairs:
    v=np.zeros(4);v[a]=1;v[b]=-1;design.append(v);target.append(delta[i,a]-delta[i,b]);pw.append(w[i]/len(pairs));state.append(i)
  D=np.array(design);y=np.array(target);pw=np.array(pw);state=np.array(state)
  def fit(mask):return np.linalg.lstsq(D[mask]*np.sqrt(pw[mask,None]),y[mask]*np.sqrt(pw[mask]),rcond=None)[0]
  bias=fit(np.ones(len(y),dtype=bool));total=float(pw@y**2);sse=float(pw@(y-D@bias)**2);pred=np.zeros_like(y);oof_logits=np.zeros_like(al)
  for held in folds:
   te=np.array([q in held for q in qids]);b=fit(~te[state]);pred[te[state]]=D[te[state]]@b;oof_logits[te]=al[te]+b
  oof_logits[~legal.numpy()]=-1e9;global_logits=al+bias;global_logits[~legal.numpy()]=-1e9
  klbias=read(BASE/'decision_policy_ablation_20261023/fit_stats.json')[f's{seed}']['bias'];kl_logits=al+np.array(klbias);kl_logits[~legal.numpy()]=-1e9
  af=fl.argmax(1);aa=al.argmax(1);ag=kl_logits.argmax(1)
  changed=af!=aa;unexplained=af!=ag
  output.append({'seed':seed,'states':len(rows),'questions':len(set(qids)),'trunk_and_normalizer_unchanged':True,'global_pairwise_bias':bias.tolist(),'pairwise_change_energy':total,'global_bias_explained_fraction_in_sample':1-sse/total,'global_bias_explained_fraction_question_oof':1-float(pw@(y-pred)**2)/total,'weighted_anchor_full_argmax_agreement':float(w@(aa==af)),'weighted_pairwise_global_full_argmax_agreement':float(w@(global_logits.argmax(1)==af)),'weighted_oof_global_full_argmax_agreement':float(w@(oof_logits.argmax(1)==af)),'weighted_kl_global_full_argmax_agreement':float(w@(ag==af)),'raw_changed_vs_anchor':int(changed.sum()),'raw_disagreement_vs_kl_bias':int(unexplained.sum()),'changed_and_not_explained_by_kl_bias':int((changed&unexplained).sum())})
 out=REPO/'lab/reports/decision_ppo_internal_signal_20261023';out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(output,indent=2)+'\n')
 lines=['# PPO 내부 신호: 전역 편향과 상태별 변화','', '기존train67개동일hidden,24질문,추가LLM실행없음. trunk/normalizer불변검증. Δ(s)=logit_PPO(s)−logit_anchor(s). Softmax공통offset은무의미하므로각합법action쌍의Δa−Δb를분해한다. 질문균등·상태내pair균등가중 least squares로4개전역bias를fit한다. 설명비율=1−잔여제곱합/원래변화제곱합;성능설명비율이아니다. 추가로질문fold밖에서biasfit후heldout신호재구성.','', '| seed | 전역bias 신호설명비율(train) | 질문분리 설명비율 | anchor/full 행동일치 | KLbias/full 행동일치 | 전체변경상태 | KLbias로불일치상태 |','|---|---:|---:|---:|---:|---:|---:|']
 for r in output:lines.append(f"| {r['seed']} | {r['global_bias_explained_fraction_in_sample']:.3f} | {r['global_bias_explained_fraction_question_oof']:.3f} | {r['weighted_anchor_full_argmax_agreement']:.3f} | {r['weighted_kl_global_full_argmax_agreement']:.3f} | {r['raw_changed_vs_anchor']} | {r['raw_disagreement_vs_kl_bias']} |")
 lines+=['','상태별수정의존재/규모는내부신호로진단가능하다. 그것이답변에유용한지는이수치만으로판정불가. 가중일치율과raw상태수분모가다르므로혼동금지. 최종PPO는원train질문으로학습됐으므로OOF도새질문정책일반화증거가아닌bias분해안정성진단. 전체편향이변화를많이설명해도기존anchor자체의상태판단을제거한것은아니다.']
 (REPO/'docs/JEV_DECISION_PPO_INTERNAL_SIGNAL_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
