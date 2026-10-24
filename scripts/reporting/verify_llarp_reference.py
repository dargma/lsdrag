"""Execute isolated original LLaRP critic/GAE/reward code without Habitat or weights.
This verifies code behavior, not the embodied benchmark or full training stack.
"""
import ast,copy,hashlib,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

SOURCE=Path('/content/local_fast/lsdrag-text-v1/source/ml-llarp-verification')
REPO=Path('/content/local_fast/lsdrag-code')
REV='214c79232f726b5bf9c9c10a4bedc6044977e7df'
def node(path,cls,method=None):
 tree=ast.parse((SOURCE/path).read_text());c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 target=c if method is None else next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
 target=copy.deepcopy(target);target.decorator_list=[]
 return target

def execute(target,namespace):
 module=ast.fix_missing_locations(ast.Module(body=[target],type_ignores=[]));exec(compile(module,'<isolated-original-llarp-code>','exec'),namespace);return namespace[target.name]

def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==REV
 torch.manual_seed(0)
 critic=execute(node('llarp/policies/llm_policy.py','LinearCriticHead'),dict(torch=torch,nn=nn))(512,False)
 x=torch.randn(3,512,requires_grad=True);out=critic(x,{})
 assert out.shape==(3,1);loss=(out-torch.ones_like(out)).square().mean();loss.backward()
 assert critic.fc.weight.grad.abs().sum()>0 and x.grad.abs().sum()>0
 returns=execute(node('llarp/policies/transformer_storage.py','TransformerRolloutStorage','compute_returns'),dict(torch=torch))
 buffers={'rewards':torch.tensor([0.,0.,1.]).reshape(3,1,1),'value_preds':torch.zeros(4,1,1),'masks':torch.tensor([1.,1.,1.,0.]).reshape(4,1,1),'returns':torch.zeros(4,1,1)}
 state=SimpleNamespace(buffers=buffers,current_rollout_step_idx=3)
 returns(state,torch.zeros(1,1),True,.99,.95)
 got=buffers['returns'][:3,0,0];expected=torch.tensor([(.99*.95)**2,.99*.95,1.]);assert torch.allclose(got,expected)
 class Invalid:
  @staticmethod
  def _get_uuid():return 'invalid'
 class Progress:
  @staticmethod
  def _get_uuid():return 'progress'
 reward=execute(node('llarp/task/measures.py','SubgoalReward','update_metric'),dict(WasPrevActionInvalid=Invalid,TaskProgressMonitor=Progress))
 def measure(value):return SimpleNamespace(get_metric=lambda:value)
 def run(invalid,progress):
  instance=SimpleNamespace(_metric=0.,invalid_ac_pen=.05,progress_reward_factor=5.,_prev_progress=0.)
  task=SimpleNamespace(measurements=SimpleNamespace(measures={'invalid':measure(invalid),'progress':measure(progress)}))
  reward(instance,task=task);return instance._metric
 invalid_reward=run(True,0.);valid_reward=run(False,0.);progress_reward=run(False,.5)
 assert invalid_reward==valid_reward==0. and progress_reward==2.5
 paths=['llarp/config/baseline/llarp.yaml','llarp/config/baseline/policy/llarp_policy.yaml','llarp/config/task/language_rearrangement.yaml','llarp/policies/llm_policy.py','llarp/policies/action_decoders.py','llarp/policies/utils.py','llarp/policies/vis_bridge.py','llarp/policies/cores/decoder.py','llarp/policies/transformer_storage.py','llarp/trainer/transformer_ppo.py','llarp/task/measures.py','LICENSE']
 evidence=dict(repository='https://github.com/apple-aiml-research/ml-llarp',revision=REV,source_directory=str(SOURCE),source_sha256={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in paths},scope='Pinned source inspection and isolated execution of original AST nodes with minimal dependency stubs; not full integration or performance reproduction',critic=dict(output_shape=list(out.shape),parameters=sum(p.numel() for p in critic.parameters()),learned_value_and_shared_feature_gradient=True),gae=dict(gamma=.99,tau=.95,terminal_only_reward=[0,0,1],earlier_return_targets=got.tolist(),verified=True),reward=dict(invalid_no_progress=invalid_reward,valid_no_progress=valid_reward,half_progress=progress_reward,overwritten_invalid_penalty_confirmed=True),benchmark_reproduced=False,lsdrag_policy_changed=False)
 dest=REPO/'lab/reports/llarp_reference_verification_20261024';dest.mkdir(parents=True,exist_ok=True);(dest/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps({k:v for k,v in evidence.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
