import torch
from lsdrag.decision_outcome_rerank import OutcomeRerankHead
from evaluate_qwen35_h3 import load_head as original_load
from lsdrag.decision_qwen35_reader import Qwen35Reader
import evaluate_decision_shared_corpus as evaluator
def load_head(spec,device='cpu'):
 if spec['kind']!='outcome_rerank':return original_load(spec,device)
 h=OutcomeRerankHead();h.load_state_dict(torch.load(spec['checkpoint'],weights_only=True,map_location='cpu'));return h.to(device).eval()
if __name__=='__main__':
 evaluator.RecoveryPolicyReader=Qwen35Reader;evaluator.load_head=load_head;evaluator.main()
