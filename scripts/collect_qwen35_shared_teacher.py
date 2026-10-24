"""Collect fresh 9B train192 states under the frozen common JSON tool contract."""
from lsdrag.decision_qwen35_reader import Qwen35Reader
import collect_decision_shared_teacher as collector
if __name__=='__main__':
    collector.RecoveryPolicyReader=Qwen35Reader
    collector.main()
