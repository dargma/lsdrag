"""Reuse verified H3 collector with a separately frozen disjoint training split."""
import run_decision_h3_policy_coverage24b_20261024 as collector
NAME='decision_h3_diversity48_20261024'
collector.REG=collector.REPO/'lab/registry'/NAME
collector.OUT=collector.ROOT/'results'/NAME
if __name__=='__main__':collector.main()
