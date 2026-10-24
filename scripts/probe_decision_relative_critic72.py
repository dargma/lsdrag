"""Use frozen critic72 execution with a single relative-quality objective change."""
import probe_decision_future_critic72 as experiment
from lsdrag.decision_relative_critic import relative_quality_loss
experiment.NAME='decision_relative_critic72_20261024'
experiment.masked_loss=relative_quality_loss
if __name__=='__main__':experiment.main()
