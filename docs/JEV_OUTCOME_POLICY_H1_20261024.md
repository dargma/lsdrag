# H1: outcome-supported MLP policy improvement

Hypothesis: training the current MLP on supported alternative-action outcomes
improves actual QA and allocates generation more usefully, rather than merely
imitating the reference or rewarding failed early exits.

One candidate, unchanged head/body/native interface. Start from cost-priority
MLP e1a630e5. Use completed/audited third48:79states/584episodes. Ten correction
states across nine training questions pass the existing two-repeat evidence
rule. Favor the largest repeat-consistent F1 improvement; among equal gains
favor token savings. If there is no strict gain, positive-F1 nonworse cheaper
actions qualify. Zero-F1 cheaper actions and errored pairs never qualify.
This changes the previous cost-first preference explicitly. Two samples provide
no statistical guarantee. Frozen exact targets and input/source hashes are in
`lab/registry/decision_outcome_policy_h1_20261024/plan.json`.

Keep the existing inexpensive fitting settings:100 CPU updates, AdamW1e-4,
seed20261066, target .5 reference distribution + .5 chosen action; equal
correction/retention group weight and question balancing. Retain current-head
behavior on original377 plus first/second48 features and uncorrected fresh
states, deduplicating public prefixes and removing conflicting corrected
prefixes. Old suffix outcomes are not mixed into new outcome targets.
Normalization/LLM remain frozen. No epoch/lambda/architecture sweep.

Evaluate current reference and the single candidate on unchanged development64
(128episodes), same model, scorer, corpus and decoding. Compare cached basic
GRPO/native and record exact reference replay. Evaluation wall cap1800seconds,
generated800k/input30M; fit cap120seconds. Shared GPU timings are descriptive.
Training run `decision_outcome_policy_h1_20261024`; evaluation run
`decision_outcome_policy_h1_validation64_20261024`.

Judge paired F1/tokens and whether changed actions recover answers or save
tokens while retaining correct answers. Preserve wrong/wrong savings separately.
No goal success from development results; reserved80 stays protected. A positive
direction leads to confirmation/expansion, a negative direction selects the
next large hypothesis from the active hypothesis plan. No detailed ablation
matrix or additional candidate launches from this protocol.
