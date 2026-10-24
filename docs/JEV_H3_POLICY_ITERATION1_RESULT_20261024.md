# Current-H3 multi-turn action update1: negative QA result

The first all-action policy improvement from H3 terminal outcomes is complete.
Preserve H3 as reference; do NOT promote this candidate or claim goal progress
from its lower token count alone.

| Policy | F1 (%) | Mean generated | Mean input | Tool calls |
|---|---:|---:|---:|---:|
| Fixed basic GRPO |22.50|98.41|4625.31|2.328|
| H3 |27.32|66.89|3417.92|1.422|
| Current-policy update1 |21.23|61.44|3535.75|1.438|

Candidate versus H3: F1−6.0904pp; marginal paired95%CI[−12.1272,−1.4027]pp.
Generated−8.15%; ratioCI[.7607,1.0809]. Development64 exposed/adaptive/single
seed; no independent generalization claim. One partial improvement versus7
quality losses. Of349net tokens saved,271are on quality-loss questions,63on
both-zero questions,83on positive-retained questions,offset by68extra tokens
on the one quality gain. Input tokens and tool calls did not decrease. This
is not useful joint accuracy/cost improvement.

## Actual learning method

Current policy pi_k is H3, including its conditional FINAL residual. Reused
fixed24train rollins; all64nonforcedstates,all231legal action branches. Each
intervention is followed by H3greedy to terminal answer, yielding measured
Q_F1^H3(s,a) and C_suffix^H3(s,a), not immediate retrieval proxy rewards.
314.46seconds/15568generated/1164617input/0errors. Raw F1, parent prefixes,
original-action terminal answers, later greedy choices,legal coverage and
Drive bytes independently verified.

Targets maximize question-balanced terminal F1 under aggregate suffix cost
no larger than H3reference. Each state's allowed alternative must have F1>=
reference; zero-to-zero cheaper actions excluded, any errored state retained.
Tie-break mincost then minchanges. This permits useful extra future work
funded by quality-preserving savings. It produced13changedstates/9questions:
8qualitygains/5ties,including non-FINALtool switches and costly continuation.
Weighted overlapping-suffix surrogate F1.302976→.414087 and cost48.6749→48.6068.
These are hindsight training targets, NOT learned-policy episode performance.

Only H3's existing128->4outputlinear layer (516parameters) was updated.
Frozen: reader,representation,normalization,H3FINALresidual. 100fullbatch
AdamW updates/.001/defaultweight_decay.01,clipnorm1,seed20261073. Correction
soft target=.5H3probabilities+.5targetonehot; currentH3 probabilities on519
retentionstates. Each group question-balanced,group weights.5/.5. No epoch
or coefficient selection on dev. Checkpoint ce0ced9b1f0b4ab0c8f151d35f0916f8581dad8a0409cd11974b39ff16705361.

## What changed and what this does not establish

The output layer learned9of13target actions (0before), but changed7of51fresh
states without correction targets. Total16of64trainingstate actions changed.
Training action-frequency totals barely changed,so this is not evidence of
only a uniform class bias. The smallhead can change actions; generalizing
those changes safely and maintaining useful paths is not established.

Per-state quality constraints apply to sampled actions followed by OLD H3,
not to multiple changes composed along a new policy's trajectory. They cannot
guarantee the updated policy's true outcome. Sparse24q/13corrections,feature
limits,update spillover and changed visitation are plausible contributors;
this one comparison does not identify a unique cause or show frozenLLMs
cannot support long-horizon policies. Do not hide failure or launch a dev
learning-rate sweep to rescue it.

Next hypothesis: broaden current-H3 terminal-outcome coverage and constrain
unjustified policy movement on training states before composing updates.
Keep H3 as starting policy. One bounded extension and conservative update,
not multiple architecture/reward grids. Exact next collection/update protocol
notyetfrozen; no nextworker. Retain frozenreader until evidence warrants
representation adaptation. Goal unchanged/ACTIVE; reserved80untouched.

## Verification and storage

Paired128episodes/175.08seconds/8213generated/445035input/0errors. H3reference
replay64exact. Independent rawrescoring and byte-verified Drive archive passed.
Only changed checkpointkeys are base.net.2.weight and base.net.2.bias.
Report directory lab/reports/decision_h3_policy_iteration1_validation64_20261024/.
ArchiveSHA c3ef54f484b28ffb4c34d23c62344a0b43fa4d6bda755c9af1ac55db58ee698e.
Coordinator35979/evaluator36050complete andabsent. No final80/reserved80usage.
