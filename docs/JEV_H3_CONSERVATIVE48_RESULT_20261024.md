# H3 coverage48 conservative update: no useful action change within training limits

Current run is complete; no GPU worker/evaluation is running. Keep H3 incumbent.
The fixed pre-evaluation gate rejected the proposal as useful learning progress;
there is no new development F1 and no goal-success claim.

## Verified evidence

New disjoint24train H3rollins/allaction suffixes complete253episodes/64states,
323.23seconds,15664generated/992700input,1error preserved. Independent rescoring,
original-action finalanswer replay, full dynamiccoverage, latergreedyactions,
training exclusion and Drivebytes passed. Combined first24+new24 gives48q/
128states,484executedepisodes (includes24newrollins;460actionbranch episodes).
The two sets use the same H3rollin/suffix and frozen generation contract.

Same categorical target objective found30corrections/20questions. Weighted
suffix F1.381812→.479266/cost46.2256→46.2208 is a hindsight training surrogate,
not QA performance. Single516parameter final-layer proposal,100updates/.001/
seed20261075,30correction+507retentionstates,reader/representation/H3stop fixed.

| Proposal fraction alpha | Question-balanced KL | Unsupported state action changes |
|---|---:|---:|
|1|.089381|12.82%|
|1/2|.017303|4.73%|
|1/4|.003903|2.17%|
|1/8|.000933|1.78%|

Frozen limits were KL<=.01 and unsupportedchange<=2%. Alpha1/8 is the largest
passing step. Independently recomputed:9of507unsupportedstates changed,0of30
supported target actions selected. Interpolation exactly matches raw proposal
and H3; all other parameters unchanged. Thus movement constraint passes but
learning-progress gate fails. Per protocol no paireddev evaluation was launched
and no dev-based alpha/threshold tuning occurred. Candidate325abfbf is an
archived diagnostic artifact, not a new incumbent.

This establishes a practical limitation of this particular output-layer update:
with this data/representation/loss, the tested proposal cannot both respect the
training preservation bound and select a supported improved action. It does
not prove all frozen-reader heads fail, nor that an accurate critic would solve
it. No-op evaluation would not advance the research claim.

Next direction must address state discrimination or the credit signal rather
than repeat output-layer coefficients. A bounded test of a richer state-conditioned
head on the SAME frozen reader and current-policy data is a possible next step;
no new representation/critic protocol has yet been frozen or launched. Preserve
terminal-outcome policy-improvement objective and H3reference; avoid a new grid.

## Reward/critic clarification requested by user

Current actor is trained from measured terminal outcomes, not a learned critic.
StopValueHead is a logit residual; its returned value is zero. For each sampled
initial/intermediate/final-controllable state, an action intervention is followed
to completion. The quality target is finalanswerF1; the cost target is remaining
generatedtokens from that state onward. These two quantities are used separately
in constrained target selection, not an implemented scalarlambda reward.

Accuracy feedback is terminal; intermediate decisions receive credit through
their own counterfactual terminal comparisons. There is no separate immediate
quality/progress reward at each turn, and no learned V(s)/Q(s,a) estimator.
This is not merely training the last action, nor does it guarantee credit remains
valid after multiple updated actions compose on a new trajectory.

## Storage

Fit/proposal/accepted diagnostic: results/decision_h3_conservative48_20261024.
Accepted diagnosticSHA325abfbfef4f23da4ba9eb586c91df808adacb1372f2a1eb5eb40e6bf4057a49.
Drive fitarchiveSHA099d19317aa976e9c7339f887dae504ddbf3dff9b8cddf981310a3fc8e9832d8.
NewcollectionarchiveSHAb10358df5d243ecfc7b8cc4b28ca4cedc5389380b28a02425efbbc79f43e9d95.
Gate verification lab/reports/decision_h3_conservative48_20261024/gate_verification.json.
GoalACTIVE; no reserved80 access, no newdev exposure in this phase.
