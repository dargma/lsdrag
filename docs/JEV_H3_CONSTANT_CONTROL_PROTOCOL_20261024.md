# H3 conditional residual versus a constant residual

User-approved diagnostic before further quality training. Freeze H3 checkpoint
936e4592. The control keeps identical H1 base/tool logits, sets the129parameter
residual weight vector to zero and uses one constant FINAL bias. H1 still uses
state: this isolates the added residual, not all state-conditioned decisions.

Train-only token calibration: use verified H1 paired prefixes (exclude errored
pairs) to propose at most3 scalar offsets. Enumerate intervals between H1
stop-margin thresholds. Choose offsets whose weighted suffix-generation proxy
is nearest H3's proxy and .85/1.15 times that target; ties prefer |bias| then bias.
No F1 enters proposal or selection. These surrogate costs are not full episodes.

Freeze24of the48training qids by SHA256(20261071:qid). Actually execute H3 and
all proposed constants on those24, at most96episodes/1800seconds. Select the
single constant with smallest absolute mean-generated difference from H3;
tie smallest |bias| then name. Do not inspect F1 to choose. Record relative cost
mismatch; tolerance10% is an interpretive gate, not an outcome threshold to tune.
No additional scalar search after this bounded calibration.

Freeze chosen checkpoint before paired development64×H3/control=128episodes,
1800seconds. Use unchanged native interface/frozen model/splits. Report paired
F1 and generation plus whether both train and dev generation ratios are within
[.9,1.1]. If not, same-budget comparison is inconclusive; no dev retuning. A
positive F1 point difference alone is suggestive, not independent proof; include
paired uncertainty and whether useful positive-answer savings survive.

This is one mechanistic comparison, not a policy grid on dev. It does not make
H3 meet the original joint goal (currently quality threshold missed). No new
H4 training or heldout80 use. Source/input hashes and raw episodes preserved,
Drive archives verified, code/memory pushed jev_v1.1.
