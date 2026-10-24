# H3 as an outcome-trained decision policy

User asks whether H3 already plays the intended role. Yes, within a restricted
policy class: a state-conditioned stopping policy trained from terminal answer
quality and remaining generated cost under H1 continuation. A learned critic,
unfrozen reader, or explicit inference-time planning is not required by that
operational definition. H3 does not establish general four-action multi-turn
optimization or literal simultaneous answer generation and action prediction.

The original goal remains F1 +5 percentage points and generated tokens <=70%
of fixed basic GRPO, jointly, with useful state-conditioned behavior and protected
independent confirmation. Do not lower that goal to architectural plausibility.

The whole H3 versus GRPO comparison includes the inherited H1 policy. Attribute
the incremental FINAL correction using H3 versus H1, not the full GRPO gap.
The constant-offset experiment did not match development cost within its frozen
tolerance; state-dependent incremental superiority remains unresolved.

Reproducible existing-episode audit:

```
/content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/reporting/audit_h3_policy_role.py
```

Output: lab/reports/decision_h3_role_audit_20261024/audit.json. Official answer
rescoring and raw generation counts; exact question/model/corpus correspondence;
checkpoint hashes checked locally or against the hash-verified Drive archive.
No new inference, no change to frozen splits. These are repeatedly exposed
64 development questions. Grouped token savings are net sums and positive F1
is not synonymous with a fully correct answer.

Next hypothesis, before another architecture or critic candidate: does the
existing H3 conditional stopping correction exploit useful state information?
Use the existing training-only measured branch outcomes to compare actual H3
corrections with shuffled state corrections, holding H1 logits and legal actions
at the original state. Freeze a single diagnostic protocol before execution;
report intervention coverage and terminal quality/cost under the measured H3
suffix, not whole-policy performance. This is a diagnostic of the existing
policy, not a success criterion replacing the goal. It cannot establish unseen
transfer because H3 training questions overlap. Do not use development labels
to tune a replacement or launch reserved80. If evidence is promising, freeze
one quality-preserving improvement and its full rollout gate separately.

The previously proposed two-action probe is deferred by this H3 refocus, not
completed or discarded. No automatic launch follows merely from this document.

Verified result: H3 versus GRPO F1 +4.81517pp, generated ratio0.679740,
7wins/5losses. Net2017saved tokens:1597bothzero,270qualityloss,
194positive-retained,−44qualitygain. H3 versus H1:3wins/4losses,
541saved (305bothzero,143loss,60positive-retained,33gain). The positive
quality groups demonstrate some useful savings, not dominance of that mechanism.
All192episodes error-free. Goal not achieved; independent confirmation absent.
