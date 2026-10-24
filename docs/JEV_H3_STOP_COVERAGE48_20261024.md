# Expand H3 stopping coverage by48 training questions

Hypothesis: constraints on72 training questions fit useful decisions but transfer
poorly because the measured stopping states cover too little of the fixed task.
Do not select new questions from development failures. SHA256(20261084:+qid)
first48 from frozen train192 minus all96 previously H3-outcome-collected questions.
These questions are new to H3 branch collection, not necessarily unseen by H1
or earlier learning. Existing24 monitor exclusions remain unchanged.

Same frozen H3 checkpoint, Qwen3-4B, shared101962 corpus, native interface. At all
nonforced H3 rollin decisions, collect exactly two greedy interventions: FINAL
and highest-probability legal non-FINAL action; subsequent decisions use H3.
One rollout each, including replay of the chosen H3 action. Full legal mask stays
four-action; sampled actions explicitly binary. No labels from missing branches.
48collectors +2*state_count, upper816episodes,1800seconds, existing1Mgenerated/
60Minput caps. Stop on a real error/budget limit; preserve partial artifacts.

Freeze selection/source/input hashes before execution. Verify raw answers with
MuSiQue F1/EM, prefix hashes, action coverage, H3 suffix decisions, reference
answer replay and actual counts. Archive and verify Drive bytes. Correct/incorrect
collection cost is workload, not policy performance because suffixes overlap.

After successful audit, same constrained129parameter fit as72: union old72fit
and new48=120; unchanged24monitor exclusions. Same target logic, hard retention
constraints, KL.01, objective and solver; no hyperparameter search. Freeze final
input hash before fit. Learning gate unchanged. If it passes, full pairedtrain120
with correct/incorrect reporting and prior strict train gate, then at most one
fixed development64 comparison. No automatic independent evaluation: reserved80
requires separately frozen protocol. Original +5pp/<=70% goal is unchanged.
