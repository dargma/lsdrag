# H3 state-correction diagnostic

Freeze before outcome computation. One question-balanced training-only diagnostic,
not a new policy candidate or development tuning. Use existing critic72 fit IDs;
exclude all prior monitor IDs, development and reserved evaluation. Select one
error-free measured state per question by SHA256(seed20261082:prefix_sha256),
order questions by SHA256(seed20261082:qid). No selection on quality or cost.

At each original state preserve H1 logits and legal actions. Compare actual H3
FINAL correction with every nonidentity cyclic shift of the72 scalar corrections
(71controls). Marginal correction distribution is unchanged. Compare measured
chosen-branch terminal F1/remaining generation under H3 suffix. Check reconstructed
H3 probabilities against collection records and score raw answers using official
F1 and normalized exact match. Correctness is EM, not positive F1. Report correct
and incorrect token totals and paired four-way correctness savings; partial F1
separate. Aggregate overall costs remain mandatory.

Coverage: at least10 unique questions with an action changed across controls.
Diagnostic positive only if actual H3 weakly dominates control mean on both F1
and generated cost and strictly improves one. If quality/cost tradeoff or low
coverage, do not claim mechanism success. No p-value, no threshold search,
no new reader calls, no independent-generalization or whole-policy claim.
Original +5pp/<=70% goal is unchanged. Result determines the next bounded step.

Command: PYTHONPATH=src:scripts /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/reporting/probe_h3_state_correction.py
