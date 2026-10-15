# Budget256 diagnostic interpretation

All20 fixed train questions×3 policies completed. Generated and first-token arms have identical tool/argument sequences and final parsed answers on all20. First-token removes9 generated tokens per question on average, with no QA difference in this block (EM0.05/F10.1105). This is evidence about the current controlled interface, not a learned-head result.

Native reports EM0.05/F10.1115, but valid answer formatting is11/20 vs19/20 for the controlled arms. Many native answers hit the256-token final output limit. Native and controlled cost/quality differences therefore include serialization, truncation, different trajectories and separate argument-prefill costs. Do not attribute the difference to the decision readout alone or claim latency improvements.

Mean prompt tokens are11271.3 native versus53233.75 in both controlled arms; mean tool calls1.95 versus3.6. These are whole-trajectory observations, not matched-state prefill causal effects. The controlled decision and argument prompts both contain the multimodal history. A small head does not remove this repeated backbone work.

The reviewed Bolduc question exposes a concrete reachable-but-unread table: semantic search returned the answer-containing chunk at rank4, then the controller finalized without reading it. Gold was used only in this post-hoc audit. This suggests a read/stop opportunity but no counterfactual success has been established.

A single common1024 output-budget check is running in a separate root/config. No256 teacher features/labels will be used for1024 training. The budget check uses AgentConfig's existing default rather than a per-question or best-score sweep. Final stage1 interpretation awaits that check.
