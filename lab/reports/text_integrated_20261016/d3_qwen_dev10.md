# Text-only D3-Flash vs non-thinking Qwen4B: initial diagnostic

Qwen3-4B-Instruct-2507 (4.0B, non-thinking-only) is the selected reader/backbone. Official model card: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 . Pinned Qwen revision cdbee75f17c01a7cc42f958dc650907174af0554; D3 revision8b35c853b90e777595ec36bacae820fe08ea077a.

MuSiQue-Ans2-hop, fixed remaining dev10, same candidate builder, same Qwen argument/final generator, same text-only paragraph pool and limits. Qwen first-token choice is a zero-shot controller baseline, not native tool generation or a trained head. D3 is an external pretrained choice controller. No images or thinking mode.

| Condition | Answer F1 | EM | Valid final JSON | Episodes with errors |
|---|---:|---:|---:|---:|
| Qwen first-token choice | 46.67% | 40% | 10/10 | 0/10 |
| D3-Flash choice + Qwen reader | 30.00% | 30% | 5/10 | 5/10 |

D3 chose new keyword generation in cases where Qwen returned a function wrapper or final-answer object instead of the requested argument-only object. All5 failures are saved; no repair/retry/omission. The current argument request names the selected tool and schema but lacks the older adapter's explicit prohibition of function-name/tool-call wrappers. This is a concrete interface weakness to harden on TRAIN-only cases before collecting head labels. The comparison does not isolate decision quality from conditional argument realization failure. Lower D3 tool/time/token totals partly reflect early failures and are NOT efficiency success.

Paired D3-minus-Qwen F1 mean -0.1667, exploratory question-bootstrap95% interval [-0.4, 0.0]. n10 and a single setting do not establish generalization or statistical superiority. No controlled speed comparison was performed; shared GPU and mixed FLA/reference convolution kernels are disclosed in d3_runtime.json.

Prompt audit: Gemma enable_thinking=False was actually passed and rendered without the activation token; one spontaneous thought-channel output was left in content by the Gemma parser, and a conditional FINAL-format instruction was ambiguous. Preserve the old2×2 Gemma/native result as format-confounded diagnostic only. New Qwen template is non-thinking-only; v2 instruction unconditionally requires final JSON and shortest answer span. Train2 preflight4/4 episodes passed formatting before dev10. Both previously inspected dev2 were excluded, not chosen according to success.

Official MuSiQue scoring source is pinned to922ac98f19a201998dbdae6d7f2887a5258dbdeb. The official public Drive ZIP hit quota; two public mirrors' train/dev SHA256 match. Exact inaccessible official ZIP equivalence is unverified. Runtime excludes answers/aliases/support flags/decomposition from controller inputs.

All runtime/model/data/code paths and resolved weight targets are local-fast. Drive is backup only. Total worker time across the format diagnostic, Qwen preflight and this dev10 run was 147.2s, below60min; this is resource accounting, not a speed benchmark.

Artifacts: /content/local_fast/lsdrag-text-v1/results/d3_qwen_dev10_20261016 . No current benchmark worker. No fresh teacher32/48 collection, action-head/arg-head learning or learned-policy evaluation has completed. Next: fix argument contract using train-only tests, freeze a new untouched small dev subset before new quality evaluation.
