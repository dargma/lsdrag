# Offline structure continuation result — 2026-10-24

576 episodes/64 shared development questions complete; 626.24sec,33679generated,1792143input. All9policies same reader/native protocol/corpus. Original d987 reference exactly reproduces64answers/support/F1/tokens/calls/errors. Drive evaluation archive independently rehashed. No reserved-test access.

| Policy | F1 (%) | Generated | Zero-tool /64 |
|---|---:|---:|---:|
| basic_grpo | 22.50 | 98.41 | 0 |
| native | 24.70 | 97.44 | 1 |
| selected_reference | 25.18 | 76.75 | 18 |
| control | 16.55 | 50.36 | 37 |
| depth | 16.55 | 58.61 | 36 |
| input_gate | 16.55 | 58.94 | 36 |
| depth_input_gate | 16.55 | 59.64 | 35 |
| observation | 16.55 | 58.98 | 36 |
| question_attention | 18.37 | 51.48 | 38 |
| action_attention | 17.72 | 62.91 | 37 |
| action_set | 16.55 | 48.56 | 38 |

## Interpretation

All continuation candidates fail joint goal and lose quality versus selected_reference. Same-architecture control drops8.63F1pp; reference-minus-control paired95%CI[1.56,16.70]pp. Thus the shared continuation procedure itself causes a clear development regression; cannot attribute all harm to added architecture. Zero-tool count rises18→35–38. Trainingloss and cachedQ improve while closed-loop quality falls: in-sample loss is unsuitable as sole selection criterion. Old-policyQ, furtheroptimization and weaker behavioral retention are plausible causes; this experiment does not isolate them.

Question attention improves1.82pp over continuationcontrol on2questions, marginalCI[0,4.6875]pp, still below original. Actionattention improves1.17pp overcontrol but increases tokens24.91%. Other four variants plusactionset have exactlysame per-questionF1 ascontrol, despite differing tool/cost traces. No attention success or intrinsic architecture failure claim. Single seed, exposedvalidation, multiplecomparisons.

## Next bounded experiment

Keep originald987 as incumbent. Test adapter-only offline optimization: freeze allbaseparameters and normalization, keep residual addedpaths trainable; compare depth andquestion/actionattention with same unchangedQ andKL first. This isolates destructive baseupdates from new-path learning without layering newrewardchanges. Preservezero-init exactbasepolicy and reportKL/argmaxdrift. Input_gate/observation/set variants are deferred pending this comparison. Do not select an early epoch retrospectively. Freeze exactprotocol beforefit. No PPO/GRPO/worldmodel, no changednativeformat, no freshtest untilcredibledevelopmentcandidate. Goal remains ACTIVE.
