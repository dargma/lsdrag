Latest override: user now explicitly permits immediate shared-GPU execution when memory allows;20minute idle monitor cancelled. User replaced previous goal with “큰 모델로 실험 완료하기”. Success means completing and verifying the9Bexperiment, not forcing old+5pp/.70 success. Historical fixed4Bbaseline is retained for context, not the new completion criterion.

# H3 method generalization to Qwen3.5-9B

User renewed authorization after the final4B experiment: download current official9B Qwen and retrain, with H3 generalization and model-dependent tendencies as the purpose. Latest hardware override: start GPU work only after20 uninterrupted minutes of observed GPU idle. Earlier shared-GPU permission is superseded.

## Model and interface

Official Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a; four BF16 shards19,306,310,880bytes downloaded into local fast and matched official LFS SHA256. Text backbone4096hidden versus prior Qwen3-4B-Instruct-2507's2560; no4Bhead weights, normalization or rollout outcomes may be treated as9B training data. Freeze reader; head input dimension adapts,128hiddenMLP and129parameter H3 stop correction remain the method.

Qwen3.5's official tool template emits XML, whereas established A-RAG contract is JSON. Use the frozen prior JSON template with9Btokenizer and official empty closed nonthinking suffix. This is a common-protocol9B baseline, not unmodified native XML behavior. All9Bconditions share it, plus identical schemas, handlers, full shared corpus, context16384, output512 and8tool-step limit. Count the empty thinking block as input; never hide generated reasoning. Image inputs prohibited.

## Scientific comparison

Primary: within each backbone, base outcome-trained policy versus H3 correction under identical inference and training split. Compare H3-minus-base F1/EM and token ratio, correct/incorrect/partial costs, paired correctness, stopping time and tool distributions. Also report model-native tokenizer totals and output character/byte counts: tokenizers differ, so cross-model token counts are not identical units. Shared-GPU time is descriptive only.

Historical4B H3/H1 is a reference;9B requires fresh base-policy preparation and fresh continuation outcomes. Do not claim exact replication of the entire exploratory4B H1 training history. Freeze9Bbase before H3 target collection, use the same H3 stopping optimization/fit rules, and report preparation differences. Model generation, size, tokenizer and attention architecture change together; two backbones cannot isolate a causal scale effect. No larger model grid is authorized by this plan.

Original immutable4BGRPO goal remains a historical fixed benchmark and must not be replaced silently.9Bpolicy efficacy requires a same9Bcontrol; improved9BQA alone does not establish H3 generalization. Frozen train192/development64 preserved; reserved80 and prior final80 are not consumed.

## Current bounded stages

1. Download verified. Initial artificial prompt smoke passed final Paris answer but keyword arguments failed with repeated zeros/top_k truncation. Preserve failure; no policy or generalization result.
2. CPU-only monitor waits1200continuous seconds, sampled15seconds: GPU utilization0, no compute process, >=40GiB free. Any activity/telemetry error resets timer. No GPU loading before gate. It launches one revised smoke using actual A-RAG prompt and512output limit; no retry loop.
3. Only if all FINAL/keyword/semantic/read smoke checks pass, collect fresh9Bnative-common-protocol teacher trajectories on the existing train192. Snapshot/data/source hashes frozen in teacher_spec.json. Stop on command failure. This stage is bootstrap data, not H3 training or quality evidence.
4. After verified teacher coverage, freeze fresh9Bbase head training and outcome collection, then originalH3 stopping target/fit and paired evaluation. No development labels to select hyperparameters. These stages are not silently substituted with old4B features or pretrained weights.

Monitor launch registry lab/registry/qwen35_9b_h3_20261024/idle_launch.json; live status /content/local_fast/lsdrag-text-v1/results/qwen35_9b_idle_gate_20261024/status.json. Do not duplicate the monitor; exclusive file lock protects it. Inspect actual PID/child and status before recovery.

## Wrapper-aware9B decoder correction

Initial two smokes failed because token3307 jointly closes argument and outer tool-call braces. The argument-only mask rejected the valid native wrapper token and selected zero repeatedly. New adapter accepts schema-valid first argument object plus only the already-open outer brace and optional known closing-tag prefix. Every emitted token is charged; raw generated text and consumed wrapper tail are recorded. Same handling for native-action baseline and head-selected actions. No numeric clamps/answer repair.

## Bounded comparisons

Fresh9Bbase policy versus its H3 correction is primary; state-independent stop bias trained/calibrated only on train is the mechanism control. Report historical4BH1→H3 direction alongside9Bbase→H3, explicitly accounting for base preparation differences. Evaluate all on frozen development64 once, without dev tuning. Prefer this single mechanistic comparison over multiple architecture/ablation variants.

## Frozen sequential implementation

Coordinator scripts/continue_qwen35_h3.py waits for the confirmed teacher PID and requires completed artifacts, then executes a finite pipeline. Source snapshot is pipeline_sources.json. Each dependent stage spec is written once with the actual checkpoint hash before GPU execution. GPU phases check >=40GiB free; other jobs are never terminated. Failed source, coverage, verification, optimization or subprocess stops the coordinator; no automatic retries or hyperparameter variants.

Fresh bootstrap: train192,4096x3 inputs ->128tanh ->4 actions,100AdamW3e-4,seed20261051,.05label smoothing, min5questions/action. Base outcome stage uses ALL nonforced prefixes from fresh bootstrap rollins on the exact historicalH3train48, each legal first action then bootstrap continuation. A single quality-first preference fit requires F1/EM nondecrease and either F1gain or cheaper positive-F1 output;100AdamW1e-4,seed20261066,.5reference-target blend plus equal-group teacher/fresh retention. This is fresh9Bbase preparation, not an identical replay of historical4BH1search.

Stop stage makes fresh rollins with that new9Bbase and evaluates FINAL versus best legal continuation at ALL controllable prefixes on the same48. Original budget_targets and original129parameter H3 fitting rules apply; same targets/retention train a bias-only control with one parameter. Both require >=3changed training questions; otherwise report coverage/feasibility failure without retuning. Source tensor, official scores, prefix identity, state dimensions, sample coverage and rawtoken sums are verified before fitting.

Evaluation: one common-protocol base/H3/constant paired64 run (192episodes), no dev selection. Verify officialF1/EM, frozen sources/checkpoints/splits, exact tokenIDs counts including wrapper-crossing tokens, same-prefix first divergences involving FINAL, correctness-separated costs and pairedgroups. Generation byte/character counts use actual emitted tokenIDs, excluding forced input prefixes. SharedGPU time is descriptive. Archive every completed stage, synthesize4B/9Btrend with stated confounds, then audit objective completion; pipeline terminalstatus alone is not a goal-completion claim.
