# Decision quality through short imagined tool outcomes: research synthesis

2026-10-15 UTC. Research proposal, not a frozen experiment or a demonstrated method. The former three-action measurements are discarded. The new full-tool A-RAG implementation is a baseline infrastructure change; its functional validation does not establish policy quality or novelty.

## Current judgment

The current direct head alone is insufficient as an ICML contribution. The stronger question is whether a small action-conditioned predictive model can improve **which action an agent chooses**, while explaining the relevant failure modes and transferring its learning principle beyond retrieval. Original A-RAG is a strong reference or distillation teacher, not a mathematical accuracy upper bound. Models can discover better actions than a teacher, or lose its reasoning capabilities.

The user's latest formulation is **predict before acting, compare imagined downstream outcomes, execute one action, then correct with actual observations**. It is not merely a post-hoc novelty detector. Do not combine the old access auxiliary, world modeling, curiosity, memory and adaptive vision into one method before determining which hypothesis works.

## What is already known

| Closest research | What it already covers | Consequence for us |
|---|---|---|
| [LatentGate, ACL Industry 2026](https://aclanthology.org/2026.acl-industry.153/) | Frozen-backbone non-generative routing | A small routing head is not the scientific novelty |
| [SII-Decision / JEVal, v2 2026-10-12](https://arxiv.org/html/2610.03935v2) | First-token decision models; local accuracy can fail to yield reliable long-horizon agents | Compare actual closed-loop success, not just local classification or median speed |
| [RAP](https://arxiv.org/abs/2305.14992), [LATS, ICML 2024](https://arxiv.org/abs/2310.04406), [DynaWeb](https://arxiv.org/abs/2601.22149) | World-model/search-based reasoning or agents | Imagining tool consequences and choosing actions is not new by itself |
| [VaGraM](https://arxiv.org/abs/2204.01464), [DRPE, September 2026](https://arxiv.org/abs/2609.32322) | Value-aware model errors and decision-relevant prediction errors | Neither lower total prediction error nor task-weighting it is an original claim alone |
| [Agent-Editing World Model, September 2026](https://arxiv.org/abs/2609.28416) | Agent task-progress/state modeling rather than reconstructing high-entropy tool responses | Predicting entire tool text or pixels may spend capacity on irrelevant details |
| [InSight-doc](https://arxiv.org/abs/2608.10628), [MAGE-RAG](https://arxiv.org/abs/2606.15906) | Learned visual resolution/zoom and adaptive evidence granularity | Selective image opening, zooming and evidence control already have close precedents |

The potential contribution must therefore be a specific learning/planning mechanism with controlled evidence, not the conjunction of these terms. Originality is unresolved. Latest preprints are primary-source claims, not independently reproduced results.

## One candidate to falsify

Given current state s, legal tool a and arguments u, a small model predicts a latent next observation/state. A short lookahead compares expected terminal task utility minus future resource costs. One actual call is dispatched; its observation supplies a residual for subsequent predictions and choices. The frozen VLM remains the reader/argument generator.

```text
observed state + legal tool/argument candidates
                 ↓
       predicted next latent states
                 ↓
       depth 0 / 1 / 2 value comparison
                 ↓
          one real tool execution
                 ↓
 actual observation → prediction residual → next iteration
```

Three distinctions matter:

1. A scalar return predictor alone is a generic Q head, not a world model. To justify the predictive state, it must support useful lookahead or correction that a capacity/data-matched direct value model cannot reproduce.
2. Raw surprise is not task utility. High-entropy irrelevant images can be surprising but useless; predictable evidence can be essential. Surprise, parameter uncertainty, information gain and return improvement require different estimators.
3. Actual surprise is available **after** a call. It cannot justify avoiding the expensive observation already paid for. Before opening an image, use only legal cheap inputs to predict expected benefit; afterward, surprise can change the next search/read/refresh/stop decision.

Arguments cannot be free hidden oracle inputs. Predictions must condition on actual proposed arguments, or explicitly average over a common argument generator. Generating candidate queries/arguments for every tool has a cost. A shared candidate pool and the same argument policy are necessary across baselines. Candidate selection, image fidelity or a different corpus changes the task kernel and requires fresh labels/contracts.

Depth-2 future tool IDs, arguments and legality cannot come from search results that were never actually observed. Constructing them from an imagined state is an unresolved design question, not an implemented capability. Start with one-step prediction; do not silently use actual future branches as online imagination inputs.

Heavy-image motivation is concrete: choose whether another visual inspection can change the next action or final success. However, the current head reads the full multimodal prefix **after vision encoding**. It cannot avoid that already-incurred cost. A future acquisition policy must act before expensive new images are inserted, using cached current state/cheap metadata. Retriever vectors and reader vision features are different representations; replacing one with the other is not an equivalent optimization.

## Minimal comparison and mechanism evidence

The first study should compare a common full-tool environment with identical frozen reader/retriever, argument policy, legal candidates, training branches and budgets:

- Generated A-RAG adaptation and constrained first-token selection.
- Behavior-cloning head and a strong generic Q/value head.
- Capacity-matched larger Q head, to rule out merely adding parameters.
- Predictive model with depth 1 and depth 2; depth 0 is the direct-value reference.
- Shuffled transitions, raw-surprise/uncertainty controls, and an actual-next-state diagnostic oracle.

The actual-next-state oracle is a diagnostic reference within fixed candidates and realized branches, not a global optimal policy or an expected-return upper bound under stochastic execution.

Compare task success/QA, observed branch regret, premature stopping, bad continuations and budget–quality curves. A predicted state is useful only if it improves real action rankings or recovery. Report prediction error on relevant and irrelevant dimensions, calibration and how imagined depth affects errors. Lower prediction loss with unchanged/worse closed-loop choices falsifies the intended benefit.

For depth 0/1/2 comparisons, match the return-label horizon, terminal value target, continuation policy and cost definition; vary planning depth rather than secretly supplying a longer teacher return only to the world model. Distinguish the generated controller under a common argument pool from original A-RAG with free argument generation: the former isolates selection, while the latter is a separate end-to-end system reference.

Use predictable-but-useful and surprising-but-irrelevant evidence; examine direct, one-bridge and two-bridge strata. For a mechanism claim, intervene on observations or prediction fidelity and check actual action/return changes. A generated explanation or attention map is insufficient.

Match offline teacher/rollout budget as well as online budget. If the old 2×2 access supervision is revisited, include a baseline with the same raw cell outcomes: a linear recombination of richer labels must not be mistaken for new information. Do not require all these controls in a large benchmark before a small predictive-state signal exists.

## Benchmark selection by claim

| Role | Recommended candidate | What it establishes and what it does not |
|---|---|---|
| Main multimodal QA, choose one | [M3DocVQA](https://m3docrag.github.io/) first for existing infrastructure; [ViDoSeek](https://huggingface.co/datasets/autumncc/ViDoSeek) if pixel-sensitive high-resolution decisions are insufficient | Actual QA and evidence acquisition. Audit whether pixels/high resolution are necessary; a PDF screenshot alone does not establish visual reasoning |
| Mechanism diagnostic | [VisDocAgentBench](https://hulx2002.github.io/VisDocAgentBench/) | Direct/one-bridge/two-bridge retrieval under a common output contract. Only 120 queries; ranking, not QA or a sole training/final evaluation set |
| Optional visual retriever check | [ViDoRe v3 official agentic pipeline](https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/submissions/vidore_v3_agentic_pipeline.md) | Retrieval quality and iterative candidate examination; does not replace end-task QA |
| Beyond-RAG transfer | [BFCL multi-turn local stateful API tasks](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) as the first feasibility candidate | Actual tool-state success and decision quality. New training tasks/splits are needed; benchmark evaluation trajectories must not become training data. No claim of transferring image-cost savings |

Use one main QA environment and one later non-RAG environment, with a small mechanism diagnostic. Do not simultaneously build every listed benchmark. Fixed six-class heads cannot demonstrate zero-shot transfer to unseen tools: use a declared new head or a genuinely schema-conditioned model, with adaptation cost reported. The original query-generator × answer-reader 2×2 intervention does not automatically carry over to stateful APIs.

## Practical next decision

First verify real action alternatives offer improvement under the restored full-tool A-RAG protocol. Then test whether one-step latent prediction improves action ranking over generic Q with the same data and capacity. Only after that should depth-2 planning or a second environment be expanded. A single 96GB GPU is compatible with frozen-backbone feature collection and small-head training; repeated branch rollouts, dataset preparation and trustworthy splits remain substantial costs.

Current success is limited to implementation: five tools plus FINAL, synchronized contracts and real-backbone functional transport. New head training, world-model training and benchmark performance have not been demonstrated. Comparative speed measurement is currently deferred; shared-GPU functionality checks are authorized.

Further reading with primary links, feasibility and learning checklists:

- [Direct control and related work](JEV_RELATED_DIRECT_CONTROL_20261015.md)
- [Methods, models and benchmark baselines](JEV_RELATED_BASELINES_20261015.md)
- [Heavy visual evidence and benchmark audit](JEV_VISUAL_ACQUISITION_REVIEW_20261015.md)
- [Surprise and world-model review](JEV_SURPRISE_WORLD_MODEL_REVIEW_20261015.md)
- [Implementation](JEV_DIRECT_ARAG_IMPLEMENTATION.md) and [Astra verification](JEV_DIRECT_ARAG_ASTRA_REVIEW.md)
