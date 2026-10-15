# Reader quality and modality recall — pending diagnostic, saved at user pause

User raised the hypothesis that weak reader VLM quality, rather than the decision module alone, explains poor answers, and requested separate text/image recall. This is a hypothesis, not an established finding. No additional model evaluation was started for this proposal.

## Read-only exploratory observations

The side conversation inspected budget256 paired20, not the unfinished common1024 experiment. Modalities overlap: text2 questions, table15, image16. Whole-trajectory supporting-source-document macro retrieval recall: text1.0 for all policies; table0.9333 for all; image0.4479 native and0.5729 generated/first_token. Image delivered-source recall0.4479 for all. These are source-document proxies, NOT fixed-top-K recall or proof of exact evidence delivery. Search snippets can supply text/table evidence without an explicit read call.

A conservative visual alignment subset covered only7/16 image-bearing questions after excluding IDs with differing accepted-figure sets across review files. Verified visual candidate/delivery hits were4/7 native,5/7 generated and first_token. Differing sets require reconciliation; they are not automatically invalid mappings. The selected7 cannot estimate whole-sample recall. Exact text/table recall requires supporting-span alignment. These exploratory figures must be regenerated from backed-up traces and alignment reviews before use as formal results.

## Proposed follow-up after explicit resume

1. Finish and audit the frozen common1024 paired20 diagnostic to separate output truncation from evidence/reader failures. Keep budget contracts separate.
2. Freeze questions, ranking and context budget. Report text-span, table-cell and actual-image recall separately at explicit K values, including denominators, alignment coverage, joint required-evidence coverage and actual reader delivery. Distinguish first search, accumulated retrieval and final reader inputs.
3. Hold evidence fixed and cross current retrieved evidence vs verified gold evidence with current reader vs a stronger reader. Use identical questions, answer scoring and explicit input/output budgets. Report per-modality answer quality, answer-format failures and evidence-sufficient subsets. This distinguishes retrieval failures from reader limitations; no stronger model is selected or downloaded yet.
4. Reassess decision-module improvements using answer quality and total token/tool cost. Do not claim speed gains from shared-GPU timings. Argument minimization remains an independent later ablation.

Existing execution contract remains in JEV_DECISION_RAG_EXECUTION_20261015.md. This proposal is saved for review; user pause overrides all automatic continuation.
