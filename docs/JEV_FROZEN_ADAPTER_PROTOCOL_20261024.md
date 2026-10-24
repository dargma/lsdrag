# Frozen-base adapter isolation — 2026-10-24

Prior full-head continuation lost quality even without a structure change. Preserve d987 and freeze every base parameter/buffer. Train only fresh zero-initialized residual depth, question-attention, or action-attention modules. Same modules/init seed20261061,377train states/train192,100epochs,AdamW1e-4,gradientclip1,Q=F1−.03mixedcost,KL.02,entropy.01. No early checkpoint selection or reward changes. AdamW applies only to trainable adapter weights. Same initial logits must equal d987; final everybase tensor must equal d987, allweights finite, probabilitydrift recorded. Freezing base does NOT guarantee unchanged decisions because residual modules can override logits.

Inference uses existing loaders; no runtime/protocol/schema/generation changes. Evaluate4×64 (reference plus3adapters), compare against archived same-architecture full-train runs and originalreference; identicalquestions/corpus/nativecontract. Max2400seconds,generated800000,input30000000. No fresh test access. Paired uncertainty, loss/entropy/zero-tool counts remain exploratory. No morearchitecture search until this isolate comparison is inspected. GoalACTIVE; no success claim.

Input/source hashes in lab/registry/decision_frozen_adapters_20261024/plan.json frozen beforefit. Exactsource unchanged forpriorruns. Newnamespace preventsoverwrite.
