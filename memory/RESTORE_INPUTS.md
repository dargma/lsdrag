# Full-head input recovery after an ephemeral runtime reset

Use the Git checkout in `/content/local_fast/lsdrag-code` for code and execution. Public model/data downloads go directly from their official web sources to `/content/local_fast`; Drive is a backup source, not an execution workspace. Do not restore historical predictions, policy labels, discarded three-action results, or world-model artifacts.

## Exact assets and paths

The fixed full-tool pilot uses:

- Corpus: `/content/local_fast/lsdrag-work/data/ir_final288` — 288 documents, 56,822 chunks, 2,438 figure JPGs.
- Composite Gemma indexes: `/content/local_fast/lsdrag-work/indexes_final288/{gemma_text,gemma_image}` and their original `indexes_development` / `indexes_final_additions` parts.
- Question pool:124 officially annotated MMQA-dev questions whose full support plus table source IDs are covered by this already fixed corpus. There are70 support-plus-table source components.
- Fixed seed20261016 yields ordered train100 and dev20; first60/80/100 are the predetermined collection prefixes. Full train overlaps51 prior100 questions; dev overlaps0. It is a shared previously used corpus pilot, not an official test or strict unseen-source evaluation.

The absolute `/content/local_fast/lsdrag-work` root is intentional: figures and composite index manifests contain these paths. Rewriting paths changes the frozen catalog/manifest identity. Keep this input root separate from the new run/results root.

## Restore the minimal derived inputs

```bash
cd /content/local_fast/lsdrag-code
python scripts/restore_full_head_inputs.py \
  --backup /content/drive/MyDrive/lsdrag-backups/20261013 \
  --report /content/local_fast/lsdrag-head-v12/recovery \
  --workers 12 --dry-run

python scripts/restore_full_head_inputs.py \
  --backup /content/drive/MyDrive/lsdrag-backups/20261013 \
  --report /content/local_fast/lsdrag-head-v12/recovery \
  --workers 12
```

The planned selection is2,463 files,1,060,223,873 bytes: only final288 IR, Gemma composite manifests and constituent Gemma index files, the original source IR files needed by those indexes, exactly2,438 referenced pixels, and the prior100 **question-only** file. It excludes Qwen indexes, unreferenced crops, PDFs, old predictions, and old labels. Existing byte-identical files are verified and reused; a conflicting existing file is rejected instead of overwritten. Backup manifests and pixel hashes are verified. Publication of the original source PDF does not reproduce the exact Mineru-extracted JPG bytes; those derived crops/indexes are the justified Drive-only exception.

The script fetches public `MMQA_dev.jsonl.gz` directly from the official repository pinned to revision `4dd14328c6d02a4daa357cc6032915a0b14602e3`, requiring SHA256 `2e348ca574b2dc368e84671709689070f943a59f2e08e6b6e374705deb712d31`. Default local annotation cache is `/content/local_fast/lsdrag-head-v12/source`.

## Reproduce the same questions and private scoring gold

Always write into a new directory. The active dataset manifest and `asset_readiness.json` must remain unchanged during or after a run.

```bash
python scripts/prepare_full_head_inputs.py \
  --output /content/local_fast/lsdrag-head-v12/recovered-data
```

The preparer reads restored corpus document aliases and the question-only prior100 file, recreates whole-component development exclusion, and verifies pinned hashes for train/dev questions, combined gold, and source groups. `--dry-run` builds in a temporary local-fast directory. To verify against a surviving dataset:

```bash
python scripts/prepare_full_head_inputs.py --dry-run \
  --compare-to /content/local_fast/lsdrag-head-v12/data
```

The CLI also accepts `--corpus`, `--indexes`, `--prior-questions`, and `--cache` when those read-only source files already exist. It refuses an output directory containing `manifest.json`. It does not alter the original source question data or corpus to improve retrieval coverage.

Output public inputs contain only `qid` and `question`. `gold.jsonl` contains scoring-only `answers: list[str]`, source group, and support document IDs; `*_gold_private.jsonl` retain original annotations for private audits only. Do not put intermediate answers, gold support selection, or question-specific document scope into the agent runtime. The global corpus remains the same for every question.

Known exact hashes:

| File | SHA256 |
|---|---|
| train_questions.jsonl | `1c91e0797e0699b3de6d9ed240a9c984aa26408cb6ee7ba802c094671e1ed79e` |
| dev_questions.jsonl | `62a6895c897a9a4c2950c9f557a8474c935ae178f07ab22f7d413de15040c76d` |
| gold.jsonl | `ceb20b6d7d7ebca04945e47e7cba7454c05832623d3b26aa9cacd3e79f581de7` |
| source_groups.jsonl | `ea7ea50ebaa76934c071c0d196ecd712290c10a3d662a39f8a53e8c52cc38757` |

`reproduced_manifest.json` intentionally records new recovery provenance. It is **not a byte-identical replacement** for the frozen original dataset manifest, which additionally contains the original postverification asset-readiness fingerprint. Recover the original registered identity from `lab/registry/full_head_20261015/data_manifest.json` and `asset_readiness.json` when resuming an existing run. Verify every original file hash and original model/code/tool contract before using an old result/checkpoint. For a new run, freeze a new provenance manifest explicitly rather than silently changing an active identity.

## Model, scorer and run provenance

These two CLIs restore inputs only; they do not install software, run the GPU, collect teacher labels, train, or resume evaluation. Obtain the frozen Gemma/embedding model snapshots from the official web hosts using the session's pinned revisions and model hashes; existing local assets may be reused. The original Gemma revision is `ee0ef6023621cff504d758262d4e04895a5af4a2`; embedding revision is `914f7f89142e33e77833254d9c9b90c3cef7303b`.

Official MMQA scorer sources and their hashes are defined by `src/lsdrag/v11_scoring.py`; install its CPU dependencies and preserve the external source directory. The original scorer's seven list-order/numeric/partial/empty-answer comparisons with pinned Bloomberg M3DocRAG evaluation matched; that validates those cases, not every imaginable input.

The original first full-head teacher attempt `full_head_20261015` had selector-format failures. Its labels must not be merged into corrected `full_head_20261015_v2`. Preserve invalid attempts as audit history and follow the current frozen plan/review before resuming. Three BC seeds are20261015/20261016/20261017;20 teacher questions are diagnostic, not permission to claim trained six-action coverage. No world model is part of this experiment.

Verification performed when adding these CLIs: CPU compilation, restore dry-run selection audit, and temporary preparation compared all10 JSONL outputs to the active data directory byte-for-byte. The active data manifest, asset-readiness fingerprint, GPU run, and model code were not modified.
