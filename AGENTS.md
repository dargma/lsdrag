# Working rules (v2.0)

- Scope: reproduce the head-policy conclusion only (`docs/JEV_HEAD_POLICY_CONCLUSION_20261028.md`). Follow `docs/REPRODUCE_HEAD_POLICY.md`.
- Work and manage Git in `/content/local_fast/lsdrag-code`. Push to branch `jev_v2.0`. Earlier experiments and repository memory live on `jev_v1.1`.
- Get models and data from their public URLs into `/content/local_fast`. Drive is backup only.
- Do not commit model weights, credentials, private tokens or bulky raw datasets.
- Do not edit the frozen records under `lab/registry/decision_shared_*_20261023/`. Write regenerated specs to `lab/registry/repro_head_policy/`.
- Do not terminate unrelated GPU jobs.
- Report reproduction numbers next to the reference table and say which steps were actually run.
