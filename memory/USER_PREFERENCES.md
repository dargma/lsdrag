# Durable user preferences

- Discuss the project in Korean; be candid about novelty and experimental evidence.
- Single NVIDIA RTX PRO 6000 Blackwell Server Edition, about96GB. Shared GPU use for functional/quality experiments is authorized within available VRAM. Speed measurement is not currently required.
- Execute code, models, datasets and Git in local fast. Download public data directly from the web. Drive is storage/backup; avoid thousands of per-file Drive writes. Bundle phase artifacts into archives.
- User explicitly authorizes code changes, staged experiments, Git commits/pushes, parallel agents and Astra review. Avoid repeated permission questions and planning without execution.
- Active priority: test the decision head first. World-model/observation-prediction/surprise ideas are deferred. Do not combine unrelated directions into a kitchen-sink method.
- Preserve original A-RAG intent, jev_v1.0 multimodal extension and current direct dispatch. Include original and new tools; synchronize actual instruction, schema, head and function binding.
- Previous reduced text/image/stop measurements were discarded by the user. Inputs may be reused with fresh full-tool labels; old predictions/features/policy labels and conclusions must not be reused.
- Research should test decision quality, why it works and later agent transfer. Direct tool calling itself is not a novelty claim. Give primary-paper links, strong baseline/benchmark recommendations and a learning checklist.
- Keep persistent Git memory especially carefully; loss of conversation recovery has repeatedly frustrated the user.
